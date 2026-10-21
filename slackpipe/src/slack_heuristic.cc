#include "slackpipe/slack_heuristic.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>

#include "heuristic_internal.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/io.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/workload_bound.h"

namespace slackpipe::heuristic_internal {

std::vector<Tick> PhaseCosts(const Instance& in, const Split& p) {
  ValidateSplit(in, p);
  std::vector<Tick> costs(2 * in.stages);
  Index begin = 0;
  for (Index s = 0; s < in.stages; ++s) {
    const Index end = CheckedAdd(begin, p[s], "heuristic layer boundary");
    // Instance already holds rounded integer prefixes, slopes and role biases.
    // Recompute all ranges: nonadjacent transfers move intermediate boundaries.
    costs[s] = in.ForwardDuration(s, begin, end);
    costs[2 * in.stages - 1 - s] = in.BackwardDuration(s, begin, end);
    begin = end;
  }
  return costs;
}

std::shared_ptr<const Graph> Compile(const Instance& in, const MachineOrders& orders) {
  const Index q = 2 * in.stages, count = in.OperationCount();
  if (orders.size() != static_cast<std::size_t>(in.workers)) return {};
  auto graph = std::make_shared<Graph>();
  graph->orders = orders;
  graph->edges.resize(count);
  graph->degree.assign(count, 0);
  graph->topo.reserve(count);
  std::vector<bool> seen(count);
  for (Index w = 0; w < in.workers; ++w) {
    std::vector<Index> last(q, -1);
    for (auto op : orders[w]) {
      const Index id = op.value;
      if (id < 0 || id >= count || seen[id]) return {};
      const Index phase = id % q, b = id / q;
      if (in.WorkerForStage(phase < in.stages ? phase : q - 1 - phase) != w ||
          b <= last[phase]) return {};
      seen[id] = true;
      last[phase] = b;
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) return {};
  std::vector<int> incoming(count);
  auto edge = [&](Index u, Index v, Tick lag) {
    graph->edges[u][graph->degree[u]++] = {v, lag};
    ++incoming[v];
  };
  for (Index id = 0; id < count; ++id) {
    if (id % q + 1 < q) {
      auto worker = [&](Index phase) {
        return in.WorkerForStage(phase < in.stages ? phase : q - 1 - phase);
      };
      edge(id, id + 1, in.EdgeDelay(worker(id % q), worker(id % q + 1)));
    }
    if (id + q < count) edge(id, id + q, 0);
  }
  for (const auto& worker : orders)
    for (std::size_t k = 1; k < worker.size(); ++k)
      edge(worker[k - 1].value, worker[k].value, 0);
  for (Index id = 0; id < count; ++id)
    if (!incoming[id]) graph->topo.push_back(id);
  for (std::size_t k = 0; k < graph->topo.size(); ++k) {
    const Index u = graph->topo[k];
    for (int j = 0; j < graph->degree[u]; ++j) {
      const Index v = graph->edges[u][j].target;
      if (!--incoming[v]) graph->topo.push_back(v);
    }
  }
  if (graph->topo.size() != static_cast<std::size_t>(count)) return {};
  return graph;
}

std::optional<Pair> Replay(const Instance& in, const Split& p,
                           std::shared_ptr<const Graph> graph,
                           const std::vector<Tick>& durations) {
  ValidateSplit(in, p);
  if (!graph) return std::nullopt;
  const Index q = 2 * in.stages, count = in.OperationCount();
  if (durations.size() != static_cast<std::size_t>(q))
    throw Error("heuristic phase cost size mismatch");
  Pair c;
  c.partition = p;
  c.graph = std::move(graph);
  c.duration = durations;
  c.start.assign(count, 0);
  c.end.resize(count);
  for (Index u : c.graph->topo) {
    if (durations[u % q] < 0) throw Error("heuristic negative duration");
    c.end[u] = CheckedAdd(c.start[u], durations[u % q], "heuristic replay end");
    c.makespan = std::max(c.makespan, c.end[u]);
    if (u % q >= in.stages)
      c.backward_sum = CheckedAdd(c.backward_sum, c.end[u], "heuristic backward completion sum");
    for (int j = 0; j < c.graph->degree[u]; ++j) {
      auto [v, lag] = c.graph->edges[u][j];
      c.start[v] = std::max(c.start[v], CheckedAdd(c.end[u], lag, "heuristic edge lag"));
    }
  }
  return c;
}

Float TotalFloat(const Pair& c) {
  const Index count = c.end.size(), q = c.duration.size();
  Float f;
  f.slack.resize(count);
  f.critical.assign(q, 0);
  std::vector<Tick> tail(count);
  for (auto it = c.graph->topo.rbegin(); it != c.graph->topo.rend(); ++it) {
    Index u = *it;
    for (int j = 0; j < c.graph->degree[u]; ++j) {
      auto [v, lag] = c.graph->edges[u][j];
      tail[u] = std::max(tail[u], CheckedAdd(lag,
          CheckedAdd(c.duration[v % q], tail[v], "heuristic tail"), "heuristic tail lag"));
    }
    f.slack[u] = c.makespan - CheckedAdd(c.end[u], tail[u], "heuristic end plus tail");
    if (f.slack[u] < 0) throw Error("heuristic negative total float");
    if (!f.slack[u]) ++f.critical[u % q];
  }
  return f;
}

std::string ImprovementReason(const Pair& a, const Pair& b) {
  if (a.makespan != b.makespan) return a.makespan < b.makespan ? "makespan" : "";
  if (a.backward_sum != b.backward_sum)
    return a.backward_sum < b.backward_sum ? "backward_completion_sum" : "";
  if (a.partition != b.partition) return a.partition < b.partition ? "partition_lex" : "";
  return a.graph->orders < b.graph->orders ? "order_lex" : "";
}
bool Better(const Pair& a, const Pair& b) {
  return std::tie(a.makespan, a.backward_sum, a.partition, a.graph->orders) <
         std::tie(b.makespan, b.backward_sum, b.partition, b.graph->orders);
}
Index AcceptChild(const Pair& child, Pair& parent, Index stale) {
  if (child.makespan < parent.makespan) {
    parent = child;
    return 0;
  }
  if (Better(child, parent)) parent = child;
  return CheckedAdd(stale, 1, "heuristic stale");
}

bool ApplyOrder(const Instance& in, const OrderAction& a, MachineOrders& orders) {
  const Index q = 2 * in.stages, count = in.OperationCount();
  if (a.worker < 0 || a.worker >= in.workers ||
      a.source < 0 || a.source >= count || a.anchor < -1 || a.anchor >= count ||
      orders.size() != static_cast<std::size_t>(in.workers)) return false;
  auto next = orders[a.worker];
  auto move = [&](Index source, Index anchor) {
    auto it = std::find(next.begin(), next.end(), OperationId{source});
    if (it == next.end() || source == anchor) return false;
    next.erase(it);
    auto at = anchor < 0 ? next.end() : std::find(next.begin(), next.end(), OperationId{anchor});
    if (anchor >= 0 && at == next.end()) return false;
    next.insert(at, {source});
    return true;
  };
  if (!a.stream) {
    if (!move(a.source, a.anchor)) return false;
  } else {
    // A forced END stream is a no-op, never a stream-to-end transform.
    if (a.anchor < 0) return in.microbatches <= 64;
    const Index source_q = a.source % q, anchor_q = a.anchor % q;
    const Index offset = a.anchor / q - a.source / q;
    if (in.microbatches > 64 && source_q == anchor_q) return false;
    if (source_q != anchor_q) {
      std::vector<OperationId> rebuilt;
      rebuilt.reserve(next.size());
      std::vector<bool> present(count);
      for (auto op : next) {
        if (op.value < 0 || op.value >= count || present[op.value]) return false;
        present[op.value] = true;
      }
      for (Index b = 0; b < in.microbatches; ++b)
        if (b + offset >= 0 && b + offset < in.microbatches &&
            (!present[b * q + source_q] || !present[(b + offset) * q + anchor_q]))
          return false;
      for (auto op : next) {
        const Index b = op.value / q, phase = op.value % q;
        if (phase == source_q && b + offset >= 0 && b + offset < in.microbatches) continue;
        if (phase == anchor_q && b - offset >= 0 && b - offset < in.microbatches)
          rebuilt.push_back({(b - offset) * q + source_q});
        rebuilt.push_back(op);
      }
      next = std::move(rebuilt);
    } else {
      for (Index b = 0; b < in.microbatches; ++b)
        if (b + offset >= 0 && b + offset < in.microbatches &&
            !move(b * q + source_q, (b + offset) * q + anchor_q)) return false;
    }
  }
  if (next.size() != orders[a.worker].size()) return false;
  std::vector<bool> membership(count);
  for (auto op : orders[a.worker]) {
    if (op.value < 0 || op.value >= count || membership[op.value]) return false;
    membership[op.value] = true;
  }
  std::vector<Index> last(q, -1);
  for (auto op : next) {
    if (op.value < 0 || op.value >= count || !membership[op.value]) return false;
    membership[op.value] = false;
    const Index phase = op.value % q;
    if (in.WorkerForStage(phase < in.stages ? phase : q - 1 - phase) != a.worker ||
        op.value / q <= last[phase]) return false;
    last[phase] = op.value / q;
  }
  // Equal size plus one-time membership checks preserve the entire permutation.
  orders[a.worker] = std::move(next);
  return true;
}

std::vector<OrderAction> OrderActions(const Instance& in, const Pair& c,
                                     const Float& f, Index round,
                                     const std::function<bool()>& stopped) {
  const Index q = 2 * in.stages;
  std::vector<OrderAction> singles, streams, out;
  for (Index w = 0; w < in.workers; ++w) {
    const auto& order = c.graph->orders[w];
    for (Index a = 0; a < static_cast<Index>(order.size()); ++a) {
      if (stopped && stopped()) return {};
      for (Index delta : {-8, -4, -3, -2, -1, 1, 2, 3, 4, 8}) {
        const Index j = a + delta;
        if (j < 0 || j >= static_cast<Index>(order.size())) continue;
        const Index u = order[a].value, phase = u % q;
        bool crossed = false, critical = false;
        for (Index k = std::min(a, j); k <= std::max(a, j); ++k) {
          crossed |= k != a && order[k].value % q == phase;
          critical |= f.slack[order[k].value] == 0;
        }
        if (crossed || !critical) continue;
        Tick ready = 0;
        if (phase) {
          const Index previous = phase - 1;
          ready = CheckedAdd(c.end[u - 1], in.EdgeDelay(
              in.WorkerForStage(previous < in.stages ? previous : q - 1 - previous), w),
              "heuristic ready");
        }
        if (u >= q) ready = std::max(ready, c.end[u - q]);
        const Tick hole_start = j ? c.end[order[j - 1].value] : 0;
        const Tick available = std::max<Tick>(0, c.start[order[j].value] - std::max(hole_start, ready));
        OrderAction action;
        action.worker = w;
        action.source = u;
        action.anchor = j < a ? order[j].value :
            (j + 1 < static_cast<Index>(order.size()) ? order[j + 1].value : -1);
        action.score = -(static_cast<long double>(std::min(c.duration[phase], available)) +
                         (f.slack[u] == 0 ? .25L * c.duration[phase] : 0));
        singles.push_back(action);
      }
    }
  }
  std::stable_sort(singles.begin(), singles.end(), [&](const auto& a, const auto& b) {
    if (a.score != b.score) return a.score < b.score;
    if (in.stages <= 16) return false;
    const Index shift = 17 * (round % q);
    return (a.source % q + shift) % q < (b.source % q + shift) % q;
  });
  std::set<std::tuple<Index, Index, Index>> seen;
  for (auto a : singles)
    if (a.anchor >= 0 && seen.emplace(a.source % q, a.anchor % q,
                                     a.anchor / q - a.source / q).second) {
      a.stream = true;
      streams.push_back(a);
    }
  out.reserve(singles.size() + streams.size());
  for (std::size_t k = 0; k < std::max(singles.size(), streams.size()); ++k) {
    if (k < streams.size()) out.push_back(streams[k]);
    if (k < singles.size()) out.push_back(singles[k]);
  }
  return out;
}
}  // namespace slackpipe::heuristic_internal

namespace slackpipe {
namespace {
using namespace heuristic_internal;
using Clock = Deadline::Clock;

struct PartitionAction {
  Index donor, receiver;
  Tick step;
  long double score;
};
struct Event {
  std::string kind, reason;
  double seconds = 0;
  Index evaluation = 0, round = 0, stale_before = 0, stale_after = 0;
  Tick before = -1, perturbation = -1, child = -1;
  Tick before_backward_sum = -1, child_backward_sum = -1;
  Index trial_evaluations = 0;
  bool refined = false, accepted = false;
};

class HeuristicSearch {
 public:
  HeuristicSearch(const Instance& in, const SlackHeuristicOptions& options)
      : in_(in), options_(options), started_(Clock::now()), n_(in.stages),
        q_(2 * n_), rng_(20261005ULL + static_cast<std::uint64_t>(options.random_seed)) {
    allowance_ = options.time_limit_seconds;
    if (options.parent_deadline && options.parent_deadline->bounded())
      allowance_ = std::min(allowance_, options.parent_deadline->remaining_seconds());
    reserve_ = std::min(1.0, .05 * allowance_);
    search_end_ = std::max(0.0, allowance_ - reserve_);
    init_end_ = std::min(2.0, .2 * search_end_);
    init_cap_ = std::min<Index>(32768, options.evaluation_limit);
    cache_limit_ = std::max<std::size_t>(1, 4194304 / (3 * n_ * sizeof(Tick)));
  }

  SearchResult Run();

 private:
  double Elapsed() const { return std::chrono::duration<double>(Clock::now() - started_).count(); }
  bool BoundHit() const { return best_ && bound_ && best_->makespan == *bound_; }
  bool GlobalStopped() const {
    return BoundHit() || evaluations_ >= options_.evaluation_limit ||
        (!options_.fixed_evaluation_test && (Elapsed() >= search_end_ ||
          (options_.parent_deadline && options_.parent_deadline->expired())));
  }
  bool Stopped() const {
    return GlobalStopped() || (initial_ && (evaluations_ >= init_cap_ ||
        (!options_.fixed_evaluation_test && Elapsed() >= init_end_)));
  }
  bool LocalStopped() const {
    return Stopped() || evaluations_ >= local_cap_ ||
        (!options_.fixed_evaluation_test && Elapsed() >= local_end_);
  }
  void Limit(Index cap, double time) { local_cap_ = cap; local_end_ = time; }
  void ClearLimit() { Limit(kTickMax, std::numeric_limits<double>::infinity()); }
  Index Cutoff(Index allowance) const {
    return evaluations_ + std::min(allowance, options_.evaluation_limit - evaluations_);
  }
  std::uint64_t Random() {
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 7;
    rng_ ^= rng_ << 17;
    return rng_;
  }
  bool Allowed(const Split& p) const {
    if (p.size() != static_cast<std::size_t>(n_)) return false;
    Tick sum = 0;
    for (Tick length : p) {
      if (length < in_.min_layers || length > in_.total_layers - sum) return false;
      sum += length;
    }
    return sum == in_.total_layers &&
        (!options_.allowed_partition || options_.allowed_partition(p));
  }
  const std::vector<Tick>& Costs(const Split& p) {
    auto it = costs_.find(p);
    if (it != costs_.end()) { ++cache_hits_; return it->second; }
    ++cache_misses_;
    if (costs_.size() >= cache_limit_) costs_.clear();
    return costs_.emplace(p, PhaseCosts(in_, p)).first->second;
  }
  void Verify(const Pair& c, const ScheduleSolution& schedule) const {
    if (!schedule.ok() || schedule.makespan != c.makespan ||
        schedule.operations_by_id.size() != c.end.size()) throw Error("heuristic general replay mismatch");
    for (std::size_t i = 0; i < c.end.size(); ++i)
      if (c.start[i] != schedule.operations_by_id[i].start ||
          c.end[i] != schedule.operations_by_id[i].end)
        throw Error("heuristic cached operation times mismatch");
  }
  std::optional<Pair> Evaluate(const Split& p, const MachineOrders& o,
                              std::shared_ptr<const Graph> reuse = {}) {
    ++evaluations_;
    if (!Allowed(p)) return {};
    std::shared_ptr<const Graph> graph;
    if (reuse && reuse->orders == o) { graph = std::move(reuse); ++graph_reuses_; }
    else { graph = Compile(in_, o); ++graph_compiles_; }
    auto c = Replay(in_, p, graph, Costs(p));
    if (options_.verify_cached_replay) {
      ++verification_calls_;
      auto general = EvaluateSchedule(in_, p, o);
      if (c) Verify(*c, general.schedule);
      else if (general.schedule.ok()) throw Error("heuristic feasibility mismatch");
    }
    if (c && bound_ && *bound_ > c->makespan)
      throw Error("heuristic lower bound exceeds feasible makespan");
    // Deliberately no parent mutation here.
    return c;
  }
  void Record(Event event) {
    if (!options_.record_trace) return;
    event.seconds = Elapsed();
    event.evaluation = evaluations_;
    event.round = rounds_;
    events_.push_back(std::move(event));
  }
  void InitializationCandidate(const Pair& c) {
    if (!best_ || Better(c, *best_)) {
      if (!best_ || c.makespan < best_->makespan) trace_.emplace_back(Elapsed(), c.makespan);
      else {
        Event event;
        event.kind = "initialization_equal";
        event.reason = ImprovementReason(c, *best_);
        event.before = best_->makespan;
        event.child = c.makespan;
        event.before_backward_sum = best_->backward_sum;
        event.child_backward_sum = c.backward_sum;
        event.accepted = true;
        Record(std::move(event));
      }
      best_ = c;
    }
  }
  MachineOrders Structured(const Split&, double stride, int shape);
  void Seeds(const Pair&, std::vector<Pair>&);
  std::vector<PartitionAction> PartitionActions(const Pair&, const Float&);
  Pair Local(Pair current, bool pd);
  void Initialize();
  void Outer();
  std::string Metadata() const;

  const Instance& in_;
  SlackHeuristicOptions options_;
  Clock::time_point started_;
  Index n_, q_;
  std::uint64_t rng_;
  double allowance_ = 0, reserve_ = 0, search_end_ = 0, init_end_ = 0;
  double local_end_ = std::numeric_limits<double>::infinity();
  Index local_cap_ = kTickMax, init_cap_ = 0;
  bool initial_ = true;
  std::size_t cache_limit_ = 1;
  std::map<Split, std::vector<Tick>> costs_;
  std::optional<Tick> bound_;
  std::optional<Pair> best_;
  std::vector<std::pair<double, Tick>> trace_;
  std::vector<Event> events_;
  std::string initialization_json_;
  Index evaluations_ = 0, rounds_ = 0, generated_ = 0, outer_iterations_ = 0, stale_ = 0;
  Index cache_hits_ = 0, cache_misses_ = 0, graph_compiles_ = 0, graph_reuses_ = 0, verification_calls_ = 0;
  Index pd_init_calls_ = 0, pd_outer_calls_ = 0, refine_init_calls_ = 0, refine_outer_calls_ = 0;
  Index pd_evaluations_ = 0, refine_evaluations_ = 0, partition_attempts_ = 0, order_attempts_ = 0;
  Index accepted_partition_ = 0, accepted_single_ = 0, accepted_stream_ = 0, feasible_trials_ = 0, worse_trials_ = 0;
  Index strict_outer_accepts_ = 0, equal_outer_accepts_ = 0, init_evaluations_ = 0;
  Tick baseline_ = 0, init_makespan_ = 0;
  double init_seconds_ = 0, pd_seconds_ = 0, refine_seconds_ = 0;
};

MachineOrders HeuristicSearch::Structured(const Split& p, double stride, int shape) {
  const auto d = Costs(p);
  const long double mean = std::accumulate(d.begin(), d.end(), 0.L) / q_;
  std::vector<long double> offset(q_, 0);
  for (Index q = 1; q < q_; ++q)
    offset[q] = offset[q - 1] + (shape && mean > 0 ? d[q - 1] / mean : 1.L);
  MachineOrders orders(in_.workers);
  const Index count = in_.OperationCount();
  for (auto& worker : orders) worker.reserve(count / in_.workers + q_);
  for (Index id = 0; id < count; ++id) {
    const Index q = id % q_, s = q < n_ ? q : q_ - 1 - q;
    orders[in_.WorkerForStage(s)].push_back({id});
  }
  for (auto& worker : orders)
    std::stable_sort(worker.begin(), worker.end(), [&](auto a, auto b) {
      return std::tuple(stride * (a.value / q_) + offset[a.value % q_], -(a.value % q_), a.value) <
             std::tuple(stride * (b.value / q_) + offset[b.value % q_], -(b.value % q_), b.value);
    });
  return orders;
}

void HeuristicSearch::Seeds(const Pair& base, std::vector<Pair>& seeds) {
  std::set<std::pair<Split, MachineOrders>> seen;
  seen.emplace(base.partition, base.graph->orders);
  auto add = [&](const Split& p, double stride, int shape) {
    if (Stopped() || !Allowed(p)) return;
    auto orders = Structured(p, stride, shape);
    if (Stopped() || !seen.emplace(p, orders).second) return;
    ++generated_;
    if (auto c = Evaluate(p, orders)) {
      InitializationCandidate(*c);
      seeds.push_back(std::move(*c));
    }
  };
  if (n_ > 16 && n_ >= 2 * in_.workers && n_ % in_.workers == 0) {
    const Tick maximum = std::max<Tick>(1, in_.total_layers / n_ / 2);
    for (Tick d = 1; d <= maximum && !Stopped(); d *= 2) {
      for (Tick sign : {1, -1}) {
        auto p = base.partition;
        for (Index w = 0; w < in_.workers; ++w) {
          p[w] += sign * d;
          p[n_ - in_.workers + w] -= sign * d;
        }
        for (double stride : {.5, 1., 1.5, 2., 4.}) add(p, stride, 0);
      }
      if (d > maximum / 2) break;
    }
  }
  for (int shape : {0, 1})
    for (double stride : {0., .25, .5, .75, 1., 1.5, 2., 4.})
      add(base.partition, stride, shape);
  for (double beta : {.5, 1., 1.5}) {
    if (Stopped()) break;
    Split p(n_, in_.min_layers);
    std::vector<long double> weight(n_);
    using Entry = std::pair<long double, Index>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (Index s = 0; s < n_; ++s) {
      const long double t = static_cast<long double>(s % in_.workers + 1) / in_.workers;
      weight[s] = std::max(.1L, 1.L + beta * ((s / in_.workers) % 2 == 0 ? t - 1 : 1 - t));
      heap.emplace(p[s] / weight[s], s);
    }
    for (Tick left = in_.total_layers - n_ * in_.min_layers; left > 0; --left) {
      if (Stopped()) return;
      const Index s = heap.top().second;
      heap.pop();
      ++p[s];
      heap.emplace(p[s] / weight[s], s);
    }
    for (double stride : {0., .5, 1., 2.}) add(p, stride, 0);
  }
}

std::vector<PartitionAction> HeuristicSearch::PartitionActions(const Pair& c, const Float& f) {
  std::vector<PartitionAction> actions;
  actions.reserve(n_ * (n_ <= 16 ? n_ : 8) * 4);
  const auto old = c.duration;
  std::vector<Index> least(n_);
  std::iota(least.begin(), least.end(), 0);
  std::stable_sort(least.begin(), least.end(), [&](Index a, Index b) {
    return f.critical[a] + f.critical[q_ - 1 - a] < f.critical[b] + f.critical[q_ - 1 - b];
  });
  std::vector<Tick> steps{1};
  const Tick maximum = std::max<Tick>(2, in_.total_layers / n_ / 2);
  for (Tick d = 2; d <= maximum; d *= 2) {
    steps.push_back(d);
    if (d > maximum / 2) break;
  }
  for (Index a = 0; a < n_ && !LocalStopped(); ++a) {
    std::set<Index> receivers;
    if (n_ <= 16) {
      for (Index b = 0; b < n_; ++b) if (a != b) receivers.insert(b);
    } else {
      for (Index b : {a - 1, a + 1, a - in_.workers, a + in_.workers})
        if (b >= 0 && b < n_ && b != a) receivers.insert(b);
      for (Index k = 0; k < std::min<Index>(4, n_); ++k)
        if (least[k] != a) receivers.insert(least[k]);
    }
    for (Index b : receivers) {
      for (Tick d : steps) {
        if (LocalStopped()) break;
        if (c.partition[a] - d < in_.min_layers) continue;
        auto p = c.partition;
        p[a] -= d;
        p[b] += d;
        if (!Allowed(p)) continue;
        long double score = 0;
        if (n_ <= 16) {
          const auto costs = Costs(p);
          Tick positive = 0, negative = 0;
          for (Index q = 0; q < q_; ++q) {
            const Tick delta = costs[q] - old[q];
            Tick& sum = delta < 0 ? negative : positive;
            sum = CheckedAdd(sum, CheckedMul(delta < 0 ? -delta : delta, f.critical[q],
                                            "heuristic proposal score"), "heuristic score sum");
          }
          score = positive - negative;
        } else {
          auto marginal = [&](Index s) {
            return (static_cast<long double>(old[s]) * f.critical[s] +
                    static_cast<long double>(old[q_ - 1 - s]) * f.critical[q_ - 1 - s]) / c.partition[s];
          };
          score = d * (marginal(b) - marginal(a));
        }
        actions.push_back({a, b, d, score});
      }
    }
  }
  std::stable_sort(actions.begin(), actions.end(), [&](const auto& a, const auto& b) {
    if (a.score != b.score) return a.score < b.score;
    if (n_ <= 16) return false;
    const Index shift = 17 * (rounds_ % n_);
    return std::pair((a.donor + shift) % n_, a.receiver) <
           std::pair((b.donor + shift) % n_, b.receiver);
  });
  return actions;
}

Pair HeuristicSearch::Local(Pair current, bool pd) {
  const double started = Elapsed();
  const Index before = evaluations_;
  if (pd) { if (initial_) ++pd_init_calls_; else ++pd_outer_calls_; }
  else { if (initial_) ++refine_init_calls_; else ++refine_outer_calls_; }
  for (int round = 0; round < (pd ? 64 : 48) && !LocalStopped(); ++round) {
    rounds_ = CheckedAdd(rounds_, 1, "heuristic rounds");
    const auto f = TotalFloat(current);
    auto partitions = PartitionActions(current, f);
    // Both lists originate from exactly the same current pair.
    auto orders = pd || LocalStopped() ? std::vector<OrderAction>{} :
                                        OrderActions(in_, current, f, rounds_, [&] { return LocalStopped(); });
    Pair next = current;
    int winning_action = 0;
    std::size_t quota = pd ? (n_ <= 16 ? partitions.size() : 96) : (n_ <= 16 ? 192 : 64);
    for (std::size_t k = 0; k < std::min(quota, partitions.size()) && !LocalStopped(); ++k) {
      ++partition_attempts_;
      ++generated_;
      const auto& a = partitions[k];
      auto p = current.partition;
      p[a.donor] -= a.step;
      p[a.receiver] += a.step;
      if (auto c = Evaluate(p, current.graph->orders, current.graph)) {
        if (initial_) InitializationCandidate(*c);
        if (Better(*c, next)) { next = std::move(*c); winning_action = 1; }
      }
    }
    for (std::size_t k = 0; k < std::min<std::size_t>(96, orders.size()) && !LocalStopped(); ++k) {
      ++order_attempts_;
      ++generated_;
      auto o = current.graph->orders;
      if (!ApplyOrder(in_, orders[k], o)) continue;
      if (auto c = Evaluate(current.partition, o, current.graph)) {
        if (initial_) InitializationCandidate(*c);
        if (Better(*c, next)) {
          next = std::move(*c);
          winning_action = orders[k].stream ? 3 : 2;
        }
      }
    }
    if (!Better(next, current)) break;
    accepted_partition_ += winning_action == 1;
    accepted_single_ += winning_action == 2;
    accepted_stream_ += winning_action == 3;
    current = std::move(next);
  }
  if (pd) { pd_evaluations_ += evaluations_ - before; pd_seconds_ += Elapsed() - started; }
  else { refine_evaluations_ += evaluations_ - before; refine_seconds_ += Elapsed() - started; }
  return current;
}

void HeuristicSearch::Initialize() {
  auto p = UniformSplit(in_);
  if (!Allowed(p)) throw Error("heuristic uniform BFS is disallowed by enabled hard constraints");
  ++generated_;
  // Mandatory valid baseline even for a deadline reached during construction.
  auto base = Evaluate(p, BreadthFirstOrders(in_));
  if (!base) throw Error("heuristic has no legal BFS baseline");
  InitializationCandidate(*base);
  baseline_ = base->makespan;
  const auto workload = ComputeWorkloadLowerBound(in_);
  if (workload.makespan_ticks) bound_ = *workload.makespan_ticks;
  if (workload.total_work_ticks) {
    Tick chain = *workload.total_work_ticks / in_.microbatches;
    for (Index q = 1; q < q_; ++q) {
      auto worker = [&](Index phase) { return in_.WorkerForStage(phase < n_ ? phase : q_ - 1 - phase); };
      chain = CheckedAdd(chain, in_.EdgeDelay(worker(q - 1), worker(q)), "heuristic chain bound");
    }
    bound_ = std::max(bound_.value_or(0), chain);
  }
  if (bound_ && *bound_ > baseline_) throw Error("heuristic global bound exceeds baseline");
  std::vector<Pair> seeds{*base};
  Seeds(*base, seeds);
  const Index quota = std::max<Index>(1, (2 * init_cap_ / 5) / seeds.size());
  for (std::size_t k = 0; k < seeds.size() && !Stopped(); ++k) {
    const double t = Elapsed();
    Limit(std::min(init_cap_, Cutoff(quota)),
          t + std::max(.001, (init_end_ - t) * .5 / (seeds.size() - k)));
    seeds[k] = Local(seeds[k], true);
    ClearLimit();
  }
  std::stable_sort(seeds.begin(), seeds.end(), Better);
  std::set<std::pair<Split, MachineOrders>> refined;
  int used = 0;
  for (const auto& seed : seeds) {
    if (Stopped() || used == 4) break;
    if (!refined.emplace(seed.partition, seed.graph->orders).second) continue;
    const double t = Elapsed();
    Limit(std::min(init_cap_, Cutoff(std::max<Index>(1, init_cap_ / 9))),
          t + std::max(.001, (init_end_ - t) / (5 - used)));
    Local(seed, false);
    ++used;
    ClearLimit();
  }
  Limit(init_cap_, init_end_);
  if (!Stopped()) Local(*best_, false);
  ClearLimit();
  for (Index w = 0; w < in_.workers && !Stopped(); ++w) {
    auto o = best_->graph->orders;
    std::stable_sort(o[w].begin(), o[w].end(), [&](auto a, auto b) {
      return std::pair(a.value % q_, a.value / q_) < std::pair(b.value % q_, b.value / q_);
    });
    ++generated_;
    if (auto c = Evaluate(best_->partition, o, best_->graph)) {
      InitializationCandidate(*c);
      Limit(std::min(init_cap_, Cutoff(std::max<Index>(128, (init_cap_ - evaluations_) / (in_.workers - w)))),
            init_end_);
      Local(*c, false);
      ClearLimit();
    }
  }
  // The seed collection is destroyed here. It is never selectable outside init.
  init_evaluations_ = evaluations_;
  init_makespan_ = best_->makespan;
  auto checked = EvaluateSchedule(in_, best_->partition, best_->graph->orders);
  Verify(*best_, checked.schedule);
  ++verification_calls_;
  initialization_json_ = ToJson(in_, checked.schedule);
  init_seconds_ = Elapsed();
  Record({"initialization_end", ""});
  initial_ = false;
}

void HeuristicSearch::Outer() {
  while (!GlobalStopped()) {
    ++outer_iterations_;
    const Index h = 1 + (stale_ / 16) % 6;
    const Pair& parent = *best_;
    auto p = parent.partition;
    auto o = parent.graph->orders;
    Event event;
    event.kind = "outer";
    event.before = parent.makespan;
    event.before_backward_sum = parent.backward_sum;
    event.stale_before = stale_;
    for (Index k = 0; k < std::min<Index>(in_.workers, 8); ++k) {
      Index a = Random() % n_;
      Index b = a + in_.workers < n_ ? a + in_.workers : (a >= in_.workers ? a - in_.workers : a);
      if (a == b || p[a] <= in_.min_layers) continue;
      Tick d = 1 + Random() % std::min<Tick>(p[a] - in_.min_layers, h);
      p[a] -= d;
      p[b] += d;
    }
    const auto f = TotalFloat(parent);
    const auto actions = OrderActions(in_, parent, f, rounds_, [&] { return GlobalStopped(); });
    for (Index k = 0; k < h && !actions.empty(); ++k) {
      auto action = actions[Random() % std::min<std::size_t>(512, actions.size())];
      if (!k) action.stream = true;
      ApplyOrder(in_, action, o);  // Atomic rollback on failure.
    }
    if (GlobalStopped()) break;
    if (p == parent.partition && o == parent.graph->orders) {
      stale_ = CheckedAdd(stale_, 1, "heuristic stale");
      event.reason = "unchanged";
    } else {
      ++generated_;
      auto trial = Evaluate(p, o, parent.graph);
      if (!trial) {
        stale_ = CheckedAdd(stale_, 1, "heuristic stale");
        event.reason = "infeasible";
      } else {
        ++feasible_trials_;
        worse_trials_ += trial->makespan > parent.makespan;
        event.perturbation = trial->makespan;
        const Index before_refinement = evaluations_;
        Limit(Cutoff(n_ <= 16 ? 3072 : 1024), search_end_);
        if (options_.refinement_mode == "pd_then_refine") *trial = Local(*trial, true);
        // One shared cutoff. Even a worse feasible trial enters coupled refine.
        *trial = Local(*trial, false);
        ClearLimit();
        event.refined = true;
        event.trial_evaluations = evaluations_ - before_refinement;
        event.child = trial->makespan;
        event.child_backward_sum = trial->backward_sum;
        event.reason = ImprovementReason(*trial, parent);
        event.accepted = !event.reason.empty();
        if (trial->makespan < parent.makespan) {
          ++strict_outer_accepts_;
          trace_.emplace_back(Elapsed(), trial->makespan);
        } else if (event.accepted) ++equal_outer_accepts_;
        if (!event.accepted) event.reason = "rejected";
        stale_ = AcceptChild(*trial, *best_, stale_);
      }
    }
    event.stale_after = stale_;
    Record(std::move(event));
  }
}

std::string HeuristicSearch::Metadata() const {
  std::ostringstream out;
  out << std::setprecision(17) << std::boolalpha
      << "{\"schema_version\":\"slackpipe.heuristic.v1\",\"method\":\"slack-heuristic\""
      << ",\"refinement_mode\":\"" << options_.refinement_mode << "\""
      << ",\"seed\":" << options_.random_seed << ",\"search_threads\":1"
      << ",\"fixed_evaluation_test\":" << options_.fixed_evaluation_test
      << ",\"evaluation_limit\":" << options_.evaluation_limit
      << ",\"requested_seconds\":" << options_.time_limit_seconds
      << ",\"search_seconds_limit\":" << search_end_ << ",\"reserve_seconds\":" << reserve_
      << ",\"lower_bound_kind\":\"minimum_work_capacity_and_chain\""
      << ",\"initialization\":{\"evaluations\":" << init_evaluations_
      << ",\"elapsed_seconds\":" << init_seconds_ << ",\"makespan_ticks\":" << init_makespan_
      << ",\"plan\":" << initialization_json_ << "}";
  const std::pair<const char*, Index> counters[] = {
      {"evaluations", evaluations_}, {"local_rounds", rounds_}, {"outer_iterations", outer_iterations_},
      {"stale", stale_}, {"cost_cache_hits", cache_hits_}, {"cost_cache_misses", cache_misses_},
      {"graph_compiles", graph_compiles_}, {"graph_reuses", graph_reuses_}, {"verification_calls", verification_calls_},
      {"initialization_pd_calls", pd_init_calls_}, {"outer_pd_calls", pd_outer_calls_},
      {"initialization_refine_calls", refine_init_calls_}, {"outer_refine_calls", refine_outer_calls_},
      {"pd_evaluations", pd_evaluations_}, {"refine_evaluations", refine_evaluations_},
      {"partition_entries_attempted", partition_attempts_}, {"order_entries_attempted", order_attempts_},
      {"accepted_partition_actions", accepted_partition_}, {"accepted_single_actions", accepted_single_},
      {"accepted_stream_actions", accepted_stream_}, {"feasible_outer_trials", feasible_trials_},
      {"worse_feasible_outer_trials", worse_trials_}, {"strict_outer_accepts", strict_outer_accepts_},
      {"equal_outer_accepts", equal_outer_accepts_}};
  for (auto [name, value] : counters) out << ",\"" << name << "\":" << value;
  out << ",\"pd_seconds\":" << pd_seconds_ << ",\"refine_seconds\":" << refine_seconds_
      << ",\"trace_enabled\":" << options_.record_trace << ",\"events\":[";
  bool comma = false;
  for (const auto& e : events_) {
    if (comma) out << ',';
    comma = true;
    out << "{\"kind\":\"" << e.kind << "\",\"reason\":\"" << e.reason << "\""
        << ",\"seconds\":" << e.seconds << ",\"evaluation\":" << e.evaluation
        << ",\"round\":" << e.round << ",\"stale_before\":" << e.stale_before
        << ",\"stale_after\":" << e.stale_after << ",\"before_ticks\":" << e.before
        << ",\"perturbation_ticks\":" << e.perturbation << ",\"child_ticks\":" << e.child
        << ",\"before_backward_sum\":" << e.before_backward_sum
        << ",\"child_backward_sum\":" << e.child_backward_sum
        << ",\"refined\":" << e.refined << ",\"accepted\":" << e.accepted
        << ",\"trial_refinement_evaluations\":" << e.trial_evaluations << '}';
  }
  out << "]}";
  return out.str();
}

SearchResult HeuristicSearch::Run() {
  Initialize();
  Outer();
  auto verified = EvaluateSchedule(in_, best_->partition, best_->graph->orders);
  ++verification_calls_;
  Verify(*best_, verified.schedule);
  if (!ValidateScheduleSolutionIndependent(in_, verified.schedule, "FEASIBLE", "slack-heuristic").passed)
    throw Error("heuristic independent final validation failed");
  SearchResult r;
  r.method = "slack-heuristic";
  r.random_seed = options_.random_seed;
  r.status = BoundHit() ? "OPTIMAL" : "FEASIBLE";
  r.solver_status_raw = r.status;
  r.proven_optimal = BoundHit();
  r.optimality_proof_source = BoundHit() ? "global_workload_and_chain_bound" : "none";
  r.best_bound_ticks = bound_.value_or(0);
  r.split = best_->partition;
  r.machine_orders = best_->graph->orders;
  r.schedule = std::move(verified.schedule);
  r.makespan_ticks = best_->makespan;
  r.solver_objective_ticks = r.makespan_ticks;
  r.baseline_primary_objective = baseline_;
  r.improved_over_baseline = r.makespan_ticks < baseline_;
  r.evaluations = evaluations_;
  r.candidates_generated = generated_;
  r.iterations = rounds_;
  r.partition_moves_accepted = accepted_partition_;
  r.order_moves_accepted = accepted_single_ + accepted_stream_;
  r.requested_seconds = allowance_;
  r.deadline_reached = !options_.fixed_evaluation_test &&
      (Elapsed() >= search_end_ || (options_.parent_deadline && options_.parent_deadline->expired()));
  r.termination_reason = BoundHit() ? "validated_lower_bound" :
      (evaluations_ >= options_.evaluation_limit ? "evaluation_budget" : "deadline");
  r.incumbent_trace = trace_;
  r.first_feasible_objective = trace_.front().second;
  r.time_to_first_feasible_seconds = trace_.front().first;
  r.time_to_best_incumbent_seconds = trace_.back().first;
  r.incumbent_improvement_count = trace_.size() - 1;
  r.solution_source = "slack-heuristic";
  r.final_solution_available = true;
  r.heuristic_metadata_json = Metadata();
  r.wall_time_seconds = Elapsed();
  r.timing.total_seconds = r.timing.incumbent_seconds = r.wall_time_seconds;
  r.diagnostic = "Slack-based Heuristic: single parent, Combined perturbation, " +
      options_.refinement_mode + "; modeled makespan, not GPU time";
  return r;
}
}  // namespace

SearchResult SolveSlackHeuristic(const Instance& in,
                                 const SlackHeuristicOptions& options) {
  in.Validate();
  for (Index s = 0; s < in.stages; ++s)
    if (in.WorkerForStage(s) != s % in.workers) throw Error("heuristic requires cyclic placement");
  if (!std::isfinite(options.time_limit_seconds) || options.time_limit_seconds <= 0 ||
      options.evaluation_limit < 1) throw Error("heuristic requires positive finite time/evaluation budgets");
  if (options.refinement_mode != "coupled" && options.refinement_mode != "pd_then_refine")
    throw Error("heuristic refinement must be coupled or pd_then_refine");
  if (options.fixed_evaluation_test && options.parent_deadline)
    throw Error("fixed-evaluation test mode cannot share a wall-clock solver deadline");
  return HeuristicSearch(in, options).Run();
}
}  // namespace slackpipe
