#include "slackpipe/octopipe_tuner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>
#include <tuple>

#include "slackpipe/result_validator.h"

namespace slackpipe {
namespace {
using Clock = std::chrono::steady_clock;

std::vector<Index> Placement(const Instance& instance) {
  std::vector<Index> result;
  for (Index s = 0; s < instance.stages; ++s) result.push_back(instance.WorkerForStage(s));
  return result;
}

bool Better(const OctoPipeState& a, const OctoPipeState& b) {
  return std::make_tuple(a.schedule.makespan, a.schedule.split,
                         Placement(a.instance), a.schedule.orders) <
         std::make_tuple(b.schedule.makespan, b.schedule.split,
                         Placement(b.instance), b.schedule.orders);
}

// Preserve the evaluated global compute order when reassigning whole stages.
// Equal-time/zero-cost ambiguities are checked by the common DAG validator.
MachineOrders ProjectOrders(const OctoPipeState& state, const Instance& placement) {
  auto operations = state.schedule.operations_by_id;
  std::sort(operations.begin(), operations.end(), [](const auto& a, const auto& b) {
    return std::tie(a.start, a.end, a.id) < std::tie(b.start, b.end, b.id);
  });
  MachineOrders orders(placement.workers);
  for (const auto& operation : operations) {
    orders[DecodeOperation(placement, operation.id).worker].push_back(operation.id);
  }
  return orders;
}
}  // namespace

const char* ToString(OctoPipePhase phase) {
  switch (phase) {
    case OctoPipePhase::kPartition: return "partition";
    case OctoPipePhase::kFixedStagePlacement: return "fixed-stage-placement";
    case OctoPipePhase::kScheduleFBOnly: return "schedule-fb-only";
  }
  throw Error("unknown OctoPipe phase");
}

OctoPipeBubbleMetrics ComputeOctoPipeBubbleMetrics(const Instance& instance,
                                                  const ScheduleSolution& schedule) {
  if (!ValidateScheduleSolutionIndependent(instance, schedule).passed) {
    throw Error("OctoPipe bubble metrics require a validated schedule");
  }
  OctoPipeBubbleMetrics result;
  for (const auto& order : schedule.orders) {
    Tick leading = schedule.makespan, trailing = 0, residual = 0;
    if (!order.empty()) {
      leading = schedule.operations_by_id[order.front().value].start;
      trailing = schedule.makespan - schedule.operations_by_id[order.back().value].end;
      for (std::size_t i = 1; i < order.size(); ++i) {
        residual = CheckedAdd(residual,
            schedule.operations_by_id[order[i].value].start -
            schedule.operations_by_id[order[i - 1].value].end, "residual bubbles");
      }
    }
    const Tick boundary = CheckedAdd(leading, trailing, "boundary bubbles");
    result.leading.push_back(leading);
    result.trailing.push_back(trailing);
    result.boundary.push_back(boundary);
    result.residual.push_back(residual);
    result.bubble.push_back(CheckedAdd(boundary, residual, "worker bubbles"));
    result.boundary_sum = CheckedAdd(result.boundary_sum, boundary, "boundary sum");
    result.residual_sum = CheckedAdd(result.residual_sum, residual, "residual sum");
  }
  const auto [lo, hi] = std::minmax_element(result.bubble.begin(), result.bubble.end());
  result.delta_b = *hi - *lo;
  return result;
}

Tick MinimumOctoPipeLayerComputeCost(const Instance& instance) {
  instance.Validate();
  Tick minimum = kTickMax;
  if (instance.HasRangeCostProfile()) {
    for (Index l = 0; l < instance.total_layers; ++l) {
      minimum = std::min(minimum, CheckedAdd(
          instance.profile_prefix_forward_ticks[l + 1] - instance.profile_prefix_forward_ticks[l],
          instance.profile_prefix_backward_ticks[l + 1] - instance.profile_prefix_backward_ticks[l],
          "minimum layer cost"));
    }
  } else if (instance.HasAffineCostProfile()) {
    // v1 has stage-indexed slopes, not individual layer observations. Exclude
    // fixed stage biases and use the minimum available per-layer F+B estimate.
    for (Index s = 0; s < instance.stages; ++s) {
      minimum = std::min(minimum, CheckedAdd(instance.profile_forward_slope_ticks[s],
          instance.profile_backward_slope_ticks[s], "minimum affine layer cost"));
    }
  } else {
    minimum = CheckedAdd(instance.backward_ratio_den, instance.backward_ratio_num,
                         "minimum uniform layer cost");
  }
  return minimum;
}

OctoPipePhase SelectOctoPipePhase(const OctoPipeBubbleMetrics& metrics, Tick t_layer) {
  if (metrics.delta_b > t_layer) return OctoPipePhase::kPartition;
  if (metrics.boundary_sum > metrics.residual_sum) return OctoPipePhase::kFixedStagePlacement;
  return OctoPipePhase::kScheduleFBOnly;
}

std::optional<OctoPipeState> EvaluateOctoPipeCandidate(
    const Instance& instance, const std::vector<Tick>& split, const MachineOrders& orders) {
  auto schedule = EvaluateSchedule(instance, split, orders).schedule;
  if (!schedule.ok() || !ValidateScheduleSolutionIndependent(instance, schedule).passed) {
    return std::nullopt;
  }
  const auto predecessors = ExtractMachinePredecessors(instance, orders);
  const auto checked = EvaluateScheduleWithPredecessors(instance, split, predecessors);
  if (!checked.schedule.ok() || checked.schedule.makespan != schedule.makespan) {
    return std::nullopt;
  }
  return OctoPipeState{instance, std::move(schedule)};
}

std::vector<OctoPipeState> OctoPipeNeighborProposals(
    const OctoPipeState& state, OctoPipePhase phase, const OctoPipeOptions& options,
    const std::function<bool()>& stop_requested) {
  if (options.candidates_per_iteration <= 0) throw Error("invalid OctoPipe candidate bound");
  const auto stopped = [&]() { return stop_requested && stop_requested(); };
  const auto& instance = state.instance;
  const auto metrics = ComputeOctoPipeBubbleMetrics(instance, state.schedule);
  std::vector<std::pair<Tick, OctoPipeState>> ranked;
  auto proposal = [&]() {
    OctoPipeState copy{instance, {}};
    copy.schedule.split = state.schedule.split;
    copy.schedule.orders = state.schedule.orders;
    return copy;
  };
  if (phase == OctoPipePhase::kPartition) {
    for (Index s = 0; s + 1 < instance.stages; ++s) {
      if (stopped()) break;
      for (Index direction : {-1, 1}) {
        auto candidate = proposal();
        const Index donor = direction == -1 ? s : s + 1;
        const Index receiver = direction == -1 ? s + 1 : s;
        if (candidate.schedule.split[donor] <= instance.min_layers) continue;
        --candidate.schedule.split[donor];
        ++candidate.schedule.split[receiver];
        // Less bubble means more occupied compute at the same global makespan.
        const Tick priority = metrics.bubble[instance.WorkerForStage(receiver)] -
                              metrics.bubble[instance.WorkerForStage(donor)];
        ranked.emplace_back(priority, std::move(candidate));
      }
    }
  } else if (phase == OctoPipePhase::kFixedStagePlacement && options.tune_placement) {
    for (Index i = 0; i < instance.stages; ++i) {
      if (stopped()) break;
      for (Index j = i + 1; j < instance.stages; ++j) {
        if (stopped()) break;
        const Index x = instance.WorkerForStage(i), y = instance.WorkerForStage(j);
        if (x == y) continue;
        auto candidate = proposal();
        candidate.instance.stage_to_worker = Placement(instance);
        std::swap(candidate.instance.stage_to_worker[i], candidate.instance.stage_to_worker[j]);
        candidate.schedule.orders = ProjectOrders(state, candidate.instance);
        ranked.emplace_back(std::abs(metrics.boundary[x] - metrics.boundary[y]),
                            std::move(candidate));
      }
    }
  } else if (phase == OctoPipePhase::kScheduleFBOnly) {
    // Largest residual gaps first; at most four later operations can advance
    // into each slot. Dependency readiness is decided by the shared validator.
    std::vector<std::tuple<Tick, Index, std::size_t>> gaps;
    for (Index w = 0; w < instance.workers; ++w) {
      if (stopped()) break;
      const auto& order = state.schedule.orders[w];
      for (std::size_t to = 0; to + 1 < order.size(); ++to) {
        const Tick start = state.schedule.operations_by_id[order[to].value].start;
        const Tick previous_end = to ? state.schedule.operations_by_id[order[to - 1].value].end : 0;
        gaps.emplace_back(-(start - previous_end), w, to);
      }
    }
    std::sort(gaps.begin(), gaps.end());
    for (const auto& [negative_gap, w, to] : gaps) {
      if (stopped()) break;
      const auto size = state.schedule.orders[w].size();
      for (std::size_t from = to + 1; from < std::min(size, to + 5); ++from) {
        if (ranked.size() >= static_cast<std::size_t>(options.candidates_per_iteration)) break;
        auto candidate = proposal();
        auto& order = candidate.schedule.orders[w];
        const auto operation = order[from];
        order.erase(order.begin() + from);
        order.insert(order.begin() + to, operation);
        ranked.emplace_back(-negative_gap, std::move(candidate));
      }
      if (ranked.size() >= static_cast<std::size_t>(options.candidates_per_iteration)) break;
    }
  }
  std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });
  std::vector<OctoPipeState> result;
  for (auto& entry : ranked) result.push_back(std::move(entry.second));
  return result;
}

OctoPipeResult TuneOctoPipeAlgorithm1(const Instance& instance,
                                     const std::vector<Tick>& initial_split,
                                     const MachineOrders& initial_orders,
                                     const OctoPipeOptions& options) {
  if (!std::isfinite(options.time_limit_seconds) || options.time_limit_seconds < 0 ||
      options.max_iterations < 0 || options.candidates_per_iteration <= 0 ||
      (options.time_limit_seconds == 0 && options.max_iterations == 0)) {
    throw Error("OctoPipe requires a positive time or iteration limit and candidate bound");
  }
  // The public wall-clock budget includes initialization and its validation.
  // Iteration-bounded trajectories are unchanged.
  const auto started = Clock::now();
  auto initial = EvaluateOctoPipeCandidate(instance, initial_split, initial_orders);
  if (!initial) throw Error("invalid OctoPipe initial state");
  OctoPipeResult result;
  result.best = std::move(*initial);
  result.initial_makespan = result.best.schedule.makespan;
  const Tick t_layer = MinimumOctoPipeLayerComputeCost(instance);
  auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - started).count(); };
  auto expired = [&]() {
    return options.time_limit_seconds > 0 && elapsed() >= options.time_limit_seconds;
  };
  while (!expired() && (!options.max_iterations || result.iterations < options.max_iterations)) {
    OctoPipeIteration log;
    log.iteration = result.iterations++;
    log.input_makespan = result.best.schedule.makespan;
    log.metrics = ComputeOctoPipeBubbleMetrics(result.best.instance, result.best.schedule);
    log.t_layer = t_layer;
    log.phase = SelectOctoPipePhase(log.metrics, t_layer);
    std::optional<OctoPipeState> best_neighbor;
    const auto proposals = OctoPipeNeighborProposals(result.best, log.phase, options, expired);
    for (const auto& proposal : proposals) {
      if (expired()) break;
      ++log.candidates;
      auto candidate = EvaluateOctoPipeCandidate(proposal.instance, proposal.schedule.split,
                                                 proposal.schedule.orders);
      if (!candidate) continue;
      ++log.valid_candidates;
      if (!best_neighbor || Better(*candidate, *best_neighbor)) best_neighbor = std::move(candidate);
    }
    if (best_neighbor && best_neighbor->schedule.makespan < result.best.schedule.makespan) {
      result.best = std::move(*best_neighbor);
      log.accepted = true;
    }
    log.best_makespan = result.best.schedule.makespan;
    log.elapsed_seconds = elapsed();
    if (options.progress) options.progress(log, result.best);
    // No phase fallback, plateau acceptance, or early convergence exit.
    if (!log.accepted && !options.max_iterations) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  result.tuning_seconds = elapsed();
  return result;
}

OctoPipePhaseDiagnostic DiagnoseOctoPipePhase(
    const OctoPipeState& initial, OctoPipePhase phase, const OctoPipeOptions& options) {
  if (!options.tune_placement) {
    throw Error("forced-phase diagnostic requires placement tuning enabled");
  }
  OctoPipePhaseDiagnostic result;
  result.phase = phase;
  result.initial_makespan = initial.schedule.makespan;
  // Reuse exactly the production proposals, evaluator/validators, and tie-breaker.
  // The caller's initial state remains const; no phase consumes another's result.
  const auto proposals = OctoPipeNeighborProposals(initial, phase, options);
  result.generated = static_cast<Index>(proposals.size());
  for (const auto& proposal : proposals) {
    auto candidate = EvaluateOctoPipeCandidate(proposal.instance, proposal.schedule.split,
                                               proposal.schedule.orders);
    if (!candidate) continue;
    ++result.valid;
    if (candidate->schedule.makespan < result.initial_makespan) ++result.improving;
    if (!result.best_valid || Better(*candidate, *result.best_valid)) {
      result.best_valid = std::move(candidate);
    }
  }
  return result;
}
}  // namespace slackpipe
