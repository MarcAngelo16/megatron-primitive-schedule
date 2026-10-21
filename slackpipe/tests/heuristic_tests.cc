#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "../src/heuristic_internal.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/io.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/instance.h"
#include "slackpipe/slack_heuristic.h"

namespace {
using namespace slackpipe;
using namespace slackpipe::heuristic_internal;
Instance Tiny() {
  Instance in;
  in.microbatches = 4; in.stages = 4; in.workers = 2; in.total_layers = 12;
  in.backward_ratio_num = 3; in.backward_ratio_den = 2; in.communication_ticks = 2;
  return in;
}
Pair Baseline(const Instance& in) {
  auto p = UniformSplit(in);
  return *Replay(in, p, Compile(in, BreadthFirstOrders(in)), PhaseCosts(in, p));
}
void Compare(const Instance& in, const Pair& c) {
  auto e = EvaluateSchedule(in, c.partition, c.graph->orders).schedule;
  ASSERT_TRUE(e.ok());
  ASSERT_EQ(c.makespan, e.makespan);
  for (std::size_t k = 0; k < c.end.size(); ++k) {
    EXPECT_EQ(c.start[k], e.operations_by_id[k].start);
    EXPECT_EQ(c.end[k], e.operations_by_id[k].end);
  }
}
template <typename F> bool Throws(F f) {
  try { f(); } catch (const Error&) { return true; }
  return false;
}
SlackHeuristicOptions Fixed(Index count) {
  SlackHeuristicOptions o;
  o.fixed_evaluation_test = true;
  o.evaluation_limit = count;
  o.record_trace = true;
  return o;
}
}

TEST(Heuristic, CompleteReplayAndTotalFloat) {
  auto in = Tiny();
  auto c = Baseline(in);
  Compare(in, c);
  auto f = TotalFloat(c);
  EXPECT_TRUE(std::all_of(f.slack.begin(), f.slack.end(), [](Tick t) { return t >= 0; }));
  Tick backwards = 0;
  for (Index id = 0; id < in.OperationCount(); ++id) {
    if (id % 8 >= 4) backwards += c.end[id];
    if (c.end[id] == c.makespan) EXPECT_EQ(f.slack[id], 0);
  }
  EXPECT_EQ(c.backward_sum, backwards);
  EXPECT_TRUE(std::accumulate(f.critical.begin(), f.critical.end(), Tick{0}) > 0);
}

TEST(Heuristic, RejectsInventoryWorkerFifoAndCrossWorkerCycles) {
  auto in = Tiny();
  const auto valid = BreadthFirstOrders(in);
  auto o = valid;
  o[0].pop_back(); EXPECT_TRUE(!Compile(in, o));
  o = valid; o[0].push_back(o[0][0]); EXPECT_TRUE(!Compile(in, o));
  o = valid; std::swap(o[0][0], o[1][0]); EXPECT_TRUE(!Compile(in, o));
  o = valid;
  auto x = std::find(o[0].begin(), o[0].end(), OperationId{0});
  auto y = std::find(o[0].begin(), o[0].end(), OperationId{8});
  std::iter_swap(x, y); EXPECT_TRUE(!Compile(in, o));
  in.microbatches = 1; in.stages = 2; in.workers = 2; in.total_layers = 2;
  o = {{{3}, {0}}, {{1}, {2}}};  // FIFO vacuous, worker edge closes a data cycle.
  EXPECT_TRUE(!Compile(in, o));
  EXPECT_TRUE(!EvaluateSchedule(in, {1, 1}, o).schedule.ok());
}

TEST(Heuristic, HeterogeneousNonadjacentTransfersRecomputeBothPhases) {
  auto in = Tiny();
  in.profile_prefix_forward_ticks = {0,1,3,7,15,31,63,127,255,511,1023,2047,4095};
  in.profile_prefix_backward_ticks = {0,3,4,9,11,18,22,30,33,39,44,53,60};
  in.profile_role_forward_bias_ticks = {2,0,5};
  in.profile_role_backward_bias_ticks = {7,1,8};
  auto base = Baseline(in);
  auto p = base.partition;
  --p[0]; ++p[2];  // Same worker, but also moves stage 1's entire interval.
  const auto costs = PhaseCosts(in, p);
  EXPECT_TRUE(costs[1] != base.duration[1]);
  EXPECT_TRUE(costs[6] != base.duration[6]);
  EXPECT_TRUE(costs[0] + costs[2] != base.duration[0] + base.duration[2]);
  auto next = Replay(in, p, base.graph, costs);
  ASSERT_TRUE(next.has_value());
  EXPECT_TRUE(next->graph == base.graph);
  Compare(in, *next);
}

TEST(Heuristic, InsertEarlierLaterEndAndAtomicRollback) {
  Instance in; in.stages = 2; in.workers = 1; in.microbatches = 3; in.total_layers = 4;
  MachineOrders base(1);
  for (Index id = 0; id < 12; ++id) base[0].push_back({id});
  auto o = base;
  EXPECT_TRUE(ApplyOrder(in, {0,2,1,false,0}, o));
  EXPECT_EQ(o[0][1].value, 2);
  o = base;
  EXPECT_TRUE(ApplyOrder(in, {0,1,3,false,0}, o));
  EXPECT_EQ(o[0][2].value, 1);
  o = base;
  EXPECT_TRUE(ApplyOrder(in, {0,10,-1,false,0}, o));
  EXPECT_EQ(o[0].back().value, 10);
  o = base;
  EXPECT_TRUE(!ApplyOrder(in, {0,0,5,false,0}, o)); // Crosses F0(b1).
  EXPECT_TRUE(o == base);
  EXPECT_TRUE(!ApplyOrder(in, {0,0,99,false,0}, o));
  EXPECT_TRUE(o == base);
}

TEST(Heuristic, StreamOffsetsBoundaryTruncationAndEndSemantics) {
  Instance in; in.stages = 2; in.workers = 1; in.microbatches = 3; in.total_layers = 4;
  MachineOrders base(1);
  for (Index id = 0; id < 12; ++id) base[0].push_back({id});
  auto o = base;
  EXPECT_TRUE(ApplyOrder(in, {0,1,4,true,0}, o)); // phase 1 before phase 0, offset +1.
  auto at = std::find(o[0].begin(), o[0].end(), OperationId{4});
  EXPECT_EQ(std::prev(at)->value, 1);
  at = std::find(o[0].begin(), o[0].end(), OperationId{8});
  EXPECT_EQ(std::prev(at)->value, 5);
  EXPECT_TRUE(std::find(o[0].begin(), o[0].end(), OperationId{9}) != o[0].end());
  o = base;
  EXPECT_TRUE(ApplyOrder(in, {0,4,3,true,0}, o)); // offset -1, source b0 is untouched.
  at = std::find(o[0].begin(), o[0].end(), OperationId{3});
  EXPECT_EQ(std::prev(at)->value, 4);
  EXPECT_EQ(o[0].front().value, 0);
  o = base;
  EXPECT_TRUE(ApplyOrder(in, {0,1,-1,true,0}, o));
  EXPECT_TRUE(o == base);
  EXPECT_TRUE(!ApplyOrder(in, {0,4,0,true,0}, o)); // Same-phase move breaks FIFO.
  EXPECT_TRUE(o == base);
  in.microbatches = 65;
  EXPECT_TRUE(!ApplyOrder(in, {0,1,-1,true,0}, o));
  EXPECT_TRUE(!ApplyOrder(in, {0,0,4,true,0}, o));
  EXPECT_TRUE(o == base);
}

TEST(Heuristic, LargeBatchStreamPreservesPermutation) {
  Instance in; in.stages = 2; in.workers = 1; in.microbatches = 129; in.total_layers = 4;
  MachineOrders orders(1);
  for (Index id = 0; id < in.OperationCount(); ++id) orders[0].push_back({id});
  EXPECT_TRUE(ApplyOrder(in, {0,2,1,true,0}, orders));
  ASSERT_EQ(orders[0].size(), static_cast<std::size_t>(in.OperationCount()));
  for (Index b = 0; b < in.microbatches; ++b) {
    EXPECT_EQ(orders[0][b * 4 + 1].value, b * 4 + 2);
    EXPECT_EQ(orders[0][b * 4 + 2].value, b * 4 + 1);
  }
}

TEST(Heuristic, RankingUsesCriticalIntervalAndOriginalRightwardHole) {
  auto in = Tiny();
  auto c = Baseline(in);
  auto f = TotalFloat(c);
  const auto actions = OrderActions(in, c, f, 0);
  Index noncritical_sources = 0, rightward = 0;
  const Index q = 2 * in.stages;
  for (const auto& action : actions) {
    if (action.stream) continue;
    const auto& order = c.graph->orders[action.worker];
    Index a = std::find(order.begin(), order.end(), OperationId{action.source}) - order.begin();
    Index anchor = action.anchor < 0 ? order.size() :
        std::find(order.begin(), order.end(), OperationId{action.anchor}) - order.begin();
    Index j = anchor < a ? anchor : anchor - 1;
    bool critical = false;
    for (Index k = std::min(a,j); k <= std::max(a,j); ++k)
      critical |= f.slack[order[k].value] == 0;
    EXPECT_TRUE(critical);
    noncritical_sources += f.slack[action.source] > 0;
    rightward += j > a;
    Tick ready = 0;
    if (action.source % q) ready = c.end[action.source - 1] + in.EdgeDelay(
        DecodeOperation(in, {action.source - 1}).worker, action.worker);
    if (action.source >= q) ready = std::max(ready, c.end[action.source - q]);
    Tick available = std::max<Tick>(0, c.start[order[j].value] -
        std::max(j ? c.end[order[j - 1].value] : 0, ready));
    long double expected = -(static_cast<long double>(
        std::min(c.duration[action.source % q], available)) +
        (f.slack[action.source] == 0 ? .25L * c.duration[action.source % q] : 0));
    EXPECT_TRUE(expected == action.score);
  }
  EXPECT_TRUE(noncritical_sources > 0);
  EXPECT_TRUE(rightward > 0);
}

TEST(Heuristic, BetterEverySecondaryComponentAndStale) {
  auto b = Baseline(Tiny()), a = b;
  --a.makespan; EXPECT_EQ(ImprovementReason(a,b), std::string("makespan"));
  a = b; --a.backward_sum;
  EXPECT_EQ(ImprovementReason(a,b), std::string("backward_completion_sum"));
  Pair parent = b;
  EXPECT_EQ(AcceptChild(a,parent,15), 16);
  EXPECT_TRUE(parent.backward_sum == a.backward_sum);
  a = b; --a.partition[0]; ++a.partition[1];
  EXPECT_EQ(ImprovementReason(a,b), std::string("partition_lex"));
  a = b;
  auto graph = std::make_shared<Graph>(*b.graph);
  // Comparator ordering is tested independently of schedule feasibility.
  graph->orders[0][0].value = -1;
  a.graph = graph;
  EXPECT_EQ(ImprovementReason(a,b), std::string("order_lex"));
  EXPECT_TRUE(Better(a,b));
  EXPECT_TRUE(!Better(b,b));
  a = b; --a.makespan;
  EXPECT_EQ(AcceptChild(a,parent,31), 0);
  EXPECT_EQ(AcceptChild(parent,parent,31), 32);
}

TEST(Heuristic, OverflowIsDiagnosticNotFloatingPointTie) {
  auto in = Tiny();
  auto c = Baseline(in);
  auto costs = c.duration;
  costs[0] = kTickMax;
  EXPECT_TRUE(Throws([&] { Replay(in, c.partition, c.graph, costs); }));
}

TEST(Heuristic, DebugReplayAndFinalGeneralValidation) {
  auto in = Tiny();
  auto options = Fixed(600);
  options.verify_cached_replay = true;
  const auto result = SolveSlackHeuristic(in, options);
  EXPECT_TRUE(result.evaluations <= 600);
  EXPECT_TRUE(result.makespan_ticks <= Baseline(in).makespan);
  EXPECT_TRUE(ValidateScheduleSolutionIndependent(in, result.schedule, result.status, "slack-heuristic").passed);
  EXPECT_TRUE(ValidateResultJsonText(ToJson(in, result)).passed);
  EXPECT_TRUE(result.heuristic_metadata_json.find("\"outer_pd_calls\":0") != std::string::npos);
}

TEST(Heuristic, FixedEvaluationDeterministicSingleParentTrajectory) {
  auto in = Tiny();
  in.profile_forward_slope_ticks = {1,3,2,4};
  in.profile_backward_slope_ticks = {6,4,5,2};
  in.profile_forward_bias_ticks = {0,2,0,4};
  in.profile_backward_bias_ticks = {1,0,2,0};
  auto options = Fixed(40000);
  const auto a = SolveSlackHeuristic(in, options);
  options.time_limit_seconds = .000001; // No wall-triggered truncation in test mode.
  const auto b = SolveSlackHeuristic(in, options);
  EXPECT_TRUE(a.split == b.split);
  EXPECT_TRUE(a.machine_orders == b.machine_orders);
  EXPECT_EQ(a.evaluations, b.evaluations);
  EXPECT_EQ(a.iterations, b.iterations);
  ASSERT_EQ(a.incumbent_trace.size(), b.incumbent_trace.size());
  for (std::size_t k = 0; k < a.incumbent_trace.size(); ++k)
    EXPECT_EQ(a.incumbent_trace[k].second, b.incumbent_trace[k].second);
  EXPECT_TRUE(a.heuristic_metadata_json.find("\"outer_pd_calls\":0") != std::string::npos);
}

TEST(Heuristic, RejectsUnsupportedSettingsAndHonorsPartitionPredicate) {
  auto in = Tiny();
  auto options = Fixed(100);
  options.allowed_partition = [](const auto&) { return false; };
  EXPECT_TRUE(Throws([&] { (void)SolveSlackHeuristic(in, options); }));
  options = Fixed(100); options.refinement_mode = "beam";
  EXPECT_TRUE(Throws([&] { (void)SolveSlackHeuristic(in, options); }));
  options = Fixed(100); in.stage_to_worker = {0,0,1,1};
  EXPECT_TRUE(Throws([&] { (void)SolveSlackHeuristic(in, options); }));
}

TEST(Heuristic, LargeStageNeighborhoodAndBoundCertificates) {
  auto in = Tiny();
  in.stages = 20; in.total_layers = 60; in.workers = 4;
  auto options = Fixed(2500);
  options.verify_cached_replay = true;
  auto r = SolveSlackHeuristic(in, options);
  EXPECT_TRUE(r.makespan_ticks <= Baseline(in).makespan);
  EXPECT_TRUE(ValidateResultJsonText(ToJson(in, r)).passed);
  EXPECT_TRUE(r.best_bound_ticks <= r.makespan_ticks);
  in.workers = 1;
  r = SolveSlackHeuristic(in, Fixed(1));
  EXPECT_TRUE(r.proven_optimal);
  EXPECT_EQ(r.best_bound_ticks, r.makespan_ticks);
  EXPECT_EQ(r.termination_reason, std::string("validated_lower_bound"));
}
