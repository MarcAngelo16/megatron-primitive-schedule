#include <algorithm>
#include <numeric>
#include <set>

#include <gtest/gtest.h>

#include "slackpipe/breadth_first.h"
#include "slackpipe/io.h"
#include "slackpipe/octopipe_tuner.h"
#include "slackpipe/plan_export.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/instance.h"

namespace {
using namespace slackpipe;

Instance Tiny() {
  Instance i;
  i.microbatches = 4;
  i.stages = 4;
  i.workers = 2;
  i.total_layers = 8;
  return i;
}

OctoPipeState Initial(const Instance& i) {
  return *EvaluateOctoPipeCandidate(i, UniformSplit(i), BreadthFirstOrders(i));
}

void CheckState(const OctoPipeState& state, Index n) {
  const auto& i = state.instance;
  const auto& s = state.schedule;
  EXPECT_EQ(i.stages, n);
  EXPECT_EQ(s.split.size(), static_cast<std::size_t>(n));
  EXPECT_EQ(std::accumulate(s.split.begin(), s.split.end(), Tick{0}), i.total_layers);
  for (Tick layers : s.split) EXPECT_TRUE(layers >= i.min_layers);
  for (Index stage = 0; stage < n; ++stage) {
    EXPECT_TRUE(i.WorkerForStage(stage) >= 0 && i.WorkerForStage(stage) < i.workers);
  }
  EXPECT_TRUE(ValidateScheduleSolutionIndependent(i, s).passed);
  std::set<std::pair<Index, Index>> unique;
  Index forward = 0, backward = 0;
  for (const auto& order : s.orders) {
    for (auto id : order) {
      auto op = DecodeOperation(i, id);
      unique.emplace(op.microbatch, op.chain_index);
      if (op.backward) ++backward; else ++forward;
      const auto name = OperationName(i, id);
      EXPECT_TRUE(name.front() == 'F' || name.front() == 'B');
    }
  }
  EXPECT_EQ(unique.size(), static_cast<std::size_t>(2 * i.microbatches * n));
  EXPECT_EQ(forward, i.microbatches * n);
  EXPECT_EQ(backward, i.microbatches * n);
}

TEST(OctoPipe, ExactPhaseRuleIncludingBothEqualities) {
  OctoPipeBubbleMetrics m;
  m.delta_b = 3; m.boundary_sum = 1; m.residual_sum = 4;
  EXPECT_TRUE(SelectOctoPipePhase(m, 2) == OctoPipePhase::kPartition);
  m.delta_b = 2; m.boundary_sum = 5;
  EXPECT_TRUE(SelectOctoPipePhase(m, 2) == OctoPipePhase::kFixedStagePlacement);
  m.boundary_sum = 4;
  EXPECT_TRUE(SelectOctoPipePhase(m, 2) == OctoPipePhase::kScheduleFBOnly);
  m.delta_b = 1; m.boundary_sum = 3;
  EXPECT_TRUE(SelectOctoPipePhase(m, 2) == OctoPipePhase::kScheduleFBOnly);
}

TEST(OctoPipe, ExactTimelineBubbles) {
  auto i = Tiny();
  i.microbatches = 1; i.stages = 2; i.total_layers = 3;
  const auto s = EvaluateOctoPipeCandidate(i, {1, 2}, BreadthFirstOrders(i));
  ASSERT_TRUE(s.has_value());
  const auto m = ComputeOctoPipeBubbleMetrics(i, s->schedule);
  EXPECT_EQ(s->schedule.makespan, 6);
  EXPECT_TRUE(m.leading == (std::vector<Tick>{0, 1}));
  EXPECT_TRUE(m.trailing == (std::vector<Tick>{0, 1}));
  EXPECT_TRUE(m.boundary == (std::vector<Tick>{0, 2}));
  EXPECT_TRUE(m.residual == (std::vector<Tick>{4, 0}));
  EXPECT_TRUE(m.bubble == (std::vector<Tick>{4, 2}));
  EXPECT_EQ(m.delta_b, 2);
  EXPECT_EQ(m.boundary_sum, 2);
  EXPECT_EQ(m.residual_sum, 4);
}

TEST(OctoPipe, PartitionNeighborsPreserveContiguityAndMinimum) {
  auto i = Tiny();
  const auto state = Initial(i);
  OctoPipeOptions options;
  const auto neighbors = OctoPipeNeighborProposals(state, OctoPipePhase::kPartition, options);
  EXPECT_EQ(neighbors.size(), 6U);
  for (const auto& p : neighbors) {
    auto candidate = EvaluateOctoPipeCandidate(i, p.schedule.split, p.schedule.orders);
    ASSERT_TRUE(candidate.has_value());
    CheckState(*candidate, 4);
    EXPECT_TRUE(p.schedule.orders == state.schedule.orders);
    Index end = 0;
    for (Index s = 0; s < i.stages; ++s) {
      EXPECT_EQ(i.StageBeginLayer(s, p.schedule.split), end);
      end = i.StageEndLayer(s, p.schedule.split);
    }
    EXPECT_EQ(end, i.total_layers);
  }
  i.min_layers = 2;
  EXPECT_TRUE(OctoPipeNeighborProposals(Initial(i), OctoPipePhase::kPartition, options).empty());
}

TEST(OctoPipe, WholeStageSwapsKeepPartitionAndWorkerStageCounts) {
  const auto state = Initial(Tiny());
  OctoPipeOptions options;
  const auto neighbors = OctoPipeNeighborProposals(state, OctoPipePhase::kFixedStagePlacement, options);
  EXPECT_EQ(neighbors.size(), 4U);
  for (const auto& p : neighbors) {
    EXPECT_TRUE(p.schedule.split == state.schedule.split);
    EXPECT_EQ(std::count(p.instance.stage_to_worker.begin(), p.instance.stage_to_worker.end(), 0), 2);
    EXPECT_EQ(std::count(p.instance.stage_to_worker.begin(), p.instance.stage_to_worker.end(), 1), 2);
    auto candidate = EvaluateOctoPipeCandidate(p.instance, p.schedule.split, p.schedule.orders);
    ASSERT_TRUE(candidate.has_value());
    CheckState(*candidate, 4);
    const auto plan = ToMegatronSlackPipePlanJson(p.instance, candidate->schedule, "FEASIBLE");
    EXPECT_NE(plan.find("\"stage_to_worker\""), std::string::npos);
  }
  options.tune_placement = false;
  EXPECT_TRUE(OctoPipeNeighborProposals(state, OctoPipePhase::kFixedStagePlacement, options).empty());
}

TEST(OctoPipe, BoundedSchedulingEditsUseCommonValidation) {
  const auto state = Initial(Tiny());
  OctoPipeOptions options;
  options.candidates_per_iteration = 16;
  const auto neighbors = OctoPipeNeighborProposals(state, OctoPipePhase::kScheduleFBOnly, options);
  EXPECT_EQ(neighbors.size(), 16U);
  Index accepted = 0, rejected = 0;
  for (const auto& p : neighbors) {
    EXPECT_TRUE(p.schedule.split == state.schedule.split);
    auto evaluated = EvaluateOctoPipeCandidate(p.instance, p.schedule.split, p.schedule.orders);
    if (evaluated) { ++accepted; CheckState(*evaluated, 4); }
    else { ++rejected; }
  }
  EXPECT_TRUE(accepted > 0);
  EXPECT_TRUE(rejected > 0);
}

TEST(OctoPipe, CombinedCycleAndReverseFifoAreRejected) {
  auto i = Tiny();
  i.microbatches = 1; i.workers = 1; i.stages = 2; i.total_layers = 2;
  auto orders = BreadthFirstOrders(i);
  std::swap(orders[0][0], orders[0][1]);
  EXPECT_FALSE(EvaluateOctoPipeCandidate(i, {1, 1}, orders).has_value());
  auto predecessors = ExtractMachinePredecessors(i, orders);
  const auto evaluated = EvaluateScheduleWithPredecessors(i, {1, 1}, predecessors);
  EXPECT_FALSE(evaluated.schedule.ok());
  i.microbatches = 2; i.stages = 1; i.total_layers = 1;
  orders = {{EncodeOperation(i, 1, 0), EncodeOperation(i, 0, 0),
             EncodeOperation(i, 0, 1), EncodeOperation(i, 1, 1)}};
  EXPECT_FALSE(EvaluateOctoPipeCandidate(i, {1}, orders).has_value());
}

TEST(OctoPipe, MonotonicFixedNAndDeterministicIterationBudget) {
  const auto i = Tiny();
  OctoPipeOptions options;
  options.max_iterations = 10;
  Tick previous = kTickMax;
  Index callbacks = 0;
  options.progress = [&](const auto& log, const auto& state) {
    CheckState(state, 4);
    EXPECT_TRUE(log.best_makespan <= previous);
    EXPECT_EQ(log.accepted, log.best_makespan < log.input_makespan);
    previous = log.best_makespan;
    ++callbacks;
  };
  const auto a = TuneOctoPipeAlgorithm1(i, {5, 1, 1, 1}, BreadthFirstOrders(i), options);
  EXPECT_EQ(callbacks, 10);
  EXPECT_TRUE(a.best.schedule.makespan < a.initial_makespan);
  options.progress = {};
  const auto b = TuneOctoPipeAlgorithm1(i, {5, 1, 1, 1}, BreadthFirstOrders(i), options);
  EXPECT_TRUE(a.best.schedule.split == b.best.schedule.split);
  EXPECT_TRUE(a.best.instance.stage_to_worker == b.best.instance.stage_to_worker);
  EXPECT_TRUE(a.best.schedule.orders == b.best.schedule.orders);
  EXPECT_EQ(a.best.schedule.makespan, b.best.schedule.makespan);
}

TEST(OctoPipe, NoFallbackOrEarlyExitWhenPlacementDisabled) {
  auto i = Tiny();
  i.microbatches = 1; i.stages = 2; i.total_layers = 3;
  OctoPipeOptions options;
  options.max_iterations = 5;
  options.tune_placement = false;
  options.progress = [](const auto& log, const auto&) {
    EXPECT_TRUE(log.phase == OctoPipePhase::kFixedStagePlacement);
    EXPECT_EQ(log.candidates, 0);
    EXPECT_FALSE(log.accepted);
  };
  const auto result = TuneOctoPipeAlgorithm1(i, UniformSplit(i), BreadthFirstOrders(i), options);
  EXPECT_EQ(result.iterations, 5);
  EXPECT_EQ(result.best.schedule.makespan, result.initial_makespan);
}

TEST(OctoPipe, ActualHeterogeneousBoundaryCostsAndUnsplitLayerMinimum) {
  auto i = Tiny();
  i.profile_prefix_forward_ticks = {0, 1, 10, 12, 15, 19, 24, 30, 37};
  i.profile_prefix_backward_ticks = {0, 2, 20, 24, 30, 38, 48, 60, 74};
  i.profile_role_forward_bias_ticks = {2, 3, 4};
  i.profile_role_backward_bias_ticks = {5, 6, 7};
  EXPECT_EQ(MinimumOctoPipeLayerComputeCost(i), 3);
  EXPECT_EQ(i.ForwardDuration(0, 0, 2), 12);
  const auto state = Initial(i);
  bool found = false;
  for (const auto& p : OctoPipeNeighborProposals(state, OctoPipePhase::kPartition, {})) {
    if (p.schedule.split != std::vector<Tick>{1, 3, 2, 2}) continue;
    found = true;
    const auto evaluated = EvaluateOctoPipeCandidate(i, p.schedule.split, p.schedule.orders);
    ASSERT_TRUE(evaluated.has_value());
    EXPECT_EQ(evaluated->schedule.operations_by_id[EncodeOperation(i, 0, 0).value].duration, 3);
    EXPECT_EQ(evaluated->schedule.operations_by_id[EncodeOperation(i, 0, 1).value].duration, 17);
    EXPECT_EQ(i.BackwardDuration(0, 0, 1), 7);
    CheckState(*evaluated, 4);
  }
  EXPECT_TRUE(found);
  OctoPipeOptions options;
  options.max_iterations = 4;
  const auto result = TuneOctoPipeAlgorithm1(i, state.schedule.split, state.schedule.orders, options);
  EXPECT_TRUE(result.best.schedule.makespan <= result.initial_makespan);
}

TEST(OctoPipe, ExplicitPlacementRoundTripsThroughCommonResultValidator) {
  auto i = Tiny();
  i.stage_to_worker = {0, 0, 1, 1};
  const auto s = EvaluateOctoPipeCandidate(i, {1, 1, 2, 4}, BreadthFirstOrders(i))->schedule;
  EXPECT_TRUE(WorkerLayerTotals(i, s.split) == (std::vector<Tick>{2, 6}));
  SearchResult result;
  result.method = "octopipe-tuning";
  result.status = "FEASIBLE";
  result.schedule = s;
  result.makespan_ticks = s.makespan;
  result.final_solution_available = true;
  EXPECT_TRUE(ValidateResultJsonText(ToJson(i, result)).passed);
  EXPECT_TRUE(ValidationInputFromResultJsonText(ToJson(i, result)).instance.stage_to_worker ==
              i.stage_to_worker);
  i.stage_to_worker = {0, 2, 0, 1};
  EXPECT_THROW(i.Validate(), Error);
}

TEST(OctoPipe, RequiresFiniteBoundAndValidInitialState) {
  const auto i = Tiny();
  EXPECT_THROW(TuneOctoPipeAlgorithm1(i, UniformSplit(i), BreadthFirstOrders(i), {}), Error);
  OctoPipeOptions options;
  options.max_iterations = 1;
  EXPECT_THROW(TuneOctoPipeAlgorithm1(i, {8}, BreadthFirstOrders(i), options), Error);
}
}  // namespace
