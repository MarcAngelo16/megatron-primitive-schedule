#include <algorithm>
#include <numeric>
#include <set>

#include <gtest/gtest.h>

#include "slackpipe/breadth_first.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/io.h"
#include "slackpipe/octopipe_tuner.h"
#include "slackpipe/one_f_one_b.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/instance.h"

namespace {
using namespace slackpipe;

OctoPipeState Initial() {
  Instance i;
  i.microbatches = 8; i.stages = 8; i.workers = 4; i.total_layers = 64;
  i.backward_ratio_num = 2;
  return *EvaluateOctoPipeCandidate(i, UniformSplit(i), BreadthFirstOrders(i));
}

void CheckState(const OctoPipeState& state) {
  EXPECT_EQ(state.instance.stages, 8);
  EXPECT_EQ(state.schedule.split.size(), 8U);
  EXPECT_EQ(std::accumulate(state.schedule.split.begin(), state.schedule.split.end(), Tick{0}), 64);
  EXPECT_TRUE(ValidateScheduleSolutionIndependent(state.instance, state.schedule).passed);
  std::set<Index> unique;
  Index forward = 0, backward = 0;
  for (const auto& order : state.schedule.orders) {
    for (const auto id : order) {
      unique.insert(id.value);
      const auto op = DecodeOperation(state.instance, id);
      if (op.backward) ++backward; else ++forward;
    }
  }
  EXPECT_EQ(unique.size(), 128U);
  EXPECT_EQ(forward, 64);
  EXPECT_EQ(backward, 64);
}

TEST(OctoPipeDiagnostic, IndependentInvocationsNeverMutateInitialState) {
  const auto initial = Initial();
  const auto snapshot = ToJson(initial.instance, initial.schedule);
  const auto metrics = ComputeOctoPipeBubbleMetrics(initial.instance, initial.schedule);
  EXPECT_EQ(initial.schedule.makespan, 456);
  EXPECT_EQ(metrics.delta_b, 0);
  EXPECT_EQ(metrics.boundary_sum, 144);
  EXPECT_EQ(metrics.residual_sum, 144);
  EXPECT_EQ(MinimumOctoPipeLayerComputeCost(initial.instance), 3);
  EXPECT_TRUE(SelectOctoPipePhase(metrics, 3) == OctoPipePhase::kScheduleFBOnly);
  Index which = 0;
  for (auto phase : {OctoPipePhase::kPartition, OctoPipePhase::kFixedStagePlacement,
                     OctoPipePhase::kScheduleFBOnly}) {
    const auto clone = initial;
    const auto report = DiagnoseOctoPipePhase(clone, phase);
    EXPECT_EQ(report.initial_makespan, 456);
    EXPECT_EQ(report.generated, (std::vector<Index>{14, 24, 64})[which++]);
    EXPECT_EQ(ToJson(clone.instance, clone.schedule), snapshot);
    EXPECT_EQ(ToJson(initial.instance, initial.schedule), snapshot);
    const auto repeated = DiagnoseOctoPipePhase(initial, phase);
    EXPECT_EQ(report.generated, repeated.generated);
    EXPECT_EQ(report.valid, repeated.valid);
    EXPECT_EQ(report.improving, repeated.improving);
    ASSERT_TRUE(report.best_valid.has_value());
    ASSERT_TRUE(repeated.best_valid.has_value());
    CheckState(*report.best_valid);
    EXPECT_EQ(ToJson(report.best_valid->instance, report.best_valid->schedule),
              ToJson(repeated.best_valid->instance, repeated.best_valid->schedule));
    if (phase == OctoPipePhase::kPartition) {
      EXPECT_TRUE(report.best_valid->schedule.orders == initial.schedule.orders);
    } else if (phase == OctoPipePhase::kFixedStagePlacement) {
      EXPECT_TRUE(report.best_valid->schedule.split == initial.schedule.split);
    }
  }
}

TEST(OctoPipeDiagnostic, CountsOnlyCommonValidatorAcceptedCandidates) {
  const auto initial = Initial();
  for (auto phase : {OctoPipePhase::kPartition, OctoPipePhase::kFixedStagePlacement,
                     OctoPipePhase::kScheduleFBOnly}) {
    const auto report = DiagnoseOctoPipePhase(initial, phase);
    Index valid = 0, improving = 0;
    Tick minimum = kTickMax;
    for (const auto& proposal : OctoPipeNeighborProposals(initial, phase, {})) {
      auto candidate = EvaluateOctoPipeCandidate(proposal.instance, proposal.schedule.split,
                                                 proposal.schedule.orders);
      if (!candidate) continue;
      ++valid;
      if (candidate->schedule.makespan < initial.schedule.makespan) ++improving;
      minimum = std::min(minimum, candidate->schedule.makespan);
      CheckState(*candidate);
    }
    EXPECT_EQ(report.valid, valid);
    EXPECT_EQ(report.improving, improving);
    ASSERT_TRUE(report.best_valid.has_value());
    EXPECT_EQ(report.best_valid->schedule.makespan, minimum);
  }
}

TEST(OctoPipeDiagnostic, ProductionSelectedPhaseAndAcceptanceRemainIdentical) {
  const auto initial = Initial();
  const auto phase = SelectOctoPipePhase(
      ComputeOctoPipeBubbleMetrics(initial.instance, initial.schedule),
      MinimumOctoPipeLayerComputeCost(initial.instance));
  const auto diagnostic = DiagnoseOctoPipePhase(initial, phase);
  OctoPipeOptions options;
  options.max_iterations = 1;
  options.progress = [&](const auto& log, const auto&) {
    EXPECT_TRUE(log.phase == phase);
    EXPECT_EQ(log.candidates, diagnostic.generated);
    EXPECT_EQ(log.valid_candidates, diagnostic.valid);
    EXPECT_EQ(log.accepted, diagnostic.improving > 0);
  };
  const auto normal = TuneOctoPipeAlgorithm1(
      initial.instance, initial.schedule.split, initial.schedule.orders, options);
  const auto& expected = diagnostic.improving ? *diagnostic.best_valid : initial;
  EXPECT_EQ(ToJson(normal.best.instance, normal.best.schedule),
            ToJson(expected.instance, expected.schedule));
}

TEST(OctoPipeDiagnostic, RequestedCaseHasAnImprovementOnlyInPartition) {
  const auto initial = Initial();
  const auto partition = DiagnoseOctoPipePhase(initial, OctoPipePhase::kPartition);
  const auto placement = DiagnoseOctoPipePhase(initial, OctoPipePhase::kFixedStagePlacement);
  const auto schedule = DiagnoseOctoPipePhase(initial, OctoPipePhase::kScheduleFBOnly);
  ASSERT_TRUE(partition.best_valid.has_value());
  ASSERT_TRUE(placement.best_valid.has_value());
  ASSERT_TRUE(schedule.best_valid.has_value());
  EXPECT_EQ(partition.valid, 14);
  EXPECT_EQ(placement.valid, 24);
  EXPECT_EQ(schedule.valid, 13);
  EXPECT_EQ(partition.improving, 1);
  EXPECT_EQ(placement.improving, 0);
  EXPECT_EQ(schedule.improving, 0);
  EXPECT_EQ(partition.best_valid->schedule.makespan, 447);
  EXPECT_EQ(placement.best_valid->schedule.makespan, 472);
  EXPECT_EQ(schedule.best_valid->schedule.makespan, 456);
  EXPECT_TRUE(partition.best_valid->schedule.split == (std::vector<Tick>{8, 8, 8, 7, 9, 8, 8, 8}));
}

TEST(OctoPipeDiagnostic, EmptyNeighborhoodAndDisabledPlacement) {
  auto initial = Initial();
  OctoPipeOptions options;
  options.tune_placement = false;
  EXPECT_THROW(DiagnoseOctoPipePhase(initial, OctoPipePhase::kFixedStagePlacement, options), Error);
  auto i = initial.instance;
  i.stages = 1; i.workers = 1; i.total_layers = 1;
  initial = *EvaluateOctoPipeCandidate(i, UniformSplit(i), BreadthFirstOrders(i));
  const auto empty = DiagnoseOctoPipePhase(initial, OctoPipePhase::kPartition);
  EXPECT_EQ(empty.generated, 0);
  EXPECT_EQ(empty.valid, 0);
  EXPECT_EQ(empty.improving, 0);
  EXPECT_FALSE(empty.best_valid.has_value());
}

TEST(OctoPipeDiagnostic, UniqueInitializationsPreserveEqualLoadNotBubbleMagnitude) {
  const auto bf = Initial();
  const auto& i = bf.instance;
  const auto interleaved = EvaluateOctoPipeCandidate(
      i, bf.schedule.split, InterleavedOneFOneBOrders(i));
  ASSERT_TRUE(interleaved.has_value());
  EXPECT_TRUE(FindEvaluationMethod(kUniformInterleavedOneFOneBMethod) == nullptr);
  EXPECT_TRUE(FindEvaluationMethod("interleaved-1f1b") != nullptr);
  EXPECT_FALSE(bf.schedule.orders == interleaved->schedule.orders);
  EXPECT_FALSE(ExtractMachinePredecessors(i, bf.schedule.orders) ==
               ExtractMachinePredecessors(i, interleaved->schedule.orders));
  for (const auto& state : {bf, *interleaved}) {
    CheckState(state);
    const auto m = ComputeOctoPipeBubbleMetrics(i, state.schedule);
    const bool is_bf = state.schedule.orders == bf.schedule.orders;
    EXPECT_EQ(state.schedule.makespan, is_bf ? 456 : 840);
    EXPECT_EQ(m.delta_b, 0);
    EXPECT_EQ(m.boundary_sum, 144);
    EXPECT_EQ(m.residual_sum, is_bf ? 144 : 1680);
    EXPECT_EQ(MinimumOctoPipeLayerComputeCost(i), 3);
    EXPECT_TRUE(SelectOctoPipePhase(m, MinimumOctoPipeLayerComputeCost(i)) == OctoPipePhase::kScheduleFBOnly);
    EXPECT_TRUE(SelectOctoPipePhase(m, std::min(i.backward_ratio_den, i.backward_ratio_num)) ==
                OctoPipePhase::kScheduleFBOnly);
    for (Index w = 0; w < i.workers; ++w) {
      Tick busy = 0;
      for (auto id : state.schedule.orders[w]) {
        const auto& op = state.schedule.operations_by_id[id.value];
        busy += op.end - op.start;
      }
      EXPECT_EQ(busy, 384);
      EXPECT_EQ(m.boundary[w], 24 * w);
      EXPECT_EQ(m.bubble[w], state.schedule.makespan - busy);
      EXPECT_EQ(m.residual[w], m.bubble[w] - m.boundary[w]);
    }
  }
}

TEST(OctoPipeDiagnostic, InterleavedPhasesAreIndependentAndMatchProductionAcceptance) {
  const auto i = Initial().instance;
  const auto initial = *EvaluateOctoPipeCandidate(i, UniformSplit(i), InterleavedOneFOneBOrders(i));
  const auto snapshot = ToJson(i, initial.schedule);
  Index index = 0;
  for (auto phase : {OctoPipePhase::kPartition, OctoPipePhase::kFixedStagePlacement,
                     OctoPipePhase::kScheduleFBOnly}) {
    const auto clone = initial;
    const auto report = DiagnoseOctoPipePhase(clone, phase);
    EXPECT_EQ(ToJson(i, clone.schedule), snapshot);
    EXPECT_EQ(ToJson(i, initial.schedule), snapshot);
    EXPECT_EQ(report.generated, (std::vector<Index>{14, 24, 64})[index]);
    EXPECT_EQ(report.valid, (std::vector<Index>{14, 24, 21})[index]);
    EXPECT_EQ(report.improving, (std::vector<Index>{3, 16, 6})[index]);
    ASSERT_TRUE(report.best_valid.has_value());
    CheckState(*report.best_valid);
    EXPECT_EQ(report.best_valid->schedule.makespan, (std::vector<Tick>{837, 728, 832})[index++]);
    for (const auto& proposal : OctoPipeNeighborProposals(initial, phase, {})) {
      const auto candidate = EvaluateOctoPipeCandidate(proposal.instance, proposal.schedule.split,
                                                       proposal.schedule.orders);
      if (candidate) CheckState(*candidate);
    }
  }
  OctoPipeOptions options;
  options.max_iterations = 1;
  const auto normal = TuneOctoPipeAlgorithm1(i, initial.schedule.split, initial.schedule.orders, options);
  const auto scheduling = DiagnoseOctoPipePhase(initial, OctoPipePhase::kScheduleFBOnly);
  ASSERT_TRUE(scheduling.best_valid.has_value());
  EXPECT_EQ(normal.best.schedule.makespan, 832);
  EXPECT_EQ(ToJson(normal.best.instance, normal.best.schedule),
            ToJson(scheduling.best_valid->instance, scheduling.best_valid->schedule));
}
}  // namespace
