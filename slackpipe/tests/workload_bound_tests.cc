#include <algorithm>
#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

#include "slackpipe/cost_profile.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/instance.h"
#include "slackpipe/workload_bound.h"

namespace {
using namespace slackpipe;

Instance Tiny() {
  Instance i;
  i.microbatches = 2; i.stages = 2; i.workers = 2; i.total_layers = 4;
  i.backward_ratio_num = 2;
  return i;
}

Instance Affine() {
  auto i = Tiny();
  i.profile_forward_slope_ticks = {2, 2};
  i.profile_backward_slope_ticks = {3, 3};
  i.profile_forward_bias_ticks = {1, 2};
  i.profile_backward_bias_ticks = {3, 4};
  return i;
}

Instance Range() {
  auto i = Tiny();
  i.profile_prefix_forward_ticks = {0, 1, 3, 6, 10};
  i.profile_prefix_backward_ticks = {0, 2, 6, 12, 20};
  i.profile_role_forward_bias_ticks = {1, 9, 2};
  i.profile_role_backward_bias_ticks = {1, 9, 3};
  return i;
}

Tick Work(const Instance& i, const std::vector<Tick>& split) {
  Tick total = 0;
  for (Index s = 0; s < i.stages; ++s) {
    total += i.microbatches * (i.Duration(s, false, split) + i.Duration(s, true, split));
  }
  return total;
}

TEST(WorkloadBound, InvariantCostsMatchEveryPartition) {
  for (const auto& i : {Tiny(), Affine(), Range()}) {
    const auto bound = ComputeWorkloadLowerBound(i);
    ASSERT_TRUE(bound.total_work_ticks.has_value());
    ASSERT_TRUE(bound.makespan_ticks.has_value());
    EXPECT_TRUE(bound.partition_invariant);
    for (Tick cut = 1; cut < i.total_layers; ++cut) {
      const Tick work = Work(i, {cut, i.total_layers - cut});
      EXPECT_EQ(*bound.total_work_ticks, work);
      EXPECT_EQ(*bound.makespan_ticks, work / i.workers + (work % i.workers != 0));
    }
  }
}

TEST(WorkloadBound, VariableAffineCostsUseGlobalMinimumNotIncumbentWork) {
  auto i = Affine();
  i.profile_forward_slope_ticks = {1, 10};
  i.profile_backward_slope_ticks = {2, 20};
  const auto bound = ComputeWorkloadLowerBound(i);
  ASSERT_TRUE(bound.total_work_ticks.has_value());
  EXPECT_FALSE(bound.partition_invariant);
  Tick minimum = kTickMax;
  for (Tick cut = 1; cut < i.total_layers; ++cut) {
    minimum = std::min(minimum, Work(i, {cut, i.total_layers - cut}));
  }
  EXPECT_EQ(*bound.total_work_ticks, minimum);
  EXPECT_TRUE(*bound.total_work_ticks < Work(i, {1, 3}));
  // Fixed combined slopes suffice even when each direction varies by stage.
  i.profile_backward_slope_ticks = {20, 11};
  EXPECT_TRUE(ComputeWorkloadLowerBound(i).partition_invariant);
}

TEST(WorkloadBound, PrefixRoundingRoleBiasAndZeroDurationSemantics) {
  auto i = Tiny();
  const auto path = std::filesystem::temp_directory_path() / "slackpipe_workload_rounding_test.json";
  WriteTextFile(path.string(), R"({"schema_version":"slackpipe.cost_profile.v2",
    "prefix_forward_us":[0,0.1,0.2,0.4,1.6],
    "prefix_backward_us":[0,0.2,0.3,0.4,2.5],
    "stage_role_bias_us":{"first":{"forward":0.1,"backward":0},
      "middle":{"forward":99,"backward":99},"last":{"forward":0,"backward":0.1}}})");
  ApplyCostProfileFile(i, path.string());
  std::filesystem::remove(path);
  EXPECT_TRUE(i.profile_prefix_forward_ticks == (std::vector<Tick>{0, 1, 1, 1, 2}));
  EXPECT_TRUE(i.profile_prefix_backward_ticks == (std::vector<Tick>{0, 1, 1, 1, 3}));
  EXPECT_EQ(i.BackwardDuration(0, 1, 2), 0);  // No post-difference min-one clamp.
  auto bound = ComputeWorkloadLowerBound(i);
  ASSERT_TRUE(bound.total_work_ticks.has_value());
  EXPECT_EQ(*bound.total_work_ticks, 14);
  i.stages = 1;
  bound = ComputeWorkloadLowerBound(i);
  EXPECT_EQ(*bound.total_work_ticks, 12);  // N=1 uses only the first role.
  i.microbatches = 1; i.workers = 4;
  bound = ComputeWorkloadLowerBound(i);
  EXPECT_EQ(*bound.makespan_ticks, 2);  // ceil(6/4), not floating rounding.
}

TEST(WorkloadBound, OverflowIsExplicitlyUnavailable) {
  auto i = Tiny();
  i.backward_ratio_num = kTickMax;
  const auto bound = ComputeWorkloadLowerBound(i);
  EXPECT_FALSE(bound.total_work_ticks.has_value());
  EXPECT_FALSE(bound.makespan_ticks.has_value());
  EXPECT_FALSE(bound.reason.empty());
}

TEST(WorkloadBound, RawBoundAndEffectiveGapRemainDistinct) {
  const auto i = Range();
  const auto report = ComputeLowerBoundReport(i, 10, 54);
  EXPECT_EQ(*report.raw_solver_bound_ticks, 10);
  EXPECT_EQ(*report.workload.makespan_ticks, 37);
  EXPECT_EQ(*report.effective_lower_bound_ticks, 37);
  EXPECT_TRUE(std::abs(*report.effective_relative_gap - 17.0 / 54) < 1e-12);
  EXPECT_FALSE(ComputeLowerBoundReport(i, 10, std::nullopt).effective_relative_gap.has_value());
  EXPECT_FALSE(ComputeLowerBoundReport(i, 100, 54).effective_relative_gap.has_value());
}

TEST(WorkloadBound, JointOptimaUnchangedAndProfileCapacityConstraintsPresent) {
  if (!IsJointOptimizerAvailable()) return;
  Index index = 0;
  for (const auto& i : {Tiny(), Affine(), Range()}) {
    JointOptimizerOptions options;
    options.time_limit_seconds = 10;
    const auto result = OptimizeJointSplitAndScheduleCpSat(i, options);
    ASSERT_TRUE(result.proven_optimal);
    EXPECT_EQ(result.makespan_ticks, (std::vector<Tick>{18, 42, 54})[index++]);
    EXPECT_EQ(result.workload_constraint_count, i.workers + 1);
    const auto bound = ComputeWorkloadLowerBound(i);
    ASSERT_TRUE(bound.makespan_ticks.has_value());
    EXPECT_TRUE(*bound.makespan_ticks <= result.makespan_ticks);
  }
}

TEST(WorkloadBound, ConditionalBoundDoesNotCertifyJointOptimality) {
  const auto report = ComputeLowerBoundReport(Range(), 63, 63, false);
  EXPECT_EQ(*report.raw_solver_bound_ticks, 63);
  EXPECT_FALSE(report.raw_bound_globally_valid);
  EXPECT_EQ(*report.effective_lower_bound_ticks, 37);
}
}  // namespace
