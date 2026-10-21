#include <gtest/gtest.h>
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/io.h"

TEST(Evaluator, RejectsNonadjacentForwardAndBackwardFifoReversals) {
  slackpipe::Instance in;
  in.microbatches = 3;
  in.stages = in.workers = in.total_layers = 1;
  for (bool backward : {false, true}) {
    // Both orders are valid with FIFO disabled; the reversed occurrences
    // are separated by an operation of the other phase.
    slackpipe::MachineOrders orders(1);
    orders[0] = backward
        ? std::vector<slackpipe::OperationId>{{0}, {2}, {3}, {4}, {1}, {5}}
        : std::vector<slackpipe::OperationId>{{4}, {5}, {2}, {3}, {0}, {1}};
    EXPECT_FALSE(slackpipe::EvaluateSchedule(in, {1}, orders).schedule.ok());
    EXPECT_TRUE(slackpipe::EvaluateSchedule(in, {1}, orders, false).schedule.ok());
  }
}

TEST(OrToolsProfileJoint, PrefixCostsAndLargeCommunicationAreActuallyUsed) {
  if (!slackpipe::IsJointOptimizerAvailable()) return;
  for (slackpipe::Tick communication : {0, 1000}) {
    slackpipe::Instance in;
    in.microbatches = 2; in.stages = 3; in.workers = 2; in.total_layers = 6;
    in.communication_ticks = communication;
    in.profile_prefix_forward_ticks = {0, 1, 19, 21, 24, 40, 41};
    in.profile_prefix_backward_ticks = {0, 13, 14, 17, 37, 39, 47};
    in.profile_role_forward_bias_ticks = {7, 3, 11};
    in.profile_role_backward_bias_ticks = {2, 5, 13};
    slackpipe::JointOptimizerOptions options;
    options.time_limit_seconds = 10;
    const auto result = slackpipe::OptimizeJointSplitAndScheduleCpSat(in, options);
    ASSERT_TRUE(result.final_solution_available);
    EXPECT_EQ(result.workload_constraint_count, 3);
    EXPECT_EQ(result.external_hint_count, 0);
    EXPECT_TRUE(slackpipe::ValidateScheduleSolutionIndependent(in, result.schedule).passed);
    slackpipe::Tick total = 0;
    for (const auto& operation : result.schedule.operations_by_id) {
      const auto view = slackpipe::DecodeOperation(in, operation.id);
      EXPECT_EQ(operation.duration, in.Duration(view.stage, view.backward, result.split));
      total += operation.duration;
    }
    EXPECT_EQ(slackpipe::ComputeScheduleMetrics(in, result.schedule).total_useful_work, total);
  }
}
