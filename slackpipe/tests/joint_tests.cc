#include <gtest/gtest.h>
#include <algorithm>
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/result_validator.h"

namespace {
slackpipe::Instance Tiny() {
  slackpipe::Instance in;
  in.microbatches = in.stages = in.workers = 2;
  in.total_layers = 4;
  in.backward_ratio_num = 2;
  return in;
}
}

TEST(EvaluationMethods, FiveMethodsWithNoRetiredAliases) {
  EXPECT_EQ(slackpipe::EvaluationMethodRegistry().size(), 5u);
  for (const auto& name : {"1f1b", "interleaved-1f1b", "slack-heuristic", "octopipe-tuning", "cpsat-no-hint"})
    EXPECT_TRUE(slackpipe::FindEvaluationMethod(name) != nullptr);
  for (const auto& name : {"slackpipe", "heuristic", "uniform-breadth-first", "uniform-interleaved-1f1b", "joint"})
    EXPECT_TRUE(slackpipe::FindEvaluationMethod(name) == nullptr);
}

TEST(EvaluationMethods, ExactDisplayNamesInRegistryAndJson) {
  const std::pair<const char*, const char*> names[] = {
      {"1f1b", "1F1B"},
      {"interleaved-1f1b", "Interleaved 1F1B"},
      {"octopipe-tuning", "OctoPipe-Tuning"},
      {"cpsat-no-hint", "CP-SAT without hints"},
      {"slack-heuristic", "Slack-based Heuristic"}};
  for (const auto& [id, display] : names) {
    const auto* method = slackpipe::FindEvaluationMethod(id);
    ASSERT_TRUE(method != nullptr);
    EXPECT_EQ(method->display_name, display);
    const auto json = slackpipe::DescribeEvaluationMethodJson(id);
    EXPECT_TRUE(json.find("\"display_name\":\"" + std::string(display) + "\"") != std::string::npos);
    auto renamed = *method;
    renamed.display_name = "Presentation only";
    EXPECT_EQ(slackpipe::EvaluationMethodContractHash(*method),
              slackpipe::EvaluationMethodContractHash(renamed));
  }
}

TEST(OrToolsJoint, PureModelHasNoHintOrFallbackAndMatchesKnownOptimum) {
  const auto in = Tiny();
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10;
  auto result = slackpipe::OptimizeJointSplitAndScheduleCpSat(in, options);
  if (!slackpipe::IsJointOptimizerAvailable()) {
    EXPECT_EQ(result.status, "UNAVAILABLE");
    EXPECT_FALSE(result.final_solution_available);
    EXPECT_FALSE(result.cp_sat_launched);
    return;
  }
  ASSERT_TRUE(result.final_solution_available);
  EXPECT_TRUE(result.proven_optimal);
  EXPECT_EQ(result.solver_status_raw, "OPTIMAL");
  EXPECT_EQ(result.makespan_ticks, 18);
  EXPECT_EQ(result.external_hint_count, 0);
  EXPECT_EQ(result.conservative_horizon, 24);
  EXPECT_EQ(result.workload_constraint_count, 3);
  EXPECT_EQ(result.fifo_constraint_count, 5);
  result.requested_command = "solver --path \"line1\nline2\tprofile\"";
  EXPECT_TRUE(slackpipe::ValidateResultJsonText(slackpipe::ToJson(in, result)).passed);
}

TEST(OrToolsJoint, InfeasibleCapacityHasNoExternalSolutionOrFallback) {
  auto in = Tiny();
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10;
  options.activation_options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_options.activation_cap_units = {0};
  options.activation_options.enforce_activation_cap = true;
  const auto result = slackpipe::OptimizeJointSplitAndScheduleCpSat(in, options);
  EXPECT_FALSE(result.final_solution_available);
  EXPECT_FALSE(result.proven_optimal);
  EXPECT_TRUE(result.schedule.operations_by_id.empty());
  EXPECT_EQ(result.solution_source, "none");
  if (slackpipe::IsJointOptimizerAvailable()) {
    EXPECT_TRUE(result.cp_sat_launched);
    EXPECT_EQ(result.solver_status_raw, "INFEASIBLE");
    EXPECT_EQ(result.external_hint_count, 0);
    EXPECT_EQ(result.conservative_horizon, 24);
  } else {
    EXPECT_EQ(result.solver_status_raw, "UNAVAILABLE");
    EXPECT_FALSE(result.solver_bound_available);
  }
  const auto json = slackpipe::ToJson(in, result);
  EXPECT_TRUE(json.find("\"makespan_ticks\": null") != std::string::npos);
  EXPECT_TRUE(json.find("\"fallback_used\": false") != std::string::npos);
  EXPECT_TRUE(json.find("\"microbatches\": 2") != std::string::npos);
  EXPECT_TRUE(json.find("\"stages\": 2") != std::string::npos);
  EXPECT_TRUE(json.find("\"workers\": 2") != std::string::npos);
}

TEST(OrToolsJoint, ActivationCapIsEnforcedOrExplicitlyUnavailable) {
  if (!slackpipe::IsJointOptimizerAvailable()) return;
  const auto in = Tiny();
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10;
  options.activation_options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_options.activation_cap_units = {0};
  options.activation_options.enforce_activation_cap = true;
  auto result = slackpipe::OptimizeJointSplitAndScheduleCpSat(in, options);
  EXPECT_FALSE(result.final_solution_available);
  EXPECT_TRUE(result.solver_status_raw == "INFEASIBLE" || result.status == "UNAVAILABLE");
  options.activation_options.activation_cap_units = {8};
  result = slackpipe::OptimizeJointSplitAndScheduleCpSat(in, options);
  if (result.status == "UNAVAILABLE") return;
  ASSERT_TRUE(result.final_solution_available);
  EXPECT_TRUE(result.activation_cap_constraints.constraints_added);
  EXPECT_TRUE(slackpipe::ActivationScheduleSatisfiesCap(in, result.schedule, options.activation_options));
}

TEST(OrToolsJoint, JointPartitionAndOrderMatchesIndependentTinyEnumeration) {
  if (!slackpipe::IsJointOptimizerAvailable()) return;
  auto in = Tiny();
  in.profile_forward_slope_ticks = {1, 7};
  in.profile_backward_slope_ticks = {9, 2};
  in.profile_forward_bias_ticks = {2, 1};
  in.profile_backward_bias_ticks = {1, 3};
  in.communication_ticks = 3;
  slackpipe::Tick optimum = slackpipe::kTickMax;
  auto orders = slackpipe::BreadthFirstOrders(in);
  auto less = [](auto a, auto b) { return a.value < b.value; };
  std::sort(orders[0].begin(), orders[0].end(), less);
  do {
    std::sort(orders[1].begin(), orders[1].end(), less);
    do {
      for (slackpipe::Tick cut = 1; cut < in.total_layers; ++cut) {
        const auto candidate = slackpipe::EvaluateSchedule(in, {cut, in.total_layers - cut}, orders);
        if (candidate.schedule.ok()) {
          EXPECT_TRUE(slackpipe::ValidateScheduleSolutionIndependent(in, candidate.schedule).passed);
          optimum = std::min(optimum, candidate.schedule.makespan);
        }
      }
    } while (std::next_permutation(orders[1].begin(), orders[1].end(), less));
  } while (std::next_permutation(orders[0].begin(), orders[0].end(), less));
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10;
  const auto result = slackpipe::OptimizeJointSplitAndScheduleCpSat(in, options);
  ASSERT_TRUE(result.proven_optimal);
  EXPECT_EQ(result.makespan_ticks, optimum);
  EXPECT_TRUE(result.best_bound_ticks <= optimum);
  EXPECT_TRUE(slackpipe::ValidateResultJsonText(slackpipe::ToJson(in, result)).passed);
}
