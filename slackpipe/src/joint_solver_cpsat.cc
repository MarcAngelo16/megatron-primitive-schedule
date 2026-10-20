#include "slackpipe/joint_solver.h"

#if SLACKPIPE_HAVE_ORTOOLS

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <cmath>
#include <utility>
#include <vector>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "slackpipe/activation_cpsat.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/deadline.h"
#include "slackpipe/hybrid_slack.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/slackpipe_solver.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;
using operations_research::Domain;
using operations_research::sat::BoolVar;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::IntervalVar;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::Model;
using operations_research::sat::NewFeasibleSolutionObserver;
using operations_research::sat::NewSatParameters;
using operations_research::sat::SatParameters;
using operations_research::sat::SolutionIntegerValue;
using operations_research::sat::SolveCpModel;

constexpr double kIncumbentBudgetFraction = kSlackPipePreparationBudgetFraction;

struct RawOperationTime {
  OperationId id;
  Tick start = 0;
  Tick end = 0;
};

[[nodiscard]] std::string EffectiveBfsMethodName(const std::string& method);

double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

double IncumbentPhaseCap(double total_limit_seconds) {
  if (total_limit_seconds <= 0.0) return 0.0;
  return total_limit_seconds * kIncumbentBudgetFraction;
}

[[nodiscard]] Index JointFifoConstraintCount(
    const Instance& instance, const JointOptimizerOptions& options) {
  if (!options.fifo_ordering || instance.microbatches <= 1) return 0;
  Index count =
      CheckedMul(instance.microbatches - 1, OperationPositionCount(instance),
                 "joint FIFO constraint count");
  if (options.symmetry_break_f0_fifo) {
    count = CheckedAdd(count, instance.microbatches - 1,
                       "joint F0 FIFO symmetry count");
  }
  return count;
}

[[nodiscard]] Tick ConservativeJointHorizon(const Instance& instance) {
  if (instance.HasRangeCostProfile()) {
    const Tick forward_total = instance.profile_prefix_forward_ticks.back();
    const Tick backward_total = instance.profile_prefix_backward_ticks.back();
    Tick max_bias = 0;
    for (Tick value : instance.profile_role_forward_bias_ticks) {
      max_bias = std::max(max_bias, value);
    }
    for (Tick value : instance.profile_role_backward_bias_ticks) {
      max_bias = std::max(max_bias, value);
    }
    const Tick max_duration =
        CheckedAdd(std::max(forward_total, backward_total), max_bias,
                   "joint range profile horizon duration");
    const Tick compute = CheckedMul(instance.OperationCount(), max_duration,
                                    "joint range profile horizon compute");
    const Tick data_edges = CheckedMul(
        instance.microbatches,
        CheckedAdd(CheckedMul(2, instance.stages, "joint data edge count"), -1,
                   "joint data edge count"),
        "joint data edge count");
    const Tick communication =
        CheckedMul(data_edges, instance.communication_ticks,
                   "joint horizon communication");
    return CheckedAdd(compute, communication, "joint range profile horizon");
  }
  if (instance.HasAffineCostProfile()) {
    Tick max_slope = 0;
    Tick max_bias = 0;
    for (Index stage = 0; stage < instance.stages; ++stage) {
      const auto index = static_cast<std::size_t>(stage);
      max_slope =
          std::max(max_slope, instance.profile_forward_slope_ticks[index]);
      max_slope =
          std::max(max_slope, instance.profile_backward_slope_ticks[index]);
      max_bias = std::max(max_bias, instance.profile_forward_bias_ticks[index]);
      max_bias =
          std::max(max_bias, instance.profile_backward_bias_ticks[index]);
    }
    const Tick max_duration =
        CheckedAdd(CheckedMul(instance.total_layers, max_slope,
                              "joint profile horizon duration"),
                   max_bias, "joint profile horizon duration");
    const Tick compute = CheckedMul(instance.OperationCount(), max_duration,
                                    "joint profile horizon compute");
    const Tick data_edges = CheckedMul(
        instance.microbatches,
        CheckedAdd(CheckedMul(2, instance.stages, "joint data edge count"), -1,
                   "joint data edge count"),
        "joint data edge count");
    const Tick communication =
        CheckedMul(data_edges, instance.communication_ticks,
                   "joint horizon communication");
    return CheckedAdd(compute, communication, "joint profile horizon");
  }
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "joint horizon ratio sum");
  const Tick compute =
      CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                            "joint horizon compute"),
                 ratio_sum, "joint horizon compute");
  const Tick data_edges = CheckedMul(
      instance.microbatches,
      CheckedAdd(CheckedMul(2, instance.stages, "joint data edge count"), -1,
                 "joint data edge count"),
      "joint data edge count");
  const Tick communication = CheckedMul(
      data_edges, instance.communication_ticks, "joint horizon communication");
  return CheckedAdd(compute, communication, "joint horizon");
}

[[nodiscard]] std::string LegacyBfsMethodForIncumbentBuild(
    const JointOptimizerOptions& options) {
  if (!options.incumbent_method) {
    return options.bfs_method;
  }
  if (*options.incumbent_method == "slack") {
    return "auto";
  }
  if (*options.incumbent_method == "canonical") {
    return "uniform";
  }
  if (*options.incumbent_method == "none") {
    return "none";
  }
  return *options.incumbent_method;
}

[[nodiscard]] std::string EffectiveIncumbentMechanismName(
    const std::string& requested, const BfsSplitOptimizationResult& incumbent) {
  if (requested == "none") {
    return "none";
  }
  if (requested == "canonical") {
    return incumbent.schedule.ok() ? "canonical" : "none";
  }
  if (requested == "slack") {
    return incumbent.schedule.ok() ? "slack" : "none";
  }
  return EffectiveBfsMethodName(incumbent.method);
}

[[nodiscard]] BfsSplitOptimizationResult NoExternalIncumbentResult() {
  BfsSplitOptimizationResult result;
  result.method = "none";
  result.status = "NOT_RUN";
  result.diagnostic = "external incumbent disabled";
  return result;
}

[[nodiscard]] std::string NoSolutionReasonForDeadlineFallback(
    const std::string& reason) {
  if (reason == "global_deadline_expired_after_replay_cap_rejections") {
    return "activation_cap_replay_rejected_until_deadline";
  }
  return "global_deadline_before_valid_solution";
}

[[nodiscard]] std::string NoSolutionReasonForCpSatStatus(
    CpSolverStatus status) {
  if (status == operations_research::sat::INFEASIBLE) {
    return "solver_no_feasible_solution";
  }
  return "global_deadline_before_valid_solution";
}

[[nodiscard]] bool BfsIncumbentCompleteForInstance(
    const Instance& instance, const BfsSplitOptimizationResult& incumbent) {
  return incumbent.schedule.ok() && incumbent.makespan_ticks > 0 &&
         incumbent.split.size() == static_cast<std::size_t>(instance.stages) &&
         incumbent.schedule.operations_by_id.size() ==
             static_cast<std::size_t>(instance.OperationCount());
}

[[nodiscard]] bool ScheduleCompleteForInstance(
    const Instance& instance, const ScheduleSolution& schedule) {
  return schedule.ok() &&
         schedule.operations_by_id.size() ==
             static_cast<std::size_t>(instance.OperationCount());
}

void EmitLifecycle(const JointOptimizerOptions& options,
                   Clock::time_point started, const std::string& phase,
                   const std::string& status = "",
                   double effective_limit_seconds = 0.0,
                   const std::string& detail = "",
                   double requested_limit_seconds = 0.0,
                   double remaining_global_time_seconds = 0.0,
                   double phase_specific_cap_seconds = 0.0) {
  if (!options.lifecycle) return;
  LifecycleEvent event;
  event.phase = phase;
  event.algorithm = "optimize-joint";
  event.solver_status = status;
  event.elapsed_seconds = Since(started);
  event.configured_solver_limit_seconds = options.time_limit_seconds;
  event.effective_solver_limit_seconds = effective_limit_seconds;
  event.requested_solver_limit_seconds = requested_limit_seconds;
  event.remaining_global_time_seconds = remaining_global_time_seconds;
  event.phase_specific_cap_seconds = phase_specific_cap_seconds;
  event.solver_threads = options.num_workers;
  event.detail = detail;
  options.lifecycle(event);
}

[[nodiscard]] std::string StatusName(CpSolverStatus status) {
  switch (status) {
    case operations_research::sat::OPTIMAL:
      return "OPTIMAL";
    case operations_research::sat::FEASIBLE:
      return "FEASIBLE";
    case operations_research::sat::INFEASIBLE:
      return "INFEASIBLE";
    case operations_research::sat::MODEL_INVALID:
      return "MODEL_INVALID";
    case operations_research::sat::UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNRECOGNIZED";
  }
}

[[nodiscard]] LinearExpr OperationDurationExpr(
    const Instance& instance, const std::vector<IntVar>& layers,
    Index chain_index) {
  const bool backward = chain_index >= instance.stages;
  const Index stage = StageForOperationPosition(instance, chain_index);
  const Tick coefficient =
      instance.HasAffineCostProfile()
          ? (backward
                 ? instance
                       .profile_backward_slope_ticks[static_cast<std::size_t>(
                           stage)]
                 : instance
                       .profile_forward_slope_ticks[static_cast<std::size_t>(
                           stage)])
          : (backward ? instance.backward_ratio_num
                      : instance.backward_ratio_den);
  const Tick bias =
      instance.HasAffineCostProfile()
          ? (backward
                 ? instance
                       .profile_backward_bias_ticks[static_cast<std::size_t>(
                           stage)]
                 : instance.profile_forward_bias_ticks[static_cast<std::size_t>(
                       stage)])
          : 0;
  return coefficient * layers[static_cast<std::size_t>(stage)] + bias;
}

[[nodiscard]] LinearExpr ChainPrefixExpr(const Instance& instance,
                                         const std::vector<IntVar>& layers,
                                         Index chain_index) {
  LinearExpr expr;
  for (Index n = 0; n < chain_index; ++n) {
    expr += OperationDurationExpr(instance, layers, n);
  }
  return expr;
}

[[nodiscard]] LinearExpr ChainSuffixExpr(const Instance& instance,
                                         const std::vector<IntVar>& layers,
                                         Index chain_index) {
  LinearExpr expr;
  for (Index n = chain_index + 1; n < 2 * instance.stages; ++n) {
    expr += OperationDurationExpr(instance, layers, n);
  }
  return expr;
}

[[nodiscard]] std::string JoinTicksForDiagnostic(
    const std::vector<Tick>& values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ",";
    out << values[i];
  }
  return out.str();
}

[[nodiscard]] std::string WorkerOrderSizesForDiagnostic(
    const MachineOrders& orders) {
  std::ostringstream out;
  for (std::size_t i = 0; i < orders.size(); ++i) {
    if (i != 0) out << ",";
    out << orders[i].size();
  }
  return out.str();
}

[[nodiscard]] std::string BfsFallbackReplayMismatchMessage(
    const std::vector<Tick>& split, const MachineOrders& orders,
    Tick stored_makespan, const ScheduleSolution& replayed) {
  std::ostringstream msg;
  msg << "BFS incumbent fallback did not replay deterministically: "
      << "stored_makespan=" << stored_makespan
      << " replay_ok=" << (replayed.ok() ? "true" : "false")
      << " replay_makespan=" << replayed.makespan << " split=["
      << JoinTicksForDiagnostic(split) << "] worker_order_sizes=["
      << WorkerOrderSizesForDiagnostic(orders) << "]";
  if (!replayed.validation_errors.empty()) {
    msg << " validation_error=" << replayed.validation_errors.front();
  }
  return msg.str();
}

[[nodiscard]] std::string ValidationFailureMessage(
    const ResultValidationResult& validation) {
  std::ostringstream msg;
  msg << validation.error_code;
  if (!validation.message.empty()) {
    msg << ": " << validation.message;
  }
  if (validation.offending_worker) {
    msg << " worker=" << *validation.offending_worker;
  }
  if (!validation.offending_operations.empty()) {
    msg << " operations=[";
    for (std::size_t i = 0; i < validation.offending_operations.size(); ++i) {
      if (i != 0) msg << ",";
      msg << validation.offending_operations[i];
    }
    msg << "]";
  }
  return msg.str();
}

[[nodiscard]] std::string AppendReason(const std::string& prefix,
                                       const std::string& reason) {
  if (prefix.empty()) return reason;
  if (reason.empty()) return prefix;
  return prefix + ": " + reason;
}

[[nodiscard]] BfsSplitOptimizationResult BuildBfsIncumbentForSplit(
    const Instance& instance, const std::vector<Tick>& split,
    const std::string& method) {
  const auto started = Clock::now();
  ValidateSplit(instance, split);
  BfsSplitOptimizationResult result;
  result.method = method;
  result.status = "FEASIBLE";
  result.proven_optimal = false;
  result.split = split;
  result.best_bound_ticks = 0;
  result.machine_orders = BreadthFirstOrders(instance);
  EvaluationResult evaluated =
      EvaluateSchedule(instance, result.split, result.machine_orders);
  if (!evaluated.schedule.ok()) {
    throw Error("failed to construct a valid BFS incumbent for joint solver");
  }
  result.makespan_ticks = evaluated.schedule.makespan;
  result.schedule = std::move(evaluated.schedule);
  const ResultValidationResult validation = ValidateScheduleSolutionIndependent(
      instance, result.schedule, "FEASIBLE", method);
  if (!validation.passed) {
    throw Error(
        "failed to independently validate BFS incumbent for joint "
        "solver: " +
        ValidationFailureMessage(validation));
  }
  result.timing.incumbent_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  result.timing.total_seconds = result.timing.incumbent_seconds;
  result.wall_time_seconds = result.timing.total_seconds;
  return result;
}

[[nodiscard]] bool ValidateBfsIncumbentForUse(
    const Instance& instance, BfsSplitOptimizationResult* incumbent,
    const std::optional<PartitionRestriction>& restriction,
    std::string* rejection_reason) {
  auto reject = [&](const std::string& reason) {
    if (rejection_reason != nullptr) *rejection_reason = reason;
    return false;
  };
  try {
    ValidateSplit(instance, incumbent->split);
    if (restriction && !SplitSatisfiesPartitionRestriction(
                           instance, incumbent->split, *restriction)) {
      return reject("incumbent violates active partition restriction");
    }
    if (!incumbent->schedule.ok()) {
      std::string reason = "incumbent has invalid schedule";
      if (!incumbent->schedule.validation_errors.empty()) {
        reason += ": " + incumbent->schedule.validation_errors.front();
      }
      return reject(reason);
    }
    EvaluationResult checked =
        EvaluateSchedule(instance, incumbent->split, incumbent->machine_orders);
    if (!checked.schedule.ok()) {
      std::string reason = "incumbent does not replay deterministically";
      if (!checked.schedule.validation_errors.empty()) {
        reason += ": " + checked.schedule.validation_errors.front();
      }
      return reject(reason);
    }
    if (checked.schedule.makespan != incumbent->makespan_ticks) {
      return reject(BfsFallbackReplayMismatchMessage(
          incumbent->split, incumbent->machine_orders,
          incumbent->makespan_ticks, checked.schedule));
    }
    const ResultValidationResult validation =
        ValidateScheduleSolutionIndependent(
            instance, checked.schedule, "FEASIBLE",
            incumbent->method.empty() ? std::string("joint-incumbent")
                                      : incumbent->method);
    if (!validation.passed) {
      return reject("independent validation failed: " +
                    ValidationFailureMessage(validation));
    }
    incumbent->schedule = std::move(checked.schedule);
    return true;
  } catch (const Error& error) {
    return reject(error.what());
  }
}

void RequireValidBfsIncumbentForUse(
    const Instance& instance, BfsSplitOptimizationResult* incumbent,
    const std::optional<PartitionRestriction>& restriction,
    const std::string& context) {
  std::string reason;
  if (!ValidateBfsIncumbentForUse(instance, incumbent, restriction, &reason)) {
    throw Error(AppendReason(context, reason));
  }
}

[[nodiscard]] BfsSplitOptimizationResult BuildBfsIncumbent(
    const Instance& instance, const JointOptimizerOptions& options,
    const Deadline& deadline, double incumbent_effective_budget_seconds,
    double incumbent_phase_cap_seconds) {
  const std::string requested_incumbent_method =
      RequestedIncumbentMethodName(options);
  const std::string build_method = LegacyBfsMethodForIncumbentBuild(options);
  if (requested_incumbent_method == "none") {
    return NoExternalIncumbentResult();
  }

  auto restricted_reference_fallback = [&]() {
    if (!options.partition_restriction) {
      return BuildBfsIncumbentForSplit(instance, UniformSplit(instance),
                                       "deterministic-uniform-bfs-fallback");
    }
    return BuildBfsIncumbentForSplit(
        instance, options.partition_restriction->reference_split,
        "restricted-reference-fallback");
  };

  auto validated_reference_fallback = [&]() {
    BfsSplitOptimizationResult fallback = restricted_reference_fallback();
    RequireValidBfsIncumbentForUse(
        instance, &fallback, options.partition_restriction,
        "joint solver safe reference fallback failed validation");
    return fallback;
  };

  auto copy_attempt_accounting =
      [&](const BfsSplitOptimizationResult& attempted,
          BfsSplitOptimizationResult* fallback) {
        fallback->best_bound_ticks = attempted.best_bound_ticks;
        fallback->solver_objective_ticks = attempted.solver_objective_ticks;
        fallback->wall_time_seconds = attempted.wall_time_seconds;
        fallback->timing = attempted.timing;
        fallback->branches = attempted.branches;
        fallback->conflicts = attempted.conflicts;
        fallback->checked_splits = attempted.checked_splits;
        fallback->cp_sat_models_solved = attempted.cp_sat_models_solved;
        fallback->deterministic_time = attempted.deterministic_time;
        fallback->hint_budget_seconds = attempted.hint_budget_seconds;
        fallback->hint_elapsed_seconds = attempted.hint_elapsed_seconds;
        fallback->hint_iterations = attempted.hint_iterations;
        fallback->hint_candidates_generated =
            attempted.hint_candidates_generated;
        fallback->hint_candidates_simulated =
            attempted.hint_candidates_simulated;
        fallback->hint_partition_moves_accepted =
            attempted.hint_partition_moves_accepted;
        fallback->hint_interleaving_moves_accepted =
            attempted.hint_interleaving_moves_accepted;
        fallback->hint_deadline_reached = attempted.hint_deadline_reached;
        fallback->hint_termination_reason = attempted.hint_termination_reason;
      };

  auto restricted_reference_fallback_after_attempt =
      [&](const BfsSplitOptimizationResult& attempted,
          const std::string& reason) {
        BfsSplitOptimizationResult fallback = validated_reference_fallback();
        copy_attempt_accounting(attempted, &fallback);
        fallback.proven_optimal = false;
        fallback.diagnostic =
            reason + "; attempted_incumbent_method=" + attempted.method +
            "; attempted_incumbent_status=" + attempted.status;
        return fallback;
      };

  auto maybe_replace_cap_violating_incumbent =
      [&](const BfsSplitOptimizationResult& incumbent,
          const std::string& reason) {
        if (!options.activation_options.enforce_activation_cap ||
            ActivationScheduleSatisfiesCap(instance, incumbent.schedule,
                                           options.activation_options)) {
          return incumbent;
        }
        BfsSplitOptimizationResult fallback = validated_reference_fallback();
        copy_attempt_accounting(incumbent, &fallback);
        fallback.proven_optimal = false;
        if (ActivationScheduleSatisfiesCap(instance, fallback.schedule,
                                           options.activation_options)) {
          fallback.diagnostic =
              reason +
              "; attempted incumbent violates activation cap; using "
              "cap-feasible reference fallback";
          return fallback;
        }
        BfsSplitOptimizationResult rejected = incumbent;
        rejected.diagnostic =
            reason +
            "; attempted incumbent violates activation cap; reference "
            "fallback also violates activation cap";
        return rejected;
      };

  if (options.bfs_incumbent_override) {
    BfsSplitOptimizationResult incumbent = *options.bfs_incumbent_override;
    std::string reason;
    if (!ValidateBfsIncumbentForUse(instance, &incumbent,
                                    options.partition_restriction, &reason)) {
      return restricted_reference_fallback_after_attempt(
          incumbent, "joint solver incumbent override rejected: " + reason);
    }
    return maybe_replace_cap_violating_incumbent(
        incumbent, "joint solver incumbent override accepted structurally");
  }

  if (options.time_limit_seconds > 0.0 &&
      incumbent_effective_budget_seconds <= 0.0) {
    BfsSplitOptimizationResult fallback = validated_reference_fallback();
    fallback.diagnostic =
        "BFS incumbent solve skipped because the incumbent budget was "
        "exhausted";
    return fallback;
  }

  if (requested_incumbent_method == "canonical") {
    BfsSplitOptimizationResult incumbent =
        options.partition_restriction
            ? BuildBfsIncumbentForSplit(
                  instance, options.partition_restriction->reference_split,
                  "restricted-reference-fallback")
            : BuildBfsIncumbentForSplit(instance, UniformSplit(instance),
                                        "uniform");
    RequireValidBfsIncumbentForUse(
        instance, &incumbent, options.partition_restriction,
        "canonical joint incumbent failed validation");
    return maybe_replace_cap_violating_incumbent(
        incumbent, "canonical joint incumbent accepted structurally");
  }

  if (build_method == "heuristic") {
    auto heuristic = options.heuristic_options;
    heuristic.random_seed = options.random_seed;
    heuristic.parent_deadline = &deadline;
    heuristic.partition_restriction = options.partition_restriction;
    const auto balance = ComputeJointWorkerBalanceConstraint(instance, options);
    const auto caller_allowed = heuristic.allowed_partition;
    heuristic.allowed_partition = [&, balance, caller_allowed](const auto& p) {
      if (caller_allowed && !caller_allowed(p)) return false;
      if (balance.enabled)
        for (Tick load : WorkerLayerTotals(instance, p))
          if (load < balance.lower_bound || load > balance.upper_bound) return false;
      return true;
    };
    if (options.time_limit_seconds > 0)
      heuristic.time_limit_seconds = std::min(heuristic.time_limit_seconds, incumbent_effective_budget_seconds);
    auto incumbent = BuildHeuristicIncumbent(instance, heuristic);
    RequireValidBfsIncumbentForUse(instance, &incumbent, options.partition_restriction,
                                  "heuristic incumbent failed validation");
    return maybe_replace_cap_violating_incumbent(incumbent, "heuristic incumbent accepted structurally");
  }
  const bool timed_auto_hybrid =
      options.time_limit_seconds > 0.0 && build_method == "auto";
  const bool explicit_hybrid = build_method == "hybrid-slack";
  if (timed_auto_hybrid || explicit_hybrid) {
    HybridSlackIncumbentOptions hybrid_options;
    hybrid_options.time_limit_seconds =
        options.time_limit_seconds > 0.0
            ? std::max(incumbent_effective_budget_seconds,
                       std::numeric_limits<double>::min())
            : 0.0;
    hybrid_options.partition_restriction = options.partition_restriction;
    BfsSplitOptimizationResult incumbent =
        BuildHybridSlackIncumbent(instance, hybrid_options);
    if (timed_auto_hybrid) incumbent.method = "auto-hybrid-slack";
    std::string reason;
    if (!ValidateBfsIncumbentForUse(instance, &incumbent,
                                    options.partition_restriction, &reason)) {
      return restricted_reference_fallback_after_attempt(
          incumbent, "hybrid-slack incumbent rejected: " + reason);
    }
    return maybe_replace_cap_violating_incumbent(
        incumbent, "hybrid-slack incumbent accepted structurally");
  }
  if (build_method == "uniform") {
    BfsSplitOptimizationResult incumbent =
        BuildBfsIncumbentForSplit(instance, UniformSplit(instance), "uniform");
    RequireValidBfsIncumbentForUse(instance, &incumbent,
                                   options.partition_restriction,
                                   "uniform joint incumbent failed validation");
    return maybe_replace_cap_violating_incumbent(
        incumbent, "uniform joint incumbent accepted structurally");
  }
  if (options.time_limit_seconds > 0.0 && build_method == "enumerate") {
    throw Error(
        "timed explicit BFS incumbent enumeration is not deadline-aware; use "
        "--bfs-method auto, --bfs-method hybrid-slack, --bfs-method cpsat, or "
        "omit the time limit");
  }

  BfsSplitOptimizerOptions bfs_options;
  bfs_options.enumeration_threshold =
      options.time_limit_seconds > 0.0 && build_method == "auto"
          ? 0
          : options.enumeration_threshold;
  bfs_options.time_limit_seconds = incumbent_effective_budget_seconds;
  bfs_options.num_workers = options.num_workers;
  bfs_options.random_seed = options.random_seed;
  bfs_options.require_optimal = false;
  bfs_options.log_search_progress = options.log_search_progress;
  if (options.lifecycle) {
    bfs_options.lifecycle = [&](LifecycleEvent event) {
      event.algorithm = "optimize-joint";
      event.configured_solver_limit_seconds = options.time_limit_seconds;
      event.requested_solver_limit_seconds = incumbent_effective_budget_seconds;
      event.remaining_global_time_seconds =
          deadline.remaining_seconds_for_reporting();
      event.phase_specific_cap_seconds = incumbent_phase_cap_seconds;
      options.lifecycle(event);
    };
  }

  BfsSplitOptimizationResult incumbent;
  if (build_method == "auto") {
    incumbent = OptimizeBfsSplitAuto(instance, bfs_options);
  } else if (build_method == "enumerate") {
    incumbent = OptimizeBfsSplitEnumerate(instance, bfs_options);
  } else if (build_method == "cpsat") {
    incumbent = OptimizeBfsSplitCpSat(instance, bfs_options);
  } else {
    throw Error("unknown BFS incumbent method: " + build_method);
  }
  if (!incumbent.schedule.ok()) {
    return restricted_reference_fallback_after_attempt(
        incumbent,
        "BFS incumbent optimizer did not produce a replayable schedule; "
        "using reference fallback");
  }
  std::string validation_reason;
  if (!ValidateBfsIncumbentForUse(instance, &incumbent,
                                  options.partition_restriction,
                                  &validation_reason)) {
    return restricted_reference_fallback_after_attempt(
        incumbent,
        "BFS incumbent optimizer failed validation: " + validation_reason +
            "; "
            "using reference fallback");
  }
  return maybe_replace_cap_violating_incumbent(
      incumbent, "BFS incumbent optimizer accepted structurally");
}

[[nodiscard]] std::string EffectiveBfsMethodName(const std::string& method) {
  if (method == "auto-hybrid-slack") return "hybrid-slack";
  if (method == "auto-cpsat") return "cpsat";
  if (method == "auto-enumerate") return "enumerate";
  if (method.rfind("auto-hybrid-slack", 0) == 0) {
    return "hybrid-slack" +
           method.substr(std::string("auto-hybrid-slack").size());
  }
  if (method.rfind("auto-cpsat", 0) == 0) {
    return "cpsat" + method.substr(std::string("auto-cpsat").size());
  }
  if (method.rfind("auto-enumerate", 0) == 0) {
    return "enumerate" + method.substr(std::string("auto-enumerate").size());
  }
  return method.empty() ? "none" : method;
}

[[nodiscard]] std::string IncumbentSourceName(const std::string& method,
                                              bool override) {
  if (method.empty()) return "none";
  if (method == "heuristic") return override ? "heuristic_incumbent_override" : "heuristic_incumbent";
  if (method == "schedule-only-hint") return "schedule_only_fixed_partition";
  if (method == "deterministic-uniform-bfs-fallback") {
    return "deterministic_uniform_breadth_first_incumbent";
  }
  if (method == "uniform-fallback") {
    return "deterministic_uniform_breadth_first_incumbent";
  }
  if (method == "restricted-reference-fallback" ||
      method.find("reference-fallback") != std::string::npos) {
    return "restricted_reference_breadth_first_incumbent";
  }
  if (method.find("hybrid-slack") != std::string::npos) {
    return override ? "hybrid_slack_incumbent_override"
                    : "hybrid_slack_incumbent";
  }
  if (method == "uniform") return "uniform_incumbent";
  if (override) return "bfs_incumbent_override";
  return "bfs_incumbent";
}

[[nodiscard]] Index BfsCpSatModelsSolved(
    const BfsSplitOptimizationResult& result) {
  if (result.cp_sat_models_solved > 0) return result.cp_sat_models_solved;
  return result.method.find("cpsat") == std::string::npos ? 0 : 1;
}

[[nodiscard]] Index PartitionRestrictionAuxiliaryVariableCount(
    const Instance& instance,
    const std::optional<PartitionRestriction>& restriction) {
  if (!restriction) return 0;
  switch (restriction->mode) {
    case SlackPipeSplitMode::kLocal:
      return instance.stages;
    case SlackPipeSplitMode::kWorkerLocal:
      return instance.workers;
    case SlackPipeSplitMode::kFixed:
    case SlackPipeSplitMode::kGlobal:
    case SlackPipeSplitMode::kWorkerFixed:
      return 0;
  }
  return 0;
}

void ValidateHintAssignments(
    const Instance& instance, const BfsSplitOptimizationResult& incumbent,
    Tick horizon, const std::optional<PartitionRestriction>& restriction) {
  ValidateSplit(instance, incumbent.split);
  if (!incumbent.schedule.ok()) {
    throw Error("cannot hint Joint CP-SAT with an invalid incumbent schedule");
  }
  if (restriction && restriction->mode == SlackPipeSplitMode::kFixed &&
      incumbent.split != restriction->reference_split) {
    throw Error("fixed-partition Joint hint does not match fixed split");
  }
  if (restriction && !SplitSatisfiesPartitionRestriction(
                         instance, incumbent.split, *restriction)) {
    throw Error("Joint hint split violates active partition restriction");
  }
  if (incumbent.schedule.operations_by_id.size() !=
      static_cast<std::size_t>(instance.OperationCount())) {
    throw Error("Joint hint schedule does not contain every operation");
  }

  std::vector<Index> seen(static_cast<std::size_t>(instance.OperationCount()),
                          0);
  for (const ScheduledOperation& op : incumbent.schedule.operations_by_id) {
    if (op.id.value < 0 || op.id.value >= instance.OperationCount()) {
      throw Error("Joint hint operation id out of range");
    }
    ++seen[static_cast<std::size_t>(op.id.value)];
    const OperationView view = DecodeOperation(instance, op.id);
    if (op.worker != view.worker) {
      throw Error("Joint hint operation is assigned to the wrong worker");
    }
    if (op.start < 0 || op.end < 0 || op.start > horizon || op.end > horizon) {
      throw Error("Joint hint operation time is outside the model horizon");
    }
    if (op.end < op.start || op.end - op.start != op.duration) {
      throw Error("Joint hint operation start/end/duration are inconsistent");
    }
    const Tick expected_duration =
        instance.Duration(view.stage, view.backward, incumbent.split);
    if (op.duration != expected_duration) {
      throw Error("Joint hint operation duration does not match hinted split");
    }
  }
  for (Index id = 0; id < instance.OperationCount(); ++id) {
    if (seen[static_cast<std::size_t>(id)] != 1) {
      throw Error("Joint hint schedule has missing or duplicate operations");
    }
  }
}

[[nodiscard]] LinearExpr WorkerLayerExpr(const Instance& instance,
                                         const std::vector<IntVar>& layers,
                                         Index worker) {
  LinearExpr expr;
  for (Index s = 0; s < instance.stages; ++s) {
    if (s % instance.workers == worker) {
      expr += layers[static_cast<std::size_t>(s)];
    }
  }
  return expr;
}

void AddStageAbsDeviation(CpModelBuilder& builder, const IntVar& layer,
                          Tick reference, Tick max_abs, Index stage,
                          std::vector<IntVar>* abs_values) {
  IntVar abs_delta = builder.NewIntVar(Domain(0, max_abs))
                         .WithName("abs_stage_delta_" + std::to_string(stage));
  builder.AddGreaterOrEqual(abs_delta, layer - reference);
  builder.AddGreaterOrEqual(abs_delta, reference - layer);
  abs_values->push_back(abs_delta);
}

void AddWorkerAbsDeviation(CpModelBuilder& builder,
                           const LinearExpr& worker_layers, Tick reference,
                           Tick max_abs, Index worker,
                           std::vector<IntVar>* abs_values) {
  IntVar abs_delta =
      builder.NewIntVar(Domain(0, max_abs))
          .WithName("abs_worker_delta_" + std::to_string(worker));
  builder.AddGreaterOrEqual(abs_delta, worker_layers - reference);
  builder.AddGreaterOrEqual(abs_delta, reference - worker_layers);
  abs_values->push_back(abs_delta);
}

void AddPartitionRestrictionConstraints(
    CpModelBuilder& builder, const Instance& instance,
    const std::vector<IntVar>& layers,
    const PartitionRestriction& restriction) {
  ValidatePartitionRestriction(instance, restriction);
  switch (restriction.mode) {
    case SlackPipeSplitMode::kGlobal:
      return;
    case SlackPipeSplitMode::kFixed:
      for (Index s = 0; s < instance.stages; ++s) {
        builder.AddEquality(
            layers[static_cast<std::size_t>(s)],
            restriction.reference_split[static_cast<std::size_t>(s)]);
      }
      return;
    case SlackPipeSplitMode::kLocal: {
      std::vector<IntVar> abs_deltas;
      abs_deltas.reserve(static_cast<std::size_t>(instance.stages));
      for (Index s = 0; s < instance.stages; ++s) {
        const Tick reference =
            restriction.reference_split[static_cast<std::size_t>(s)];
        if (restriction.per_stage_delta) {
          const Tick lower =
              CheckedAdd(reference, -*restriction.per_stage_delta,
                         "CP-SAT local per-stage lower bound");
          const Tick upper = CheckedAdd(reference, *restriction.per_stage_delta,
                                        "CP-SAT local per-stage upper bound");
          builder.AddGreaterOrEqual(layers[static_cast<std::size_t>(s)], lower);
          builder.AddLessOrEqual(layers[static_cast<std::size_t>(s)], upper);
        }
        AddStageAbsDeviation(builder, layers[static_cast<std::size_t>(s)],
                             reference, instance.total_layers, s, &abs_deltas);
      }
      builder.AddLessOrEqual(LinearExpr::Sum(abs_deltas),
                             CheckedMul(2, *restriction.move_budget,
                                        "CP-SAT local partition move budget"));
      return;
    }
    case SlackPipeSplitMode::kWorkerFixed:
      for (Index w = 0; w < instance.workers; ++w) {
        builder.AddEquality(
            WorkerLayerExpr(instance, layers, w),
            WorkerLayerTotals(
                instance,
                restriction.reference_split)[static_cast<std::size_t>(w)]);
      }
      return;
    case SlackPipeSplitMode::kWorkerLocal: {
      const std::vector<Tick> reference_workers =
          WorkerLayerTotals(instance, restriction.reference_split);
      std::vector<IntVar> abs_deltas;
      abs_deltas.reserve(static_cast<std::size_t>(instance.workers));
      for (Index w = 0; w < instance.workers; ++w) {
        const LinearExpr worker_layers = WorkerLayerExpr(instance, layers, w);
        const Tick reference = reference_workers[static_cast<std::size_t>(w)];
        if (restriction.per_worker_delta) {
          const Tick lower =
              CheckedAdd(reference, -*restriction.per_worker_delta,
                         "CP-SAT worker-local per-worker lower bound");
          const Tick upper =
              CheckedAdd(reference, *restriction.per_worker_delta,
                         "CP-SAT worker-local per-worker upper bound");
          builder.AddGreaterOrEqual(worker_layers, lower);
          builder.AddLessOrEqual(worker_layers, upper);
        }
        AddWorkerAbsDeviation(builder, worker_layers, reference,
                              instance.total_layers, w, &abs_deltas);
      }
      builder.AddLessOrEqual(
          LinearExpr::Sum(abs_deltas),
          CheckedMul(2, *restriction.worker_move_budget,
                     "CP-SAT worker-local partition move budget"));
      return;
    }
  }
}

void RestrictPressurePartitionsToMode(
    const Instance& instance, const PartitionRestriction* restriction,
    const PressurePruningOptions& pressure_options,
    std::vector<PressurePartitionCandidate>* pressure_partitions,
    PressurePruningStats* stats) {
  if (restriction == nullptr ||
      restriction->mode == SlackPipeSplitMode::kGlobal) {
    return;
  }
  std::vector<PressurePartitionCandidate> filtered;
  filtered.reserve(pressure_partitions->size());
  bool found_reference = false;
  for (const PressurePartitionCandidate& candidate : *pressure_partitions) {
    if (SplitSatisfiesPartitionRestriction(instance, candidate.split,
                                           *restriction)) {
      if (candidate.split == restriction->reference_split) {
        found_reference = true;
      }
      filtered.push_back(candidate);
    }
  }
  if (!found_reference) {
    if (!SplitSatisfiesPartitionRestriction(
            instance, restriction->reference_split, *restriction)) {
      throw Error("partition restriction reference split is not feasible");
    }
    filtered.push_back(PressurePartitionCandidate{
        restriction->reference_split,
        ComputePartitionPressure(instance, restriction->reference_split,
                                 pressure_options)});
    if (stats != nullptr) {
      stats->incumbent_split_included = true;
      ++stats->anchor_splits_added;
    }
  }
  if (filtered.empty()) {
    throw Error(
        "pressure partition pruning produced no partitions after applying "
        "the active split-mode restriction");
  }
  *pressure_partitions = std::move(filtered);
  if (stats != nullptr) {
    const std::vector<Tick> uniform = BuildUniformPressureSplit(instance);
    const std::vector<Tick> cost_balanced =
        BuildCostBalancedPressureSplit(instance);
    stats->incumbent_split_included = false;
    stats->uniform_split_included = false;
    stats->cost_balanced_split_included = false;
    for (const PressurePartitionCandidate& candidate : *pressure_partitions) {
      if (candidate.split == restriction->reference_split) {
        stats->incumbent_split_included = true;
      }
      if (candidate.split == uniform) {
        stats->uniform_split_included = true;
      }
      if (candidate.split == cost_balanced) {
        stats->cost_balanced_split_included = true;
      }
    }
    stats->anchor_splits_added =
        static_cast<Index>(stats->incumbent_split_included) +
        static_cast<Index>(stats->uniform_split_included) +
        static_cast<Index>(stats->cost_balanced_split_included);
    stats->partitions_after =
        static_cast<std::int64_t>(pressure_partitions->size());
    stats->selected_splits_total = stats->partitions_after;
  }
}

[[nodiscard]] MachineOrders ExtractOrdersFromRawTimes(
    const Instance& instance, const std::vector<RawOperationTime>& raw_times) {
  MachineOrders orders(static_cast<std::size_t>(instance.workers));
  for (const RawOperationTime& raw : raw_times) {
    const OperationView view = DecodeOperation(instance, raw.id);
    orders[static_cast<std::size_t>(view.worker)].push_back(raw.id);
  }
  for (auto& order : orders) {
    std::sort(order.begin(), order.end(), [&](OperationId a, OperationId b) {
      const RawOperationTime& raw_a =
          raw_times[static_cast<std::size_t>(a.value)];
      const RawOperationTime& raw_b =
          raw_times[static_cast<std::size_t>(b.value)];
      const OperationView view_a = DecodeOperation(instance, a);
      const OperationView view_b = DecodeOperation(instance, b);
      return std::tuple<Tick, Tick, Index, Index>{raw_a.start, raw_a.end,
                                                  view_a.microbatch,
                                                  view_a.chain_index} <
             std::tuple<Tick, Tick, Index, Index>{
                 raw_b.start, raw_b.end, view_b.microbatch, view_b.chain_index};
    });
  }
  return orders;
}

void ValidateRawNoOverlap(const Instance& instance, const MachineOrders& orders,
                          const std::vector<RawOperationTime>& raw_times) {
  for (Index w = 0; w < instance.workers; ++w) {
    const auto& order = orders[static_cast<std::size_t>(w)];
    for (std::size_t i = 1; i < order.size(); ++i) {
      const RawOperationTime& prev =
          raw_times[static_cast<std::size_t>(order[i - 1].value)];
      const RawOperationTime& current =
          raw_times[static_cast<std::size_t>(order[i].value)];
      if (current.start < prev.end) {
        const OperationView view = DecodeOperation(instance, order[i]);
        std::ostringstream msg;
        msg << "WORKER_OVERLAP worker=" << w << " b=" << view.microbatch
            << " n=" << view.chain_index << " previous_end=" << prev.end
            << " current_start=" << current.start;
        throw Error(msg.str());
      }
    }
  }
}

[[nodiscard]] bool ValidateScheduleForJointReturn(
    const Instance& instance, const ScheduleSolution& schedule,
    const std::string& status, const std::string& method, bool fifo_ordering,
    std::string* rejection_reason) {
  const ResultValidationResult validation = ValidateScheduleSolutionIndependent(
      instance, schedule, status, method, fifo_ordering);
  if (validation.passed) return true;
  if (rejection_reason != nullptr) {
    *rejection_reason = ValidationFailureMessage(validation);
  }
  return false;
}

[[nodiscard]] bool AddReplayCombinationNogood(
    CpModelBuilder& builder, const Instance& instance,
    const std::vector<IntVar>& layers, const std::vector<IntVar>& starts,
    const std::vector<IntVar>& ends, const std::vector<Tick>& split,
    const MachineOrders& orders, Index rejection_index) {
  std::vector<BoolVar> alternatives;
  for (Index s = 0; s < instance.stages; ++s) {
    const Tick rejected_layers = split[static_cast<std::size_t>(s)];
    if (rejected_layers > instance.min_layers) {
      BoolVar less = builder.NewBoolVar().WithName(
          "replay_cap_reject_split_less_" + std::to_string(rejection_index) +
          "_" + std::to_string(s));
      builder
          .AddLessOrEqual(layers[static_cast<std::size_t>(s)],
                          rejected_layers - 1)
          .OnlyEnforceIf(less);
      alternatives.push_back(less);
    }
    if (rejected_layers < instance.total_layers) {
      BoolVar greater = builder.NewBoolVar().WithName(
          "replay_cap_reject_split_greater_" + std::to_string(rejection_index) +
          "_" + std::to_string(s));
      builder
          .AddGreaterOrEqual(layers[static_cast<std::size_t>(s)],
                             rejected_layers + 1)
          .OnlyEnforceIf(greater);
      alternatives.push_back(greater);
    }
  }
  for (const std::vector<OperationId>& worker_order : orders) {
    for (std::size_t i = 1; i < worker_order.size(); ++i) {
      const OperationId prev = worker_order[i - 1];
      const OperationId next = worker_order[i];
      BoolVar inverted = builder.NewBoolVar().WithName(
          "replay_cap_reject_order_" + std::to_string(rejection_index) + "_" +
          std::to_string(prev.value) + "_" + std::to_string(next.value));
      builder
          .AddGreaterOrEqual(starts[static_cast<std::size_t>(prev.value)],
                             ends[static_cast<std::size_t>(next.value)])
          .OnlyEnforceIf(inverted);
      alternatives.push_back(inverted);
    }
  }
  if (alternatives.empty()) return false;
  builder.AddBoolOr(alternatives);
  return true;
}

}  // namespace

JointOptimizationResult OptimizeJointSplitAndScheduleCpSat(
    const Instance& instance, const JointOptimizerOptions& options) {
  const auto started = Clock::now();
  const Deadline deadline(options.time_limit_seconds);
  instance.Validate();
  if (options.partition_restriction) {
    ValidatePartitionRestriction(instance, *options.partition_restriction);
  }
  ValidateJointOptimizerMechanismOptions(options);
  if (kIncumbentBudgetFraction < 0.0 || kIncumbentBudgetFraction > 1.0) {
    throw Error("invalid fixed Joint incumbent budget fraction");
  }
  const std::string incumbent_method_requested =
      RequestedIncumbentMethodName(options);
  const bool external_incumbent_enabled = incumbent_method_requested != "none";
  const bool incumbent_hints_requested = IncumbentHintsRequested(options);
  const bool incumbent_fallback_enabled = IncumbentFallbackRequested(options);
  const bool worker_balance_pruning_requested =
      WorkerBalancePruningRequested(options);
  const bool activation_partition_optimized =
      !(options.partition_restriction &&
        options.partition_restriction->mode == SlackPipeSplitMode::kFixed);
  const std::vector<Tick>* activation_fixed_split =
      activation_partition_optimized
          ? nullptr
          : &options.partition_restriction->reference_split;
  const WorkerBalanceConstraintResult worker_balance_constraint =
      ComputeJointWorkerBalanceConstraint(instance, options);
  Index workload_constraint_count = 0;
  auto populate_mechanism_provenance = [&](JointOptimizationResult* result) {
    result->raw_bound_globally_valid = !worker_balance_constraint.enabled &&
        (!options.partition_restriction ||
         options.partition_restriction->mode == SlackPipeSplitMode::kGlobal);
    result->workload_constraint_count = workload_constraint_count;
    result->worker_balance_pruning_requested = worker_balance_pruning_requested;
    result->worker_balance_pruning_effective =
        worker_balance_constraint.enabled;
    if (options.worker_balance_tolerance_percent >= 0.0) {
      result->worker_balance_tolerance_requested_percent =
          options.worker_balance_tolerance_percent;
    }
    result->worker_balance_tolerance_requested_layers =
        options.worker_balance_tolerance_layers;
    result->incumbent_method_requested_normalized = incumbent_method_requested;
    result->incumbent_bound_requested = options.incumbent_bound;
    result->incumbent_hints_requested = incumbent_hints_requested;
    result->fallback_enabled = incumbent_fallback_enabled;
    result->global_time_limit_seconds = options.time_limit_seconds;
    result->fifo_ordering_requested = options.fifo_ordering;
    result->fifo_ordering_effective = options.fifo_ordering;
    result->fifo_constraint_count = JointFifoConstraintCount(instance, options);
    result->symmetry_break_f0_fifo =
        options.fifo_ordering && options.symmetry_break_f0_fifo;
    result->final_solution_source = result->solution_source.empty()
                                        ? std::string("none")
                                        : result->solution_source;
  };
  ActivationCapConstraintMetadata activation_metadata;
  activation_metadata.model_support_level =
      ToString(ActivationCapSolverSupport());
  activation_metadata.solver_supported = ActivationCapSolverCanEnforce(
      options.activation_options, activation_partition_optimized);
  if (options.activation_options.enforce_activation_cap &&
      !activation_metadata.solver_supported) {
    activation_metadata.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options,
                                             activation_partition_optimized);
    JointOptimizationResult result;
    result.status = "UNAVAILABLE";
    result.joint_status = "UNAVAILABLE";
    result.proven_optimal = false;
    result.wall_time_seconds = Since(started);
    result.timing.total_seconds = result.wall_time_seconds;
    result.incumbent_status = "NOT_RUN";
    result.incumbent_method_requested = incumbent_method_requested;
    result.incumbent_method_effective = "none";
    result.bfs_incumbent_method_requested = options.bfs_method;
    result.bfs_incumbent_method_effective = "none";
    result.incumbent_source = "none";
    result.incumbent_feasible = false;
    result.horizon_source = "none";
    result.hints_requested = incumbent_hints_requested;
    result.hints_effective = false;
    result.incumbent_hints_effective = false;
    result.hint_source = "none";
    result.hint_scope = "none";
    result.hint_termination_reason = "unavailable";
    result.fallback_available = false;
    result.fallback_source = "none";
    result.solution_source = "none";
    result.fallback_used = false;
    result.symmetry_break_f0_fifo =
        options.fifo_ordering && options.symmetry_break_f0_fifo;
    result.num_workers = options.num_workers;
    result.worker_balance_constraint = worker_balance_constraint;
    result.activation_cap_constraints = activation_metadata;
    result.search_stats_enabled = options.search_stats != nullptr;
    result.diagnostic = activation_metadata.unsupported_reason;
    populate_mechanism_provenance(&result);
    if (options.search_stats != nullptr)
      result.search_stats = *options.search_stats;
    return result;
  }
  SearchStats* stats = options.search_stats;
  if (stats != nullptr) {
    constexpr std::uint64_t kMaxCount =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const std::uint64_t count = CountValidSplitsCapped(instance, kMaxCount);
    if (count < kMaxCount) {
      stats->stage_partitions_theoretical_available = true;
      stats->stage_partitions_theoretical = static_cast<std::int64_t>(count);
    }
  }
  const double incumbent_phase_cap_seconds =
      options.bfs_incumbent_override || !external_incumbent_enabled
          ? 0.0
          : IncumbentPhaseCap(options.time_limit_seconds);
  const double incumbent_remaining_global_seconds =
      deadline.remaining_seconds_for_reporting();
  const double incumbent_effective_budget_seconds =
      options.bfs_incumbent_override || !external_incumbent_enabled
          ? 0.0
          : deadline.clamp_solver_limit(incumbent_phase_cap_seconds);
  const auto incumbent_started = Clock::now();
  EmitLifecycle(options, started, "MODEL_BUILD_START", "", 0.0,
                "bfs_incumbent_prepare", incumbent_effective_budget_seconds,
                incumbent_remaining_global_seconds,
                incumbent_phase_cap_seconds);
  BfsSplitOptimizationResult bfs_incumbent = BuildBfsIncumbent(
      instance, options, deadline, incumbent_effective_budget_seconds,
      incumbent_phase_cap_seconds);
  const double incumbent_seconds =
      std::chrono::duration<double>(Clock::now() - incumbent_started).count();
  EmitLifecycle(options, started, "MODEL_BUILD_END", bfs_incumbent.status, 0.0,
                "bfs_incumbent_prepare", incumbent_effective_budget_seconds,
                deadline.remaining_seconds_for_reporting(),
                incumbent_phase_cap_seconds);
  const bool incumbent_structurally_valid =
      BfsIncumbentCompleteForInstance(instance, bfs_incumbent);
  const bool incumbent_cap_feasible =
      incumbent_structurally_valid &&
      (!options.activation_options.enforce_activation_cap ||
       ActivationScheduleSatisfiesCap(instance, bfs_incumbent.schedule,
                                      options.activation_options));
  const bool external_incumbent_available =
      external_incumbent_enabled && incumbent_cap_feasible;
  activation_metadata.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      incumbent_structurally_valid && !incumbent_cap_feasible;
  const bool incumbent_bound_effective =
      options.incumbent_bound && external_incumbent_available;
  const Tick horizon =
      incumbent_bound_effective && bfs_incumbent.makespan_ticks > 0
          ? bfs_incumbent.makespan_ticks
          : ConservativeJointHorizon(instance);
  const std::string incumbent_source = IncumbentSourceName(
      bfs_incumbent.method, options.bfs_incumbent_override.has_value());
  const HybridSlackScores incumbent_hybrid_scores =
      incumbent_structurally_valid
          ? ComputeHybridSlackScores(instance, bfs_incumbent.split)
          : HybridSlackScores{};
  const double incumbent_hybrid_min_slack =
      bfs_incumbent.hybrid_min_slack > 0.0
          ? bfs_incumbent.hybrid_min_slack
          : incumbent_hybrid_scores.partition_min_slack;
  const std::vector<double> incumbent_hybrid_stage_scores =
      bfs_incumbent.hybrid_stage_scores.empty()
          ? incumbent_hybrid_scores.stage_scores
          : bfs_incumbent.hybrid_stage_scores;
  const std::vector<Index> incumbent_hybrid_bottleneck_stages =
      bfs_incumbent.hybrid_bottleneck_stages.empty()
          ? incumbent_hybrid_scores.bottleneck_stages
          : bfs_incumbent.hybrid_bottleneck_stages;
  const Tick incumbent_baseline_primary_objective =
      bfs_incumbent.baseline_primary_objective > 0
          ? bfs_incumbent.baseline_primary_objective
          : bfs_incumbent.makespan_ticks;
  const double incumbent_baseline_hybrid_min_slack =
      bfs_incumbent.baseline_hybrid_min_slack > 0.0
          ? bfs_incumbent.baseline_hybrid_min_slack
          : incumbent_hybrid_min_slack;
  const Index incumbent_cp_sat_models_solved =
      options.bfs_incumbent_override ? 0 : BfsCpSatModelsSolved(bfs_incumbent);
  PressurePruningStats pressure_pruning_stats;
  std::vector<PressurePartitionCandidate> pressure_partitions;

  auto make_deadline_fallback_result =
      [&](const std::string& reason, double model_build_seconds,
          const JointOptimizationResult* last_attempt =
              nullptr) -> JointOptimizationResult {
    JointOptimizationResult result;
    const bool has_incumbent =
        external_incumbent_available && incumbent_fallback_enabled;
    result.status = has_incumbent ? "FEASIBLE" : "NO_VALID_SOLUTION";
    result.joint_status = "NOT_RUN";
    result.proven_optimal = false;
    result.best_bound_ticks = 0.0;
    result.solver_objective_ticks =
        has_incumbent ? static_cast<double>(bfs_incumbent.makespan_ticks) : 0.0;
    result.wall_time_seconds = deadline.elapsed_seconds();
    result.timing.incumbent_seconds = incumbent_seconds;
    result.timing.model_build_seconds = model_build_seconds;
    result.timing.solver_seconds = 0.0;
    result.timing.total_seconds = result.wall_time_seconds;
    if (last_attempt != nullptr) {
      result.joint_status = last_attempt->joint_status;
      result.best_bound_ticks = last_attempt->best_bound_ticks;
      result.solver_objective_ticks = last_attempt->solver_objective_ticks;
      result.timing.solver_seconds = last_attempt->timing.solver_seconds;
      result.timing.ortools_wall_time_seconds =
          last_attempt->timing.ortools_wall_time_seconds;
      result.timing.extraction_seconds =
          last_attempt->timing.extraction_seconds;
      result.timing.canonicalization_seconds =
          last_attempt->timing.canonicalization_seconds;
      result.joint_budget_seconds = last_attempt->joint_budget_seconds;
      result.joint_solve_seconds = last_attempt->joint_solve_seconds;
      result.deterministic_time = last_attempt->deterministic_time;
      result.branches = last_attempt->branches;
      result.conflicts = last_attempt->conflicts;
      result.time_to_first_cpsat_feasible_seconds =
          last_attempt->time_to_first_cpsat_feasible_seconds;
      result.first_cpsat_feasible_objective =
          last_attempt->first_cpsat_feasible_objective;
      result.solver_solution_available =
          last_attempt->solver_solution_available;
      result.incumbent_improvement_count =
          last_attempt->incumbent_improvement_count;
      result.incumbent_trace = last_attempt->incumbent_trace;
      result.cp_sat_models_solved = last_attempt->cp_sat_models_solved;
    }
    result.incumbent_budget_seconds = incumbent_effective_budget_seconds;
    result.incumbent_model_build_seconds =
        bfs_incumbent.timing.model_build_seconds;
    result.incumbent_solve_seconds = bfs_incumbent.timing.solver_seconds;
    if (result.incumbent_solve_seconds == 0.0 &&
        bfs_incumbent.wall_time_seconds > 0.0 &&
        bfs_incumbent.timing.model_build_seconds == 0.0) {
      result.incumbent_solve_seconds = bfs_incumbent.wall_time_seconds;
    }
    result.incumbent_status = bfs_incumbent.status;
    if (last_attempt == nullptr) {
      result.joint_budget_seconds = 0.0;
    }
    result.joint_model_build_seconds = model_build_seconds;
    if (last_attempt == nullptr) {
      result.joint_solve_seconds = 0.0;
    }
    result.incumbent_method_requested = incumbent_method_requested;
    result.incumbent_method_effective = EffectiveIncumbentMechanismName(
        incumbent_method_requested, bfs_incumbent);
    result.bfs_incumbent_method_requested = options.bfs_method;
    result.bfs_incumbent_method_effective =
        EffectiveBfsMethodName(bfs_incumbent.method);
    result.incumbent_source = incumbent_source;
    result.incumbent_feasible = external_incumbent_available;
    result.incumbent_primary_objective = bfs_incumbent.makespan_ticks;
    result.incumbent_found = incumbent_structurally_valid;
    result.incumbent_valid = external_incumbent_available;
    result.incumbent_makespan = bfs_incumbent.makespan_ticks;
    result.incumbent_hybrid_min_slack = incumbent_hybrid_min_slack;
    result.incumbent_baseline_primary_objective =
        incumbent_baseline_primary_objective;
    result.incumbent_baseline_hybrid_min_slack =
        incumbent_baseline_hybrid_min_slack;
    result.incumbent_improved_over_baseline =
        bfs_incumbent.improved_over_baseline;
    result.incumbent_hybrid_stage_scores = incumbent_hybrid_stage_scores;
    result.incumbent_hybrid_bottleneck_stages =
        incumbent_hybrid_bottleneck_stages;
    result.horizon_source =
        incumbent_bound_effective ? incumbent_source : "conservative";
    result.incumbent_bound_effective = incumbent_bound_effective;
    result.incumbent_bound_horizon =
        incumbent_bound_effective ? bfs_incumbent.makespan_ticks : 0;
    result.hint_budget_seconds = bfs_incumbent.hint_budget_seconds > 0.0
                                     ? bfs_incumbent.hint_budget_seconds
                                     : incumbent_effective_budget_seconds;
    result.hint_elapsed_seconds = bfs_incumbent.hint_elapsed_seconds > 0.0
                                      ? bfs_incumbent.hint_elapsed_seconds
                                      : incumbent_seconds;
    result.hint_iterations = bfs_incumbent.hint_iterations;
    result.hint_candidates_generated = bfs_incumbent.hint_candidates_generated;
    result.hint_candidates_simulated = bfs_incumbent.hint_candidates_simulated;
    result.hint_partition_moves_accepted =
        bfs_incumbent.hint_partition_moves_accepted;
    result.hint_interleaving_moves_accepted =
        bfs_incumbent.hint_interleaving_moves_accepted;
    result.hint_deadline_reached = bfs_incumbent.hint_deadline_reached;
    result.hint_termination_reason =
        bfs_incumbent.hint_termination_reason.empty()
            ? "not_applicable"
            : bfs_incumbent.hint_termination_reason;
    result.hints_requested = incumbent_hints_requested;
    result.hints_effective = false;
    result.incumbent_hints_effective = false;
    result.hint_source = "none";
    result.hint_scope = "none";
    result.fallback_available =
        incumbent_fallback_enabled && external_incumbent_available;
    result.fallback_source =
        result.fallback_available ? incumbent_source : "none";
    result.solution_source = has_incumbent ? "bfs_incumbent_fallback" : "none";
    result.fallback_used = has_incumbent;
    result.final_solution_available = has_incumbent;
    result.final_solution_source =
        has_incumbent ? "external_incumbent" : "none";
    result.no_solution_reason =
        has_incumbent ? std::string()
                      : NoSolutionReasonForDeadlineFallback(reason);
    if (last_attempt == nullptr) {
      result.cp_sat_models_solved = incumbent_cp_sat_models_solved;
    }
    result.symmetry_break_f0_fifo =
        options.fifo_ordering && options.symmetry_break_f0_fifo;
    result.num_workers = options.num_workers;
    result.worker_balance_constraint = worker_balance_constraint;
    result.pressure_pruning_stats = pressure_pruning_stats;
    result.activation_cap_constraints = activation_metadata;
    result.search_stats_enabled = stats != nullptr;
    result.diagnostic = reason;
    if (!has_incumbent &&
        activation_metadata.incumbent_rejected_for_activation_cap) {
      result.diagnostic += "; no cap-feasible incumbent is available";
    }
    if (stats != nullptr) {
      if (stats->algorithm.empty()) stats->algorithm = "optimize-joint";
      stats->enumerative_search = false;
      stats->stage_partitions_note =
          "Stage partitions are encoded as CP-SAT variables, not explicitly "
          "enumerated.";
      stats->interleave_orders_note =
          "Worker-local orders are encoded in the CP-SAT model, not explicitly "
          "enumerated.";
      stats->cp_sat_available = true;
      stats->cp_sat_status =
          last_attempt != nullptr ? last_attempt->joint_status : "NOT_RUN";
    }
    if (has_incumbent) {
      result.split = bfs_incumbent.split;
      result.machine_orders = bfs_incumbent.machine_orders;
      result.machine_predecessors = ExtractMachinePredecessors(
          instance, result.machine_orders, options.fifo_ordering);
      result.makespan_ticks = bfs_incumbent.makespan_ticks;
      result.schedule = bfs_incumbent.schedule;
      result.time_to_first_feasible_seconds = incumbent_seconds;
      result.time_to_best_incumbent_seconds = incumbent_seconds;
      result.first_feasible_objective = bfs_incumbent.makespan_ticks;
      EvaluationResult checked = EvaluateSchedule(
          instance, result.split, result.machine_orders, options.fifo_ordering);
      if (!checked.schedule.ok() ||
          checked.schedule.makespan != result.makespan_ticks) {
        throw Error(BfsFallbackReplayMismatchMessage(
            result.split, result.machine_orders, result.makespan_ticks,
            checked.schedule));
      }
      result.schedule = std::move(checked.schedule);
    }
    populate_mechanism_provenance(&result);
    result.incumbent_bound_effective = incumbent_bound_effective;
    result.incumbent_bound_horizon =
        incumbent_bound_effective ? bfs_incumbent.makespan_ticks : 0;
    result.incumbent_hints_effective = false;
    result.final_solution_available = has_incumbent;
    result.final_solution_source =
        has_incumbent ? "external_incumbent" : "none";
    if (!has_incumbent && result.no_solution_reason.empty()) {
      result.no_solution_reason = NoSolutionReasonForDeadlineFallback(reason);
    }
    if (options.require_optimal && has_incumbent && !result.proven_optimal) {
      result.status += "_REJECTED_REQUIRE_OPTIMAL";
    }
    result.timing.total_seconds = deadline.elapsed_seconds();
    result.wall_time_seconds = result.timing.total_seconds;
    result.bfs_incumbent = std::move(bfs_incumbent);
    if (stats != nullptr) result.search_stats = *stats;
    return result;
  };

  if (deadline.expired()) {
    return make_deadline_fallback_result(
        "global_deadline_expired_before_cp_sat", 0.0);
  }

  const auto model_build_started = Clock::now();
  EmitLifecycle(options, started, "MODEL_BUILD_START", "", 0.0, "joint_cp_sat");
  CpModelBuilder builder;
  std::vector<IntVar> layers;
  layers.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    layers.push_back(
        builder.NewIntVar(Domain(instance.min_layers, instance.total_layers))
            .WithName("layers_" + std::to_string(s)));
  }
  builder.AddEquality(LinearExpr::Sum(layers), instance.total_layers);
  std::vector<IntVar> cuts;
  cuts.reserve(static_cast<std::size_t>(instance.stages + 1));
  for (Index s = 0; s <= instance.stages; ++s) {
    const Domain domain =
        s == 0 ? Domain(0, 0)
               : (s == instance.stages
                      ? Domain(instance.total_layers, instance.total_layers)
                      : Domain(0, instance.total_layers));
    cuts.push_back(
        builder.NewIntVar(domain).WithName("cut_" + std::to_string(s)));
  }
  for (Index s = 0; s < instance.stages; ++s) {
    builder.AddEquality(cuts[static_cast<std::size_t>(s + 1)],
                        cuts[static_cast<std::size_t>(s)] +
                            layers[static_cast<std::size_t>(s)]);
  }
  Index auxiliary_variable_count = PartitionRestrictionAuxiliaryVariableCount(
      instance, options.partition_restriction);
  if (options.partition_restriction) {
    AddPartitionRestrictionConstraints(builder, instance, layers,
                                       *options.partition_restriction);
  }

  if (options.pressure_pruning.enabled) {
    PressurePruningAnchors pressure_anchors;
    pressure_anchors.has_incumbent_split = external_incumbent_available;
    pressure_anchors.incumbent_split = bfs_incumbent.split;
    pressure_anchors.has_uniform_split = true;
    pressure_anchors.uniform_split = BuildUniformPressureSplit(instance);
    pressure_anchors.has_cost_balanced_split = true;
    pressure_anchors.cost_balanced_split =
        BuildCostBalancedPressureSplit(instance);
    pressure_partitions =
        SelectPressurePartitions(instance, options.pressure_pruning,
                                 &pressure_pruning_stats, pressure_anchors);
    RestrictPressurePartitionsToMode(
        instance,
        options.partition_restriction ? &*options.partition_restriction
                                      : nullptr,
        options.pressure_pruning, &pressure_partitions,
        &pressure_pruning_stats);
    if (pressure_partitions.empty()) {
      throw Error(
          "pressure partition pruning produced no candidate partitions");
    }
    std::vector<IntVar> selectors;
    selectors.reserve(pressure_partitions.size());
    for (std::size_t i = 0; i < pressure_partitions.size(); ++i) {
      selectors.push_back(builder.NewIntVar(Domain(0, 1))
                              .WithName("pressure_split_" + std::to_string(i)));
    }
    auxiliary_variable_count += static_cast<Index>(selectors.size());
    builder.AddEquality(LinearExpr::Sum(selectors), 1);
    for (Index s = 0; s < instance.stages; ++s) {
      LinearExpr selected_layers;
      for (std::size_t i = 0; i < pressure_partitions.size(); ++i) {
        selected_layers +=
            pressure_partitions[i].split[static_cast<std::size_t>(s)] *
            selectors[i];
      }
      builder.AddEquality(layers[static_cast<std::size_t>(s)], selected_layers);
    }
  }

  if (worker_balance_constraint.enabled) {
    for (Index w = 0; w < instance.workers; ++w) {
      LinearExpr worker_layers;
      for (Index s = 0; s < instance.stages; ++s) {
        if (s % instance.workers == w) {
          worker_layers += layers[static_cast<std::size_t>(s)];
        }
      }
      builder.AddGreaterOrEqual(worker_layers,
                                worker_balance_constraint.lower_bound);
      builder.AddLessOrEqual(worker_layers,
                             worker_balance_constraint.upper_bound);
    }
  }

  const Index op_count = instance.OperationCount();
  std::vector<IntVar> starts;
  std::vector<IntVar> ends;
  std::vector<IntVar> durations;
  std::vector<IntervalVar> intervals;
  starts.reserve(static_cast<std::size_t>(op_count));
  ends.reserve(static_cast<std::size_t>(op_count));
  durations.reserve(static_cast<std::size_t>(op_count));
  intervals.reserve(static_cast<std::size_t>(op_count));

  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    starts.push_back(builder.NewIntVar(Domain(0, horizon))
                         .WithName("start_" + std::to_string(id)));
    ends.push_back(builder.NewIntVar(Domain(0, horizon))
                       .WithName("end_" + std::to_string(id)));
    if (instance.HasRangeCostProfile()) {
      const std::vector<Tick>& prefix =
          view.backward ? instance.profile_prefix_backward_ticks
                        : instance.profile_prefix_forward_ticks;
      const std::vector<Tick>& role_biases =
          view.backward ? instance.profile_role_backward_bias_ticks
                        : instance.profile_role_forward_bias_ticks;
      const Tick bias = role_biases[static_cast<std::size_t>(
          view.stage == 0 ? 0 : (view.stage == instance.stages - 1 ? 2 : 1))];
      const Tick max_duration =
          CheckedAdd(prefix.back(), bias, "joint range maximum duration");
      durations.push_back(builder.NewIntVar(Domain(0, max_duration))
                              .WithName("duration_" + std::to_string(id)));
      IntVar prefix_begin = builder.NewIntVar(Domain(0, prefix.back()))
                                .WithName("prefix_begin_" + std::to_string(id));
      IntVar prefix_end = builder.NewIntVar(Domain(0, prefix.back()))
                              .WithName("prefix_end_" + std::to_string(id));
      builder.AddElement(cuts[static_cast<std::size_t>(view.stage)], prefix,
                         prefix_begin);
      builder.AddElement(cuts[static_cast<std::size_t>(view.stage + 1)], prefix,
                         prefix_end);
      builder.AddEquality(durations[static_cast<std::size_t>(id)],
                          prefix_end - prefix_begin + bias);
      auxiliary_variable_count += 2;
    } else {
      const Tick coefficient =
          instance.HasAffineCostProfile()
              ? (view.backward ? instance.profile_backward_slope_ticks
                                     [static_cast<std::size_t>(view.stage)]
                               : instance.profile_forward_slope_ticks
                                     [static_cast<std::size_t>(view.stage)])
              : (view.backward ? instance.backward_ratio_num
                               : instance.backward_ratio_den);
      const Tick bias =
          instance.HasAffineCostProfile()
              ? (view.backward
                     ? instance.profile_backward_bias_ticks
                           [static_cast<std::size_t>(view.stage)]
                     : instance
                           .profile_forward_bias_ticks[static_cast<std::size_t>(
                               view.stage)])
              : 0;
      durations.push_back(
          builder
              .NewIntVar(Domain(
                  CheckedAdd(CheckedMul(instance.min_layers, coefficient,
                                        "joint minimum duration"),
                             bias, "joint minimum duration"),
                  CheckedAdd(CheckedMul(instance.total_layers, coefficient,
                                        "joint maximum duration"),
                             bias, "joint maximum duration")))
              .WithName("duration_" + std::to_string(id)));
      builder.AddEquality(
          durations[static_cast<std::size_t>(id)],
          coefficient * layers[static_cast<std::size_t>(view.stage)] + bias);
    }
    builder.AddEquality(ends[static_cast<std::size_t>(id)],
                        starts[static_cast<std::size_t>(id)] +
                            durations[static_cast<std::size_t>(id)]);
    intervals.push_back(
        builder.NewIntervalVar(starts[static_cast<std::size_t>(id)],
                               durations[static_cast<std::size_t>(id)],
                               ends[static_cast<std::size_t>(id)]));
  }

  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    if (view.chain_index + 1 < 2 * instance.stages) {
      const OperationId next =
          EncodeOperation(instance, view.microbatch, view.chain_index + 1);
      const OperationView next_view = DecodeOperation(instance, next);
      builder.AddGreaterOrEqual(
          starts[static_cast<std::size_t>(next.value)],
          ends[static_cast<std::size_t>(id)] +
              instance.EdgeDelay(view.worker, next_view.worker));
    }
    if (options.fifo_ordering) {
      if (const std::optional<OperationId> fifo =
              FifoPredecessor(instance, OperationId{id})) {
        builder.AddGreaterOrEqual(starts[static_cast<std::size_t>(id)],
                                  ends[static_cast<std::size_t>(fifo->value)]);
      }
    }
    if (!instance.HasRangeCostProfile()) {
      builder.AddGreaterOrEqual(
          starts[static_cast<std::size_t>(id)],
          ChainPrefixExpr(instance, layers, view.chain_index));
    }
  }

  std::vector<std::vector<IntervalVar>> worker_intervals(
      static_cast<std::size_t>(instance.workers));
  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    worker_intervals[static_cast<std::size_t>(view.worker)].push_back(
        intervals[static_cast<std::size_t>(id)]);
  }
  for (auto& worker : worker_intervals) builder.AddNoOverlap(worker);

  if (options.fifo_ordering && options.symmetry_break_f0_fifo) {
    // Valid in this phase because microbatches are identical and have no
    // microbatch-specific data, costs, or cross-microbatch dependencies. Any
    // feasible schedule can relabel microbatches by the order of their F_0
    // operations without changing the objective, yielding an equivalent FIFO
    // representative.
    for (Index b = 1; b < instance.microbatches; ++b) {
      const OperationId prev = EncodeOperation(instance, b - 1, 0);
      const OperationId current = EncodeOperation(instance, b, 0);
      builder.AddGreaterOrEqual(starts[static_cast<std::size_t>(current.value)],
                                ends[static_cast<std::size_t>(prev.value)]);
    }
  }

  IntVar makespan = builder.NewIntVar(Domain(0, horizon)).WithName("makespan");
  for (Index b = 0; b < instance.microbatches; ++b) {
    const OperationId final =
        EncodeOperation(instance, b, 2 * instance.stages - 1);
    builder.AddGreaterOrEqual(makespan,
                              ends[static_cast<std::size_t>(final.value)]);
  }
  builder.AddLessOrEqual(makespan, horizon);
  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    if (!instance.HasRangeCostProfile()) {
      builder.AddLessOrEqual(
          ends[static_cast<std::size_t>(id)] +
              ChainSuffixExpr(instance, layers, view.chain_index),
          makespan);
    }
  }

  if (!instance.HasCostProfile()) {
    const Tick ratio_sum =
        CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                   "joint critical-chain lower bound");
    builder.AddGreaterOrEqual(makespan,
                              CheckedMul(instance.total_layers, ratio_sum,
                                         "joint critical-chain lower bound"));
  }
  // Each duration variable already represents one F/B operation of one
  // microbatch. These redundant capacity bounds also cover variable profiles.
  std::vector<LinearExpr> worker_work(instance.workers);
  for (Index id = 0; id < op_count; ++id) {
    worker_work[DecodeOperation(instance, OperationId{id}).worker] += durations[id];
  }
  for (Index w = 0; w < instance.workers; ++w) {
    builder.AddGreaterOrEqual(makespan, worker_work[w])
        .WithName("worker_workload_bound_" + std::to_string(w));
    ++workload_constraint_count;
  }
  builder.AddGreaterOrEqual(instance.workers * LinearExpr(makespan), LinearExpr::Sum(durations))
      .WithName("global_workload_capacity_bound");
  ++workload_constraint_count;
  activation_metadata = AddActivationCapacityConstraints(
      builder, instance, options.activation_options, starts, ends, &layers,
      activation_fixed_split, horizon, activation_partition_optimized);
  activation_metadata.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      incumbent_structurally_valid && !incumbent_cap_feasible;
  if (options.activation_options.enforce_activation_cap &&
      !activation_metadata.constraints_added) {
    JointOptimizationResult result;
    result.status = "UNAVAILABLE";
    result.joint_status = "UNAVAILABLE";
    result.proven_optimal = false;
    result.wall_time_seconds = Since(started);
    result.timing.incumbent_seconds = incumbent_seconds;
    result.timing.model_build_seconds = Since(model_build_started);
    result.timing.total_seconds = result.wall_time_seconds;
    result.incumbent_budget_seconds = incumbent_effective_budget_seconds;
    result.incumbent_status = bfs_incumbent.status;
    result.incumbent_method_requested = incumbent_method_requested;
    result.incumbent_method_effective = EffectiveIncumbentMechanismName(
        incumbent_method_requested, bfs_incumbent);
    result.bfs_incumbent_method_requested = options.bfs_method;
    result.bfs_incumbent_method_effective =
        EffectiveBfsMethodName(bfs_incumbent.method);
    result.incumbent_source = incumbent_source;
    result.incumbent_feasible = false;
    result.incumbent_found = incumbent_structurally_valid;
    result.incumbent_valid = false;
    result.incumbent_makespan = bfs_incumbent.makespan_ticks;
    result.horizon_source = "none";
    result.incumbent_bound_effective = false;
    result.incumbent_bound_horizon = 0;
    result.hints_requested = incumbent_hints_requested;
    result.hints_effective = false;
    result.incumbent_hints_effective = false;
    result.hint_source = "none";
    result.hint_scope = "none";
    result.hint_termination_reason = "unavailable";
    result.fallback_available = false;
    result.fallback_source = "none";
    result.solution_source = "none";
    result.fallback_used = false;
    result.final_solution_source = "none";
    result.cp_sat_models_solved = incumbent_cp_sat_models_solved;
    result.symmetry_break_f0_fifo =
        options.fifo_ordering && options.symmetry_break_f0_fifo;
    result.num_workers = options.num_workers;
    result.worker_balance_constraint = worker_balance_constraint;
    result.pressure_pruning_stats = pressure_pruning_stats;
    result.activation_cap_constraints = activation_metadata;
    result.search_stats_enabled = stats != nullptr;
    result.diagnostic = activation_metadata.unsupported_reason.empty()
                            ? "activation cap constraints were not added"
                            : activation_metadata.unsupported_reason;
    result.bfs_incumbent = std::move(bfs_incumbent);
    populate_mechanism_provenance(&result);
    result.incumbent_bound_effective = false;
    result.incumbent_bound_horizon = 0;
    result.incumbent_hints_effective = false;
    result.final_solution_source = "none";
    if (stats != nullptr) result.search_stats = *stats;
    return result;
  }
  builder.Minimize(makespan);

  const bool hint_assignments_available = external_incumbent_available;
  const bool hints_effective =
      incumbent_hints_requested && hint_assignments_available;
  Index hinted_layer_variable_count = 0;
  Index hinted_operation_variable_count = 0;
  Index hinted_scalar_variable_count = 0;
  if (hints_effective) {
    ValidateHintAssignments(instance, bfs_incumbent, horizon,
                            options.partition_restriction);
    for (Index s = 0; s < instance.stages; ++s) {
      builder.AddHint(layers[static_cast<std::size_t>(s)],
                      bfs_incumbent.split[static_cast<std::size_t>(s)]);
    }
    hinted_layer_variable_count = instance.stages;
    for (const ScheduledOperation& op :
         bfs_incumbent.schedule.operations_by_id) {
      builder.AddHint(starts[static_cast<std::size_t>(op.id.value)], op.start);
      builder.AddHint(ends[static_cast<std::size_t>(op.id.value)], op.end);
      builder.AddHint(durations[static_cast<std::size_t>(op.id.value)],
                      op.duration);
    }
    hinted_operation_variable_count = 3 * instance.OperationCount();
    builder.AddHint(makespan, bfs_incumbent.makespan_ticks);
    hinted_scalar_variable_count = 1;
  }

  auto cp_model = builder.Build();
  const auto model_built = Clock::now();
  double model_build_seconds =
      std::chrono::duration<double>(model_built - model_build_started).count();
  const double joint_remaining_global_seconds =
      deadline.remaining_seconds_for_reporting();
  const double joint_phase_cap_seconds = joint_remaining_global_seconds;
  const double joint_effective_budget_seconds =
      deadline.clamp_solver_limit(joint_phase_cap_seconds);
  EmitLifecycle(options, started, "MODEL_BUILD_END", "", 0.0, "joint_cp_sat",
                joint_effective_budget_seconds, joint_remaining_global_seconds,
                joint_phase_cap_seconds);
  if (options.time_limit_seconds > 0.0 &&
      joint_effective_budget_seconds <= 0.0) {
    return make_deadline_fallback_result(
        "global_deadline_expired_before_cp_sat", model_build_seconds);
  }

  Index replay_cap_rejection_count = 0;
  std::optional<JointOptimizationResult> last_replay_cap_rejected_result;
  while (true) {
    const double solve_remaining_global_seconds =
        deadline.remaining_seconds_for_reporting();
    const double solve_phase_cap_seconds = solve_remaining_global_seconds;
    const double solve_effective_budget_seconds =
        deadline.clamp_solver_limit(solve_phase_cap_seconds);
    if (options.time_limit_seconds > 0.0 &&
        solve_effective_budget_seconds <= 0.0) {
      return make_deadline_fallback_result(
          replay_cap_rejection_count > 0
              ? "global_deadline_expired_after_replay_cap_rejections"
              : "global_deadline_expired_before_cp_sat",
          model_build_seconds,
          last_replay_cap_rejected_result ? &(*last_replay_cap_rejected_result)
                                          : nullptr);
    }

    Model model;
    SatParameters parameters;
    parameters.set_num_workers(options.num_workers);
    parameters.set_random_seed(options.random_seed);
    parameters.set_log_search_progress(options.log_search_progress);
    if (options.time_limit_seconds > 0.0) {
      parameters.set_max_time_in_seconds(solve_effective_budget_seconds);
    }
    model.Add(NewSatParameters(parameters));
    const auto solve_started = Clock::now();

    std::mutex incumbent_mutex;
    double time_to_first_feasible_seconds = 0.0;
    double time_to_best_incumbent_seconds = 0.0;
    Tick first_feasible_objective = 0;
    Tick best_incumbent_objective = 0;
    Index incumbent_improvement_count = 0;
    std::vector<std::pair<double, Tick>> incumbent_trace;
    model.Add(NewFeasibleSolutionObserver(
        [&](const CpSolverResponse& callback_response) {
          const double elapsed =
              std::chrono::duration<double>(Clock::now() - solve_started)
                  .count();
          const Tick objective =
              static_cast<Tick>(callback_response.objective_value());
          std::lock_guard<std::mutex> lock(incumbent_mutex);
          if (incumbent_trace.empty()) {
            time_to_first_feasible_seconds = elapsed;
            time_to_best_incumbent_seconds = elapsed;
            first_feasible_objective = objective;
            best_incumbent_objective = objective;
            incumbent_improvement_count = 1;
            incumbent_trace.push_back({elapsed, objective});
            return;
          }
          if (objective < best_incumbent_objective) {
            best_incumbent_objective = objective;
            time_to_best_incumbent_seconds = elapsed;
            ++incumbent_improvement_count;
            if (incumbent_trace.size() < 64) {
              incumbent_trace.push_back({elapsed, objective});
            }
          }
        }));
    EmitLifecycle(options, started, "SOLVE_START", "",
                  parameters.max_time_in_seconds(), "joint_cp_sat",
                  solve_effective_budget_seconds,
                  solve_remaining_global_seconds, solve_phase_cap_seconds);
    CpSolverResponse response;
    response = SolveCpModel(cp_model, &model);
    const double solver_seconds =
        std::chrono::duration<double>(Clock::now() - solve_started).count();
    EmitLifecycle(options, started, "SOLVE_END", StatusName(response.status()),
                  parameters.max_time_in_seconds(), "joint_cp_sat",
                  solve_effective_budget_seconds,
                  deadline.remaining_seconds_for_reporting(),
                  solve_phase_cap_seconds);

    JointOptimizationResult result;
    result.status = StatusName(response.status());
    result.best_bound_ticks = response.best_objective_bound();
    result.solver_objective_ticks = response.objective_value();
    result.wall_time_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    result.timing.incumbent_seconds = incumbent_seconds;
    result.timing.model_build_seconds = model_build_seconds;
    result.timing.solver_seconds = solver_seconds;
    result.timing.ortools_wall_time_seconds = response.wall_time();
    result.incumbent_budget_seconds = incumbent_effective_budget_seconds;
    result.incumbent_model_build_seconds =
        bfs_incumbent.timing.model_build_seconds;
    result.incumbent_solve_seconds = bfs_incumbent.timing.solver_seconds;
    if (result.incumbent_solve_seconds == 0.0 &&
        bfs_incumbent.wall_time_seconds > 0.0 &&
        bfs_incumbent.timing.model_build_seconds == 0.0) {
      result.incumbent_solve_seconds = bfs_incumbent.wall_time_seconds;
    }
    result.incumbent_status = bfs_incumbent.status;
    result.joint_budget_seconds = solve_effective_budget_seconds;
    result.joint_model_build_seconds = model_build_seconds;
    result.joint_solve_seconds = solver_seconds;
    result.joint_status = StatusName(response.status());
    result.cp_sat_models_solved =
        incumbent_cp_sat_models_solved + replay_cap_rejection_count + 1;
    result.incumbent_method_requested = incumbent_method_requested;
    result.incumbent_method_effective = EffectiveIncumbentMechanismName(
        incumbent_method_requested, bfs_incumbent);
    result.bfs_incumbent_method_requested = options.bfs_method;
    result.bfs_incumbent_method_effective =
        EffectiveBfsMethodName(bfs_incumbent.method);
    result.incumbent_source = incumbent_source;
    result.incumbent_feasible = external_incumbent_available;
    result.incumbent_primary_objective = bfs_incumbent.makespan_ticks;
    result.incumbent_found = incumbent_structurally_valid;
    result.incumbent_valid = external_incumbent_available;
    result.incumbent_makespan = bfs_incumbent.makespan_ticks;
    result.incumbent_hybrid_min_slack = incumbent_hybrid_min_slack;
    result.incumbent_baseline_primary_objective =
        incumbent_baseline_primary_objective;
    result.incumbent_baseline_hybrid_min_slack =
        incumbent_baseline_hybrid_min_slack;
    result.incumbent_improved_over_baseline =
        bfs_incumbent.improved_over_baseline;
    result.incumbent_hybrid_stage_scores = incumbent_hybrid_stage_scores;
    result.incumbent_hybrid_bottleneck_stages =
        incumbent_hybrid_bottleneck_stages;
    result.horizon_source =
        incumbent_bound_effective ? incumbent_source : "conservative";
    result.incumbent_bound_effective = incumbent_bound_effective;
    result.incumbent_bound_horizon =
        incumbent_bound_effective ? bfs_incumbent.makespan_ticks : 0;
    result.hint_budget_seconds = bfs_incumbent.hint_budget_seconds > 0.0
                                     ? bfs_incumbent.hint_budget_seconds
                                     : incumbent_effective_budget_seconds;
    result.hint_elapsed_seconds = bfs_incumbent.hint_elapsed_seconds > 0.0
                                      ? bfs_incumbent.hint_elapsed_seconds
                                      : incumbent_seconds;
    result.hint_iterations = bfs_incumbent.hint_iterations;
    result.hint_candidates_generated = bfs_incumbent.hint_candidates_generated;
    result.hint_candidates_simulated = bfs_incumbent.hint_candidates_simulated;
    result.hint_partition_moves_accepted =
        bfs_incumbent.hint_partition_moves_accepted;
    result.hint_interleaving_moves_accepted =
        bfs_incumbent.hint_interleaving_moves_accepted;
    result.hint_deadline_reached = bfs_incumbent.hint_deadline_reached;
    result.hint_termination_reason =
        bfs_incumbent.hint_termination_reason.empty()
            ? "not_applicable"
            : bfs_incumbent.hint_termination_reason;
    result.hints_requested = incumbent_hints_requested;
    result.hints_effective = hints_effective;
    result.incumbent_hints_effective = hints_effective;
    result.hint_source = hints_effective ? incumbent_source : "none";
    result.hint_scope = hints_effective ? "basic_integer_variables" : "none";
    result.hint_complete_for_basic_model = hints_effective;
    result.hint_complete_for_full_model =
        hints_effective && auxiliary_variable_count == 0;
    result.hinted_layer_variable_count = hinted_layer_variable_count;
    result.hinted_operation_variable_count = hinted_operation_variable_count;
    result.hinted_scalar_variable_count = hinted_scalar_variable_count;
    result.hinted_auxiliary_variable_count = 0;
    result.hinted_total_variable_count = hinted_layer_variable_count +
                                         hinted_operation_variable_count +
                                         hinted_scalar_variable_count;
    result.auxiliary_variable_count = auxiliary_variable_count;
    result.fallback_available =
        incumbent_fallback_enabled && external_incumbent_available;
    result.fallback_source =
        result.fallback_available ? incumbent_source : "none";
    result.solution_source = "none";
    result.fallback_used = false;
    result.deterministic_time = response.deterministic_time();
    result.branches = response.num_branches();
    result.conflicts = response.num_conflicts();
    const double solve_phase_offset_seconds =
        std::chrono::duration<double>(solve_started - started).count();
    if (external_incumbent_available) {
      result.time_to_first_feasible_seconds = incumbent_seconds;
      result.time_to_best_incumbent_seconds = incumbent_seconds;
      result.first_feasible_objective = bfs_incumbent.makespan_ticks;
    }
    if (time_to_first_feasible_seconds > 0.0) {
      result.time_to_first_cpsat_feasible_seconds =
          solve_phase_offset_seconds + time_to_first_feasible_seconds;
      result.first_cpsat_feasible_objective = first_feasible_objective;
    }
    if (time_to_first_feasible_seconds > 0.0 && !external_incumbent_available) {
      result.time_to_first_feasible_seconds =
          solve_phase_offset_seconds + time_to_first_feasible_seconds;
      result.first_feasible_objective = first_feasible_objective;
    }
    if (time_to_best_incumbent_seconds > 0.0 &&
        (!external_incumbent_available ||
         best_incumbent_objective < bfs_incumbent.makespan_ticks)) {
      result.time_to_best_incumbent_seconds =
          solve_phase_offset_seconds + time_to_best_incumbent_seconds;
    }
    result.incumbent_improvement_count = incumbent_improvement_count;
    result.incumbent_trace = incumbent_trace;
    for (auto& entry : result.incumbent_trace) {
      entry.first += solve_phase_offset_seconds;
    }
    result.proven_optimal =
        response.status() == operations_research::sat::OPTIMAL;
    result.symmetry_break_f0_fifo =
        options.fifo_ordering && options.symmetry_break_f0_fifo;
    result.num_workers = options.num_workers;
    result.worker_balance_constraint = worker_balance_constraint;
    result.pressure_pruning_stats = pressure_pruning_stats;
    result.activation_cap_constraints = activation_metadata;
    result.search_stats_enabled = stats != nullptr;
    populate_mechanism_provenance(&result);
    result.incumbent_bound_effective = incumbent_bound_effective;
    result.incumbent_bound_horizon =
        incumbent_bound_effective ? bfs_incumbent.makespan_ticks : 0;
    result.incumbent_hints_effective = hints_effective;
    if (stats != nullptr) {
      if (stats->algorithm.empty()) stats->algorithm = "optimize-joint";
      stats->enumerative_search = false;
      stats->stage_partitions_note =
          "Stage partitions are encoded as CP-SAT "
          "variables, not explicitly enumerated.";
      stats->interleave_orders_note =
          "Worker-local orders are encoded in the CP-SAT model, not explicitly "
          "enumerated.";
      stats->cp_sat_available = true;
      stats->cp_sat_status = result.status;
      stats->cp_sat_objective = result.solver_objective_ticks;
      stats->cp_sat_best_bound = result.best_bound_ticks;
      stats->cp_sat_branches = result.branches;
      stats->cp_sat_conflicts = result.conflicts;
      stats->cp_sat_wall_time_seconds = result.timing.ortools_wall_time_seconds;
      stats->cp_sat_deterministic_time = result.deterministic_time;
    }
    // A replay-cap rejection can discard this per-attempt result and retry the
    // solve loop, so the shared incumbent must remain intact for later
    // fallbacks and hints.
    result.bfs_incumbent = bfs_incumbent;

    auto apply_incumbent_fallback = [&](const std::string& reason) {
      result.status = "FEASIBLE";
      result.proven_optimal = false;
      result.split = result.bfs_incumbent.split;
      result.machine_orders = result.bfs_incumbent.machine_orders;
      result.machine_predecessors = ExtractMachinePredecessors(
          instance, result.machine_orders, options.fifo_ordering);
      result.solver_objective_ticks = result.bfs_incumbent.makespan_ticks;
      result.makespan_ticks = result.bfs_incumbent.makespan_ticks;
      result.schedule = result.bfs_incumbent.schedule;
      result.solution_source = "bfs_incumbent_fallback";
      result.fallback_used = true;
      result.final_solution_available = true;
      result.final_solution_source = "external_incumbent";
      result.no_solution_reason.clear();
      result.diagnostic = reason;
      EvaluationResult checked = EvaluateSchedule(
          instance, result.split, result.machine_orders, options.fifo_ordering);
      if (!checked.schedule.ok() ||
          checked.schedule.makespan != result.makespan_ticks) {
        throw Error(BfsFallbackReplayMismatchMessage(
            result.split, result.machine_orders, result.makespan_ticks,
            checked.schedule));
      }
      std::string validation_reason;
      if (!ValidateScheduleForJointReturn(
              instance, checked.schedule, result.status,
              "joint-unrestricted-no-overlap", options.fifo_ordering,
              &validation_reason)) {
        throw Error(
            "validated BFS incumbent fallback failed independent "
            "validation: " +
            validation_reason);
      }
      if (options.activation_options.enforce_activation_cap &&
          !ActivationScheduleSatisfiesCap(instance, checked.schedule,
                                          options.activation_options)) {
        throw Error("validated BFS incumbent fallback violates activation cap");
      }
      result.schedule = std::move(checked.schedule);
    };

    if (response.status() == operations_research::sat::OPTIMAL ||
        response.status() == operations_research::sat::FEASIBLE) {
      const auto extraction_started = Clock::now();
      EmitLifecycle(options, started, "SOLUTION_EXTRACTION_START",
                    result.status, parameters.max_time_in_seconds(),
                    "joint_cp_sat");
      result.split.reserve(static_cast<std::size_t>(instance.stages));
      for (const IntVar& layer : layers) {
        result.split.push_back(SolutionIntegerValue(response, layer));
      }
      if (stats != nullptr) {
        ++stats->candidate_schedules_extracted;
      }

      std::vector<RawOperationTime> raw_times(
          static_cast<std::size_t>(op_count));
      for (Index id = 0; id < op_count; ++id) {
        raw_times[static_cast<std::size_t>(id)] = RawOperationTime{
            OperationId{id}, SolutionIntegerValue(response, starts[id]),
            SolutionIntegerValue(response, ends[id])};
      }
      result.machine_orders = ExtractOrdersFromRawTimes(instance, raw_times);
      result.machine_predecessors = ExtractMachinePredecessors(
          instance, result.machine_orders, options.fifo_ordering);
      result.timing.extraction_seconds =
          std::chrono::duration<double>(Clock::now() - extraction_started)
              .count();
      EmitLifecycle(options, started, "SOLUTION_EXTRACTION_END", result.status,
                    parameters.max_time_in_seconds(), "joint_cp_sat");
      const auto canonical_started = Clock::now();
      EmitLifecycle(options, started, "REPLAY_START", result.status,
                    parameters.max_time_in_seconds(), "joint_cp_sat");
      ValidateSplit(instance, result.split);
      if (options.partition_restriction &&
          !SplitSatisfiesPartitionRestriction(instance, result.split,
                                              *options.partition_restriction)) {
        throw Error(PartitionRestrictionViolationMessage(
            instance, result.split, *options.partition_restriction));
      }
      ValidateRawNoOverlap(instance, result.machine_orders, raw_times);
      if (stats != nullptr)
        ++stats->candidate_schedules_deterministically_evaluated;
      EvaluationResult evaluated = EvaluateSchedule(
          instance, result.split, result.machine_orders, options.fifo_ordering);
      if (!evaluated.schedule.ok()) {
        if (stats != nullptr) ++stats->candidate_schedules_rejected;
        std::ostringstream msg;
        msg << "deterministic evaluator rejected joint CP-SAT worker orders: ";
        if (!evaluated.schedule.validation_errors.empty()) {
          msg << evaluated.schedule.validation_errors.front();
        } else {
          msg << "unknown validation failure";
        }
        throw Error(msg.str());
      }
      if (stats != nullptr) ++stats->candidate_schedules_accepted;
      if (evaluated.schedule.makespan >
          static_cast<Tick>(response.objective_value())) {
        std::ostringstream msg;
        msg << "MAKESPAN_MISMATCH joint CP-SAT extraction evaluates worse than "
            << "solver objective: evaluated=" << evaluated.schedule.makespan
            << " solver_objective=" << response.objective_value();
        throw Error(msg.str());
      }
      if (evaluated.schedule.makespan <
          static_cast<Tick>(response.objective_value())) {
        std::ostringstream diagnostic;
        diagnostic << "independent evaluator improved raw CP-SAT makespan from "
                   << response.objective_value() << " to "
                   << evaluated.schedule.makespan;
        result.diagnostic = diagnostic.str();
      }
      result.makespan_ticks = evaluated.schedule.makespan;
      result.schedule = std::move(evaluated.schedule);
      std::string validation_reason;
      if (!ValidateScheduleForJointReturn(
              instance, result.schedule, result.status,
              "joint-unrestricted-no-overlap", options.fifo_ordering,
              &validation_reason)) {
        if (result.fallback_available) {
          apply_incumbent_fallback(
              "joint CP-SAT solution failed independent validation: " +
              validation_reason +
              "; returned BFS incumbent after CP-SAT "
              "status=" +
              StatusName(response.status()));
        } else {
          throw Error("joint CP-SAT solution failed independent validation: " +
                      validation_reason);
        }
      } else {
        result.solver_solution_available = true;
        if (result.fallback_available &&
            result.bfs_incumbent.makespan_ticks > 0 &&
            result.makespan_ticks > result.bfs_incumbent.makespan_ticks) {
          std::ostringstream diagnostic;
          diagnostic << "joint CP-SAT solution objective "
                     << result.makespan_ticks
                     << " is worse than external incumbent "
                     << result.bfs_incumbent.makespan_ticks
                     << "; returned external incumbent";
          apply_incumbent_fallback(diagnostic.str());
        } else {
          result.solution_source =
              result.bfs_incumbent.makespan_ticks > 0 &&
                      result.makespan_ticks <
                          result.bfs_incumbent.makespan_ticks
                  ? "joint_cpsat_improved_bfs"
                  : "joint_cpsat";
          result.fallback_used = false;
          result.final_solution_available = true;
          result.final_solution_source = "cpsat";
          result.no_solution_reason.clear();
        }
      }
      result.timing.canonicalization_seconds =
          std::chrono::duration<double>(Clock::now() - canonical_started)
              .count();
      EmitLifecycle(options, started, "REPLAY_END", result.status,
                    parameters.max_time_in_seconds(), "joint_cp_sat");
    } else if (result.bfs_incumbent.schedule.ok() &&
               result.fallback_available) {
      EmitLifecycle(options, started, "SOLUTION_EXTRACTION_START",
                    result.status, parameters.max_time_in_seconds(),
                    "bfs_incumbent_fallback");
      std::ostringstream diagnostic;
      diagnostic << "returned BFS incumbent after CP-SAT status="
                 << StatusName(response.status());
      apply_incumbent_fallback(diagnostic.str());
      EmitLifecycle(options, started, "SOLUTION_EXTRACTION_END", result.status,
                    parameters.max_time_in_seconds(), "bfs_incumbent_fallback");
      EmitLifecycle(options, started, "REPLAY_START", result.status,
                    parameters.max_time_in_seconds(), "bfs_incumbent_fallback");
      EmitLifecycle(options, started, "REPLAY_END", result.status,
                    parameters.max_time_in_seconds(), "bfs_incumbent_fallback");
    }

    if (options.require_optimal && !result.proven_optimal) {
      result.status += "_REJECTED_REQUIRE_OPTIMAL";
    }
    result.timing.total_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    result.wall_time_seconds = result.timing.total_seconds;
    if (stats != nullptr) result.search_stats = *stats;
    if (options.activation_options.enforce_activation_cap &&
        IsTerminalFeasibleStatus(result.status) && result.schedule.ok() &&
        result.schedule.operations_by_id.size() ==
            static_cast<std::size_t>(instance.OperationCount()) &&
        !ActivationScheduleSatisfiesCap(instance, result.schedule,
                                        options.activation_options)) {
      if (AddReplayCombinationNogood(builder, instance, layers, starts, ends,
                                     result.split, result.machine_orders,
                                     replay_cap_rejection_count)) {
        last_replay_cap_rejected_result = result;
        ++replay_cap_rejection_count;
        const auto rebuild_started = Clock::now();
        cp_model = builder.Build();
        model_build_seconds +=
            std::chrono::duration<double>(Clock::now() - rebuild_started)
                .count();
        continue;
      }
      result.status = "INFEASIBLE";
      result.joint_status = "INFEASIBLE";
      result.proven_optimal = false;
      result.split.clear();
      result.machine_orders.clear();
      result.machine_predecessors.clear();
      result.schedule = ScheduleSolution{};
      result.makespan_ticks = 0;
      result.solution_source = "none";
      result.fallback_used = false;
      result.solver_solution_available = false;
      result.final_solution_available = false;
      result.final_solution_source = "none";
      result.no_solution_reason = "replay_validation_failed";
      result.diagnostic =
          "replayed CP-SAT solution violates activation cap and no exact "
          "partition/order no-good could be added";
    } else if (replay_cap_rejection_count > 0) {
      std::ostringstream diagnostic;
      if (!result.diagnostic.empty()) {
        diagnostic << result.diagnostic << "; ";
      }
      diagnostic << "rejected " << replay_cap_rejection_count
                 << " CP-SAT solution(s) whose canonical replay violated the "
                    "activation cap";
      result.diagnostic = diagnostic.str();
    }
    const bool has_complete_schedule =
        ScheduleCompleteForInstance(instance, result.schedule);
    if (!has_complete_schedule && result.status != "UNAVAILABLE") {
      result.status = "NO_VALID_SOLUTION";
      result.proven_optimal = false;
      result.makespan_ticks = 0;
      result.solution_source = "none";
      result.fallback_used = false;
      result.final_solution_available = false;
      result.final_solution_source = "none";
      if (result.no_solution_reason.empty()) {
        result.no_solution_reason =
            NoSolutionReasonForCpSatStatus(response.status());
      }
    } else if (has_complete_schedule) {
      result.final_solution_available = true;
    }
    return result;
  }
}

JointOptimizationResult OptimizeScheduleForFixedSplitCpSat(
    const Instance& instance, const std::vector<Tick>& fixed_split,
    const JointOptimizerOptions& options) {
  instance.Validate();
  ValidateSplit(instance, fixed_split);
  if (options.partition_restriction) {
    throw Error(
        "schedule-only does not accept an existing partition restriction");
  }
  if (options.bfs_incumbent_override) {
    throw Error("schedule-only constructs its own fixed-split incumbent");
  }

  JointOptimizerOptions fixed_options = options;
  fixed_options.bfs_method = "fixed";
  PartitionRestriction restriction;
  restriction.mode = SlackPipeSplitMode::kFixed;
  restriction.reference_split = fixed_split;
  ValidatePartitionRestriction(instance, restriction);
  fixed_options.partition_restriction = restriction;
  fixed_options.bfs_incumbent_override =
      BuildBfsIncumbentForSplit(instance, fixed_split, "schedule-only-hint");

  JointOptimizationResult result =
      OptimizeJointSplitAndScheduleCpSat(instance, fixed_options);
  if ((result.status == "OPTIMAL" || result.status == "FEASIBLE") &&
      result.split != fixed_split) {
    throw Error("SCHEDULE_ONLY_PARTITION_MUTATION fixed partition changed");
  }
  return result;
}

bool CpSatPartitionRestrictionAcceptsSplitForTesting(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction) {
  instance.Validate();
  ValidateSplit(instance, split);
  ValidatePartitionRestriction(instance, restriction);

  CpModelBuilder builder;
  std::vector<IntVar> layers;
  layers.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    layers.push_back(
        builder.NewIntVar(Domain(instance.min_layers, instance.total_layers))
            .WithName("layers_" + std::to_string(s)));
  }
  builder.AddEquality(LinearExpr::Sum(layers), instance.total_layers);
  AddPartitionRestrictionConstraints(builder, instance, layers, restriction);
  for (Index s = 0; s < instance.stages; ++s) {
    builder.AddEquality(layers[static_cast<std::size_t>(s)],
                        split[static_cast<std::size_t>(s)]);
  }

  Model model;
  SatParameters parameters;
  parameters.set_num_workers(1);
  parameters.set_max_time_in_seconds(1.0);
  model.Add(NewSatParameters(parameters));
  const CpSolverResponse response = SolveCpModel(builder.Build(), &model);
  return response.status() == operations_research::sat::OPTIMAL ||
         response.status() == operations_research::sat::FEASIBLE;
}

bool IsJointOptimizerAvailable() { return true; }

}  // namespace slackpipe

#endif
