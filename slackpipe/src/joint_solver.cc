#include "slackpipe/joint_solver.h"

#include "slackpipe/breadth_first.h"
#include "slackpipe/operation.h"
#include "slackpipe/slackpipe_solver.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace slackpipe {

namespace {

[[nodiscard]] Index FifoConstraintCountForOptions(
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

void ApplyFifoProvenance(const Instance& instance,
                         const JointOptimizerOptions& options,
                         JointOptimizationResult* result) {
  result->fifo_ordering_requested = options.fifo_ordering;
  result->fifo_ordering_effective = options.fifo_ordering;
  result->fifo_constraint_count =
      FifoConstraintCountForOptions(instance, options);
  result->symmetry_break_f0_fifo =
      options.fifo_ordering && options.symmetry_break_f0_fifo;
}

}  // namespace

WorkerBalanceConstraintResult ComputeWorkerBalanceConstraint(
    const Instance& instance, double tolerance_percent,
    std::optional<Index> tolerance_layers) {
  instance.Validate();
  WorkerBalanceConstraintResult result;
  if (tolerance_percent < 0.0 && !tolerance_layers) {
    return result;
  }
  if (tolerance_layers && *tolerance_layers < 0) {
    return result;
  }
  if (tolerance_percent < 0.0 && tolerance_layers) {
    result.tolerance_percent = -1.0;
  } else {
    result.tolerance_percent = tolerance_percent;
  }
  result.enabled = true;
  if (tolerance_percent >= 0.0) {
    result.tolerance_layers = static_cast<Index>(
        std::ceil((tolerance_percent / 100.0) *
                  static_cast<double>(instance.total_layers)));
  }
  if (tolerance_layers) {
    result.tolerance_layers = *tolerance_layers;
  }
  if (result.tolerance_layers < 0) {
    throw Error("worker balance tolerance layers must be non-negative");
  }
  const Tick base_lower = instance.total_layers / instance.workers;
  const Tick base_upper =
      instance.total_layers / instance.workers +
      (instance.total_layers % instance.workers == 0 ? 0 : 1);
  result.lower_bound = std::max<Tick>(0, base_lower - result.tolerance_layers);
  result.upper_bound = base_upper + result.tolerance_layers;
  return result;
}

bool WorkerBalanceToleranceProvided(const JointOptimizerOptions& options) {
  return options.worker_balance_tolerance_percent >= 0.0 ||
         options.worker_balance_tolerance_layers.has_value();
}

bool WorkerBalancePruningRequested(const JointOptimizerOptions& options) {
  return options.worker_balance_pruning.value_or(
      WorkerBalanceToleranceProvided(options));
}

WorkerBalanceConstraintResult ComputeJointWorkerBalanceConstraint(
    const Instance& instance, const JointOptimizerOptions& options) {
  const bool tolerance_provided = WorkerBalanceToleranceProvided(options);
  const bool pruning_requested = WorkerBalancePruningRequested(options);
  if (!pruning_requested) {
    return WorkerBalanceConstraintResult{};
  }
  if (!tolerance_provided) {
    throw Error(
        "--worker-balance-pruning=on requires "
        "--worker-balance-tolerance-percent or "
        "--worker-balance-tolerance-layers");
  }
  return ComputeWorkerBalanceConstraint(
      instance, options.worker_balance_tolerance_percent,
      options.worker_balance_tolerance_layers);
}

std::string RequestedIncumbentMethodName(const JointOptimizerOptions& options) {
  if (options.incumbent_method) {
    return *options.incumbent_method;
  }
  if (options.bfs_method == "auto" || options.bfs_method == "hybrid-slack") {
    return "slack";
  }
  if (options.bfs_method == "uniform") {
    return "canonical";
  }
  if (options.bfs_method == "none") {
    return "none";
  }
  return options.bfs_method;
}

bool IncumbentHintsRequested(const JointOptimizerOptions& options) {
  return options.incumbent_hints.value_or(options.use_bfs_hints);
}

bool IncumbentFallbackRequested(const JointOptimizerOptions& options) {
  return RequestedIncumbentMethodName(options) != "none" &&
         options.incumbent_fallback;
}

void ValidateJointOptimizerMechanismOptions(
    const JointOptimizerOptions& options) {
  const std::string incumbent_method = RequestedIncumbentMethodName(options);
  if (incumbent_method != "slack" && incumbent_method != "canonical" &&
      incumbent_method != "none" && incumbent_method != "auto" &&
      incumbent_method != "hybrid-slack" && incumbent_method != "uniform" &&
      incumbent_method != "enumerate" && incumbent_method != "cpsat" &&
      incumbent_method != "fixed" &&
      incumbent_method != "heuristic") {
    throw Error("unknown incumbent method: " + incumbent_method);
  }
  if (incumbent_method == "none") {
    if (options.incumbent_bound) {
      throw Error(
          "--incumbent-method=none cannot be combined with "
          "--incumbent-bound=on");
    }
    if (IncumbentHintsRequested(options)) {
      throw Error(
          "--incumbent-method=none cannot be combined with "
          "--incumbent-hints=on");
    }
  }
  if (options.worker_balance_pruning == false &&
      WorkerBalanceToleranceProvided(options)) {
    throw Error(
        "--worker-balance-pruning=off cannot be combined with worker "
        "balance tolerance flags");
  }
  if (options.worker_balance_pruning == true &&
      !WorkerBalanceToleranceProvided(options)) {
    throw Error(
        "--worker-balance-pruning=on requires "
        "--worker-balance-tolerance-percent or "
        "--worker-balance-tolerance-layers");
  }
}

JointOptimizationResult BuildScheduleOnlyFixedSplitDeadlineFallback(
    const Instance& instance, const std::vector<Tick>& fixed_split,
    const JointOptimizerOptions& options, double elapsed_seconds,
    const std::string& reason) {
  instance.Validate();
  ValidateSplit(instance, fixed_split);
  ValidateJointOptimizerMechanismOptions(options);
  const MachineOrders orders = BreadthFirstOrders(instance);
  EvaluationResult evaluated =
      EvaluateSchedule(instance, fixed_split, orders, options.fifo_ordering);
  const bool cap_feasible =
      !options.activation_options.enforce_activation_cap ||
      ActivationScheduleSatisfiesCap(instance, evaluated.schedule,
                                     options.activation_options);
  const WorkerBalanceConstraintResult worker_balance_constraint =
      ComputeJointWorkerBalanceConstraint(instance, options);
  const bool hints_requested = IncumbentHintsRequested(options);
  const bool fallback_enabled = IncumbentFallbackRequested(options);
  const bool fallback_usable =
      fallback_enabled && evaluated.schedule.ok() && cap_feasible;

  JointOptimizationResult result;
  result.split = fixed_split;
  result.status = fallback_usable ? "FEASIBLE" : "NOT_RUN";
  result.joint_status = "NOT_RUN";
  result.proven_optimal = false;
  result.wall_time_seconds = elapsed_seconds;
  result.timing.total_seconds = elapsed_seconds;
  result.incumbent_status = evaluated.schedule.ok() ? "FEASIBLE" : "INVALID";
  result.incumbent_method_requested = "fixed";
  result.incumbent_method_effective = "fixed";
  result.incumbent_method_requested_normalized = "fixed";
  result.bfs_incumbent_method_requested = "fixed";
  result.bfs_incumbent_method_effective = "fixed";
  result.incumbent_source = "schedule_only_fixed_split";
  result.incumbent_feasible = evaluated.schedule.ok() && cap_feasible;
  result.incumbent_found = evaluated.schedule.ok();
  result.incumbent_valid = evaluated.schedule.ok() && cap_feasible;
  result.incumbent_makespan =
      evaluated.schedule.ok() ? evaluated.schedule.makespan : 0;
  result.horizon_source = evaluated.schedule.ok() && cap_feasible
                              ? "schedule_only_fixed_split"
                              : "none";
  result.incumbent_bound_requested = options.incumbent_bound;
  result.incumbent_bound_effective =
      options.incumbent_bound && evaluated.schedule.ok() && cap_feasible;
  result.incumbent_bound_horizon =
      result.incumbent_bound_effective ? evaluated.schedule.makespan : 0;
  result.hints_requested = hints_requested;
  result.hints_effective = false;
  result.incumbent_hints_requested = hints_requested;
  result.incumbent_hints_effective = false;
  result.hint_source = "none";
  result.hint_scope = "none";
  result.hint_termination_reason = "not_applicable";
  result.fallback_available = fallback_usable;
  result.fallback_enabled = fallback_enabled;
  result.fallback_source =
      result.fallback_available ? "schedule_only_fixed_split" : "none";
  result.solution_source =
      result.fallback_available ? "fixed_split_breadth_first_fallback" : "none";
  result.fallback_used = result.fallback_available;
  result.final_solution_source =
      result.fallback_used ? "external_incumbent" : "none";
  result.global_time_limit_seconds = options.time_limit_seconds;
  result.cp_sat_models_solved = 0;
  ApplyFifoProvenance(instance, options, &result);
  result.num_workers = options.num_workers;
  result.worker_balance_constraint = worker_balance_constraint;
  result.worker_balance_pruning_requested =
      WorkerBalancePruningRequested(options);
  result.worker_balance_pruning_effective = worker_balance_constraint.enabled;
  if (options.worker_balance_tolerance_percent >= 0.0) {
    result.worker_balance_tolerance_requested_percent =
        options.worker_balance_tolerance_percent;
  }
  result.worker_balance_tolerance_requested_layers =
      options.worker_balance_tolerance_layers;
  result.diagnostic = reason;
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported =
      ActivationCapSolverCanEnforce(options.activation_options, false);
  result.activation_cap_constraints.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      evaluated.schedule.ok() && !cap_feasible;
  if (options.activation_options.enforce_activation_cap &&
      !result.activation_cap_constraints.solver_supported) {
    result.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, false);
  }
  if (fallback_usable) {
    result.makespan_ticks = evaluated.schedule.makespan;
    result.solver_objective_ticks =
        static_cast<double>(evaluated.schedule.makespan);
    result.schedule = std::move(evaluated.schedule);
    result.machine_orders = orders;
    result.machine_predecessors = ExtractMachinePredecessors(
        instance, result.machine_orders, options.fifo_ordering);
    result.incumbent_primary_objective = result.makespan_ticks;
    result.time_to_first_feasible_seconds = elapsed_seconds;
    result.time_to_best_incumbent_seconds = elapsed_seconds;
    result.first_feasible_objective = result.makespan_ticks;
    result.bfs_incumbent.method = "schedule-only-hint";
    result.bfs_incumbent.status = "FEASIBLE";
    result.bfs_incumbent.split = fixed_split;
    result.bfs_incumbent.makespan_ticks = result.makespan_ticks;
    result.bfs_incumbent.machine_orders = result.machine_orders;
    result.bfs_incumbent.schedule = result.schedule;
  } else if (evaluated.schedule.ok() && !cap_feasible) {
    result.diagnostic += ": fixed-split fallback violates activation cap";
  } else if (!evaluated.schedule.validation_errors.empty()) {
    result.diagnostic += ": " + evaluated.schedule.validation_errors.front();
  }
  if (options.require_optimal && !result.proven_optimal) {
    result.status += "_REJECTED_REQUIRE_OPTIMAL";
  }
  return result;
}

#if !SLACKPIPE_HAVE_ORTOOLS
JointOptimizationResult OptimizeJointSplitAndScheduleCpSat(
    const Instance& instance, const JointOptimizerOptions& options) {
  instance.Validate();
  ValidateJointOptimizerMechanismOptions(options);
  JointOptimizationResult result;
  const WorkerBalanceConstraintResult worker_balance_constraint =
      ComputeJointWorkerBalanceConstraint(instance, options);
  const std::string incumbent_method = RequestedIncumbentMethodName(options);
  const bool hints_requested = IncumbentHintsRequested(options);
  result.status = "UNAVAILABLE";
  result.proven_optimal = false;
  result.num_workers = options.num_workers;
  ApplyFifoProvenance(instance, options, &result);
  result.incumbent_status = "UNAVAILABLE";
  result.joint_status = "UNAVAILABLE";
  result.incumbent_method_requested = incumbent_method;
  result.incumbent_method_effective = "unavailable";
  result.bfs_incumbent_method_requested = options.bfs_method;
  result.bfs_incumbent_method_effective = "unavailable";
  result.incumbent_source = "none";
  result.incumbent_feasible = false;
  result.incumbent_primary_objective = 0;
  result.incumbent_hybrid_min_slack = 0.0;
  result.incumbent_baseline_primary_objective = 0;
  result.incumbent_baseline_hybrid_min_slack = 0.0;
  result.incumbent_improved_over_baseline = false;
  result.horizon_source = "none";
  result.hint_budget_seconds = 0.0;
  result.hint_elapsed_seconds = 0.0;
  result.hint_termination_reason = "unavailable";
  result.incumbent_method_requested_normalized = incumbent_method;
  result.incumbent_bound_requested = options.incumbent_bound;
  result.incumbent_bound_effective = false;
  result.incumbent_hints_requested = hints_requested;
  result.hints_requested = hints_requested;
  result.hints_effective = false;
  result.incumbent_hints_effective = false;
  result.hint_source = "none";
  result.hint_scope = "none";
  result.hint_complete_for_basic_model = false;
  result.hint_complete_for_full_model = false;
  result.fallback_available = false;
  result.fallback_source = "none";
  result.solution_source = "none";
  result.fallback_used = false;
  result.final_solution_source = "none";
  result.fallback_enabled = IncumbentFallbackRequested(options);
  result.global_time_limit_seconds = options.time_limit_seconds;
  result.worker_balance_constraint = worker_balance_constraint;
  result.worker_balance_pruning_requested =
      WorkerBalancePruningRequested(options);
  result.worker_balance_pruning_effective = worker_balance_constraint.enabled;
  if (options.worker_balance_tolerance_percent >= 0.0) {
    result.worker_balance_tolerance_requested_percent =
        options.worker_balance_tolerance_percent;
  }
  result.worker_balance_tolerance_requested_layers =
      options.worker_balance_tolerance_layers;
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported =
      ActivationCapSolverCanEnforce(options.activation_options, true);
  if (options.activation_options.enforce_activation_cap) {
    result.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, true);
  }
  result.diagnostic =
      "optimize-joint requires a build with OR-Tools CP-SAT support";
  return result;
}

JointOptimizationResult OptimizeScheduleForFixedSplitCpSat(
    const Instance& instance, const std::vector<Tick>& fixed_split,
    const JointOptimizerOptions& options) {
  instance.Validate();
  ValidateSplit(instance, fixed_split);
  ValidateJointOptimizerMechanismOptions(options);
  JointOptimizationResult result;
  const WorkerBalanceConstraintResult worker_balance_constraint =
      ComputeJointWorkerBalanceConstraint(instance, options);
  const bool hints_requested = IncumbentHintsRequested(options);
  result.status = "UNAVAILABLE";
  result.proven_optimal = false;
  result.num_workers = options.num_workers;
  ApplyFifoProvenance(instance, options, &result);
  result.incumbent_status = "UNAVAILABLE";
  result.joint_status = "UNAVAILABLE";
  result.incumbent_method_requested = "fixed";
  result.incumbent_method_effective = "unavailable";
  result.bfs_incumbent_method_requested = "fixed";
  result.bfs_incumbent_method_effective = "unavailable";
  result.incumbent_source = "none";
  result.incumbent_feasible = false;
  result.incumbent_primary_objective = 0;
  result.incumbent_hybrid_min_slack = 0.0;
  result.incumbent_baseline_primary_objective = 0;
  result.incumbent_baseline_hybrid_min_slack = 0.0;
  result.incumbent_improved_over_baseline = false;
  result.horizon_source = "none";
  result.hint_budget_seconds = 0.0;
  result.hint_elapsed_seconds = 0.0;
  result.hint_termination_reason = "unavailable";
  result.incumbent_method_requested_normalized = "fixed";
  result.incumbent_bound_requested = options.incumbent_bound;
  result.incumbent_bound_effective = false;
  result.incumbent_hints_requested = hints_requested;
  result.hints_requested = hints_requested;
  result.hints_effective = false;
  result.incumbent_hints_effective = false;
  result.hint_source = "none";
  result.hint_scope = "none";
  result.hint_complete_for_basic_model = false;
  result.hint_complete_for_full_model = false;
  result.fallback_available = false;
  result.fallback_source = "none";
  result.solution_source = "none";
  result.fallback_used = false;
  result.final_solution_source = "none";
  result.fallback_enabled = IncumbentFallbackRequested(options);
  result.global_time_limit_seconds = options.time_limit_seconds;
  result.worker_balance_constraint = worker_balance_constraint;
  result.worker_balance_pruning_requested =
      WorkerBalancePruningRequested(options);
  result.worker_balance_pruning_effective = worker_balance_constraint.enabled;
  if (options.worker_balance_tolerance_percent >= 0.0) {
    result.worker_balance_tolerance_requested_percent =
        options.worker_balance_tolerance_percent;
  }
  result.worker_balance_tolerance_requested_layers =
      options.worker_balance_tolerance_layers;
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported =
      ActivationCapSolverCanEnforce(options.activation_options, false);
  if (options.activation_options.enforce_activation_cap) {
    result.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, false);
  }
  result.split = fixed_split;
  result.diagnostic =
      "schedule-only requires a build with OR-Tools CP-SAT support";
  return result;
}

bool IsJointOptimizerAvailable() { return false; }

bool CpSatPartitionRestrictionAcceptsSplitForTesting(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction) {
  return SplitSatisfiesPartitionRestriction(instance, split, restriction);
}
#endif

}  // namespace slackpipe
