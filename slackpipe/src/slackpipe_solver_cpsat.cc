#include "slackpipe/slackpipe_solver.h"

#if SLACKPIPE_HAVE_ORTOOLS

#include <algorithm>
#include <chrono>
#include <limits>

#include "slackpipe/breadth_first.h"
#include "slackpipe/deadline.h"
#include "slackpipe/hybrid_slack.h"
#include "slackpipe/joint_solver.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kReferenceBudgetFraction = kSlackPipePreparationBudgetFraction;

[[nodiscard]] BfsSplitOptimizationResult UniformBfsIncumbent(
    const Instance& instance) {
  const auto started = Clock::now();
  BfsSplitOptimizationResult result;
  result.method = "uniform";
  result.status = "FEASIBLE";
  result.proven_optimal = false;
  result.split = UniformSplit(instance);
  result.best_bound_ticks = 0;
  result.machine_orders = BreadthFirstOrders(instance);
  EvaluationResult evaluated =
      EvaluateSchedule(instance, result.split, result.machine_orders);
  if (!evaluated.schedule.ok()) {
    throw Error("failed to construct uniform SlackPipe incumbent");
  }
  result.makespan_ticks = evaluated.schedule.makespan;
  result.schedule = std::move(evaluated.schedule);
  result.wall_time_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  result.timing.incumbent_seconds = result.wall_time_seconds;
  result.timing.total_seconds = result.wall_time_seconds;
  return result;
}

[[nodiscard]] BfsSplitOptimizationResult BuildSlackPipeReferenceIncumbent(
    const Instance& instance, const SlackPipeOptions& options,
    const Deadline& deadline, double reference_time_limit_seconds) {
  if (options.time_limit_seconds > 0.0 && reference_time_limit_seconds <= 0.0) {
    BfsSplitOptimizationResult fallback = UniformBfsIncumbent(instance);
    fallback.method = "uniform-reference-deadline-fallback";
    fallback.diagnostic =
        "SlackPipe reference solve skipped because the global deadline was "
        "exhausted";
    return fallback;
  }
  if (options.bfs_method == "heuristic") {
    auto heuristic = options.heuristic_options;
    heuristic.random_seed = options.random_seed;
    heuristic.parent_deadline = &deadline;
    const auto balance = ComputeWorkerBalanceConstraint(instance,
        options.worker_balance_tolerance_percent, options.worker_balance_tolerance_layers);
    const auto caller_allowed = heuristic.allowed_partition;
    heuristic.allowed_partition = [&, balance, caller_allowed](const auto& p) {
      if (caller_allowed && !caller_allowed(p)) return false;
      if (balance.enabled)
        for (Tick load : WorkerLayerTotals(instance, p))
          if (load < balance.lower_bound || load > balance.upper_bound) return false;
      return true;
    };
    if (options.time_limit_seconds > 0)
      heuristic.time_limit_seconds = std::min(heuristic.time_limit_seconds, reference_time_limit_seconds);
    return BuildHeuristicIncumbent(instance, heuristic);
  }
  const bool timed_auto_hybrid =
      options.time_limit_seconds > 0.0 && options.bfs_method == "auto";
  const bool explicit_hybrid = options.bfs_method == "hybrid-slack";
  if (timed_auto_hybrid || explicit_hybrid) {
    HybridSlackIncumbentOptions hybrid_options;
    hybrid_options.time_limit_seconds =
        options.time_limit_seconds > 0.0
            ? std::max(reference_time_limit_seconds,
                       std::numeric_limits<double>::min())
            : 0.0;
    BfsSplitOptimizationResult result =
        BuildHybridSlackIncumbent(instance, hybrid_options);
    if (timed_auto_hybrid) result.method = "auto-hybrid-slack";
    return result;
  }
  if (options.bfs_method == "uniform") {
    return UniformBfsIncumbent(instance);
  }
  if (options.time_limit_seconds > 0.0 && options.bfs_method == "enumerate") {
    throw Error(
        "timed explicit SlackPipe BFS reference enumeration is not "
        "deadline-aware; use --bfs-method auto, --bfs-method hybrid-slack, "
        "--bfs-method cpsat, or omit the time limit");
  }

  BfsSplitOptimizerOptions bfs_options;
  bfs_options.enumeration_threshold =
      options.time_limit_seconds > 0.0 && options.bfs_method == "auto"
          ? 0
          : options.enumeration_threshold;
  bfs_options.time_limit_seconds = reference_time_limit_seconds;
  bfs_options.num_workers = options.num_workers;
  bfs_options.random_seed = options.random_seed;
  bfs_options.require_optimal = false;
  bfs_options.log_search_progress = options.log_search_progress;
  bfs_options.search_stats = options.search_stats;
  bfs_options.activation_options = options.activation_options;
  if (options.lifecycle) {
    bfs_options.lifecycle = [&](LifecycleEvent event) {
      event.algorithm = "slackpipe";
      event.configured_solver_limit_seconds = options.time_limit_seconds;
      event.requested_solver_limit_seconds = reference_time_limit_seconds;
      event.remaining_global_time_seconds =
          deadline.remaining_seconds_for_reporting();
      event.phase_specific_cap_seconds = reference_time_limit_seconds;
      options.lifecycle(event);
    };
  }

  BfsSplitOptimizationResult result;
  if (options.bfs_method == "auto") {
    result = OptimizeBfsSplitAuto(instance, bfs_options);
  } else if (options.bfs_method == "enumerate") {
    result = OptimizeBfsSplitEnumerate(instance, bfs_options);
  } else if (options.bfs_method == "cpsat") {
    result = OptimizeBfsSplitCpSat(instance, bfs_options);
  } else {
    throw Error("unknown BFS incumbent method: " + options.bfs_method);
  }
  auto uniform_fallback_after_attempt = [&](const std::string& reason) {
    BfsSplitOptimizationResult fallback = UniformBfsIncumbent(instance);
    fallback.method = result.method + "-uniform-fallback";
    fallback.status = result.status;
    fallback.best_bound_ticks = result.best_bound_ticks;
    fallback.solver_objective_ticks = result.solver_objective_ticks;
    fallback.wall_time_seconds = result.wall_time_seconds;
    fallback.timing = result.timing;
    fallback.branches = result.branches;
    fallback.conflicts = result.conflicts;
    fallback.proven_optimal = false;
    fallback.diagnostic = reason;
    return fallback;
  };
  if (!result.schedule.ok()) {
    return uniform_fallback_after_attempt(
        "SlackPipe reference optimizer did not produce a replayable schedule; "
        "using uniform fallback");
  }
  EvaluationResult checked =
      EvaluateSchedule(instance, result.split, result.machine_orders);
  if (!checked.schedule.ok() ||
      checked.schedule.makespan != result.makespan_ticks) {
    return uniform_fallback_after_attempt(
        "SlackPipe reference optimizer did not replay deterministically; "
        "using uniform fallback");
  }
  result.schedule = std::move(checked.schedule);
  return result;
}

[[nodiscard]] Index BfsCpSatModelsSolved(
    const BfsSplitOptimizationResult& result) {
  return result.method.find("cpsat") == std::string::npos ? 0 : 1;
}

[[nodiscard]] std::string EffectiveBfsMethodName(const std::string& method) {
  if (method == "auto-hybrid-slack") return "hybrid-slack";
  if (method == "auto-cpsat") return "cpsat";
  if (method == "auto-enumerate") return "enumerate";
  if (method.find("deadline-fallback") != std::string::npos) {
    return "uniform";
  }
  return method;
}

}  // namespace

SlackPipeResult SolveCanonicalSlackPipe(const Instance& instance,
                                        const SlackPipeOptions& options) {
  const auto started = Clock::now();
  const Deadline deadline(options.time_limit_seconds);
  instance.Validate();
  ValidateSlackPipeOptions(options);
  const double reference_budget_seconds =
      options.time_limit_seconds > 0.0
          ? deadline.clamp_solver_limit(options.time_limit_seconds *
                                        kReferenceBudgetFraction)
          : 0.0;
  const auto reference_started = Clock::now();
  BfsSplitOptimizationResult reference = BuildSlackPipeReferenceIncumbent(
      instance, options, deadline, reference_budget_seconds);
  const double reference_solve_seconds =
      std::chrono::duration<double>(Clock::now() - reference_started).count();
  PartitionRestriction restriction =
      MakePartitionRestriction(instance, reference.split, options);
  const double joint_remaining_budget_before_joint =
      deadline.remaining_seconds_for_reporting();
  const double joint_remaining_budget_seconds =
      options.time_limit_seconds > 0.0 ? deadline.clamp_solver_limit(0.0) : 0.0;

  JointOptimizerOptions joint_options;
  joint_options.time_limit_seconds =
      options.time_limit_seconds > 0.0 ? joint_remaining_budget_seconds : 0.0;
  joint_options.num_workers = options.num_workers;
  joint_options.random_seed = options.random_seed;
  joint_options.require_optimal = options.require_optimal;
  joint_options.log_search_progress = options.log_search_progress;
  joint_options.fifo_ordering = options.fifo_ordering;
  joint_options.symmetry_break_f0_fifo = options.symmetry_break_f0_fifo;
  joint_options.bfs_method = options.bfs_method;
  joint_options.heuristic_options = options.heuristic_options;
  joint_options.enumeration_threshold = options.enumeration_threshold;
  joint_options.worker_balance_tolerance_percent =
      options.worker_balance_tolerance_percent;
  joint_options.worker_balance_tolerance_layers =
      options.worker_balance_tolerance_layers;
  joint_options.pressure_pruning = options.pressure_pruning;
  joint_options.partition_restriction = restriction;
  joint_options.bfs_incumbent_override = reference;
  joint_options.activation_options = options.activation_options;
  joint_options.use_bfs_hints = options.use_bfs_hints;
  joint_options.search_stats = options.search_stats;
  joint_options.lifecycle = options.lifecycle;

  JointOptimizationResult joint;
  if (options.time_limit_seconds > 0.0 && deadline.expired()) {
    const bool reference_cap_feasible =
        !options.activation_options.enforce_activation_cap ||
        ActivationScheduleSatisfiesCap(instance, reference.schedule,
                                       options.activation_options);
    joint.split = reference.split;
    joint.makespan_ticks = reference.makespan_ticks;
    joint.solver_objective_ticks =
        static_cast<double>(reference.makespan_ticks);
    joint.best_bound_ticks = 0.0;
    joint.status = reference.schedule.ok() && reference_cap_feasible
                       ? "FEASIBLE"
                       : "NOT_RUN";
    joint.joint_status = "NOT_RUN";
    joint.proven_optimal = false;
    joint.wall_time_seconds = deadline.elapsed_seconds();
    joint.timing.total_seconds = joint.wall_time_seconds;
    joint.incumbent_method_requested = options.bfs_method;
    joint.incumbent_method_effective = EffectiveBfsMethodName(reference.method);
    joint.bfs_incumbent_method_requested = joint.incumbent_method_requested;
    joint.bfs_incumbent_method_effective = joint.incumbent_method_effective;
    joint.incumbent_source = "slackpipe_reference";
    joint.incumbent_feasible =
        reference.schedule.ok() && reference_cap_feasible;
    joint.incumbent_primary_objective = reference.makespan_ticks;
    const HybridSlackScores scores =
        ComputeHybridSlackScores(instance, reference.split);
    joint.incumbent_hybrid_min_slack = reference.hybrid_min_slack > 0.0
                                           ? reference.hybrid_min_slack
                                           : scores.partition_min_slack;
    joint.incumbent_hybrid_stage_scores = reference.hybrid_stage_scores.empty()
                                              ? scores.stage_scores
                                              : reference.hybrid_stage_scores;
    joint.incumbent_hybrid_bottleneck_stages =
        reference.hybrid_bottleneck_stages.empty()
            ? scores.bottleneck_stages
            : reference.hybrid_bottleneck_stages;
    joint.horizon_source = reference.schedule.ok() && reference_cap_feasible
                               ? "slackpipe_reference"
                               : "none";
    joint.hints_requested = options.use_bfs_hints;
    joint.hints_effective = false;
    joint.hint_source = "none";
    joint.hint_scope = "none";
    joint.fallback_available =
        reference.schedule.ok() && reference_cap_feasible;
    joint.fallback_source = reference.schedule.ok() && reference_cap_feasible
                                ? "slackpipe_reference"
                                : "none";
    joint.solution_source = reference.schedule.ok() && reference_cap_feasible
                                ? "bfs_incumbent_fallback"
                                : "none";
    joint.fallback_used = reference.schedule.ok() && reference_cap_feasible;
    joint.fifo_ordering_requested = options.fifo_ordering;
    joint.fifo_ordering_effective = options.fifo_ordering;
    joint.fifo_constraint_count =
        options.fifo_ordering
            ? (instance.microbatches - 1) * (2 * instance.stages) +
                  (options.symmetry_break_f0_fifo ? instance.microbatches - 1
                                                  : 0)
            : 0;
    joint.symmetry_break_f0_fifo =
        options.fifo_ordering && options.symmetry_break_f0_fifo;
    joint.num_workers = options.num_workers;
    if (reference.schedule.ok() && reference_cap_feasible) {
      joint.schedule = reference.schedule;
      joint.machine_orders = reference.machine_orders;
      joint.machine_predecessors = ExtractMachinePredecessors(
          instance, joint.machine_orders, options.fifo_ordering);
    }
    joint.bfs_incumbent = reference;
    joint.time_to_first_feasible_seconds =
        reference.schedule.ok() && reference_cap_feasible
            ? reference_solve_seconds
            : 0.0;
    joint.time_to_best_incumbent_seconds =
        reference.schedule.ok() && reference_cap_feasible
            ? reference_solve_seconds
            : 0.0;
    joint.first_feasible_objective =
        reference.schedule.ok() && reference_cap_feasible
            ? reference.makespan_ticks
            : 0;
    joint.worker_balance_constraint = ComputeWorkerBalanceConstraint(
        instance, options.worker_balance_tolerance_percent,
        options.worker_balance_tolerance_layers);
    joint.search_stats_enabled = options.search_stats != nullptr;
    joint.activation_cap_constraints.model_support_level =
        ToString(ActivationCapSolverSupport());
    joint.activation_cap_constraints.solver_supported =
        ActivationCapSolverCanEnforce(
            options.activation_options,
            !SlackPipeModeFixesFullPartition(options.split_mode));
    joint.activation_cap_constraints.incumbent_rejected_for_activation_cap =
        options.activation_options.enforce_activation_cap &&
        !reference_cap_feasible;
    joint.diagnostic = "global_deadline_expired_before_joint_cp_sat";
    if (joint.activation_cap_constraints
            .incumbent_rejected_for_activation_cap) {
      joint.diagnostic += "; no cap-feasible SlackPipe reference is available";
    }
    if (options.require_optimal && !joint.proven_optimal) {
      joint.status += "_REJECTED_REQUIRE_OPTIMAL";
    }
    if (options.search_stats != nullptr) {
      options.search_stats->cp_sat_available = true;
      options.search_stats->cp_sat_status = "NOT_RUN";
      joint.search_stats = *options.search_stats;
    }
  } else {
    joint = OptimizeJointSplitAndScheduleCpSat(instance, joint_options);
  }

  SlackPipeResult result;
  result.algorithm = "slackpipe";
  result.initial_split_method = "optimized-bfs";
  result.split_mode = ToString(options.split_mode);
  result.effective_split_mode = ToString(restriction.mode);
  result.move_budget = options.move_budget;
  result.per_stage_delta = options.per_stage_delta;
  result.worker_move_budget = options.worker_move_budget;
  result.per_worker_delta = options.per_worker_delta;
  result.analytical_global_lower_bound = AnalyticalGlobalLowerBound(instance);
  result.bfs = std::move(joint.bfs_incumbent);
  if (result.bfs.split.empty()) {
    result.bfs = std::move(reference);
  }
  if (result.bfs.method == "heuristic") {
    result.initial_split_method = result.bfs.method;
  } else if (result.bfs.method.find("hybrid-slack") != std::string::npos) {
    result.initial_split_method = "hybrid-slack";
  } else if (result.bfs.method.find("uniform") != std::string::npos) {
    result.initial_split_method = "uniform";
  }
  result.initial_uniform_split = UniformSplit(instance);
  result.initial_uniform_bfs_makespan =
      EvaluateSchedule(instance, result.initial_uniform_split,
                       BreadthFirstOrders(instance))
          .schedule.makespan;
  result.baseline_worker_layers = WorkerLayerTotals(instance, result.bfs.split);

  result.split = std::move(joint.split);
  result.makespan_ticks = joint.makespan_ticks;
  result.solver_objective_ticks = joint.solver_objective_ticks;
  result.best_bound_ticks = joint.best_bound_ticks;
  result.status = std::move(joint.status);
  result.joint_status = std::move(joint.joint_status);
  result.proven_optimal = joint.proven_optimal;
  result.wall_time_seconds = joint.wall_time_seconds;
  result.timing = joint.timing;
  result.timing.incumbent_seconds = reference_solve_seconds;
  result.deterministic_time = joint.deterministic_time;
  result.branches = joint.branches;
  result.conflicts = joint.conflicts;
  result.time_to_first_feasible_seconds = joint.time_to_first_feasible_seconds;
  result.time_to_best_incumbent_seconds = joint.time_to_best_incumbent_seconds;
  result.first_feasible_objective = joint.first_feasible_objective;
  result.incumbent_improvement_count = joint.incumbent_improvement_count;
  result.incumbent_trace = std::move(joint.incumbent_trace);
  result.reference_budget_seconds = reference_budget_seconds;
  result.reference_solve_seconds = reference_solve_seconds;
  result.joint_remaining_budget_seconds =
      options.time_limit_seconds > 0.0 ? joint_remaining_budget_before_joint
                                       : 0.0;
  result.joint_incumbent_method_requested = joint.incumbent_method_requested;
  result.joint_incumbent_method_effective = joint.incumbent_method_effective;
  result.joint_bfs_incumbent_method_requested =
      joint.bfs_incumbent_method_requested;
  result.joint_bfs_incumbent_method_effective =
      joint.bfs_incumbent_method_effective;
  result.joint_incumbent_source = joint.incumbent_source;
  result.joint_incumbent_feasible = joint.incumbent_feasible;
  result.joint_incumbent_primary_objective = joint.incumbent_primary_objective;
  result.joint_incumbent_hybrid_min_slack = joint.incumbent_hybrid_min_slack;
  result.joint_incumbent_baseline_primary_objective =
      joint.incumbent_baseline_primary_objective;
  result.joint_incumbent_baseline_hybrid_min_slack =
      joint.incumbent_baseline_hybrid_min_slack;
  result.joint_incumbent_improved_over_baseline =
      joint.incumbent_improved_over_baseline;
  result.joint_incumbent_hybrid_stage_scores =
      joint.incumbent_hybrid_stage_scores;
  result.joint_incumbent_hybrid_bottleneck_stages =
      joint.incumbent_hybrid_bottleneck_stages;
  result.joint_horizon_source = joint.horizon_source;
  result.joint_hint_budget_seconds = joint.hint_budget_seconds;
  result.joint_hint_elapsed_seconds = joint.hint_elapsed_seconds;
  result.joint_hint_iterations = joint.hint_iterations;
  result.joint_hint_candidates_generated = joint.hint_candidates_generated;
  result.joint_hint_candidates_simulated = joint.hint_candidates_simulated;
  result.joint_hint_partition_moves_accepted =
      joint.hint_partition_moves_accepted;
  result.joint_hint_interleaving_moves_accepted =
      joint.hint_interleaving_moves_accepted;
  result.joint_hint_deadline_reached = joint.hint_deadline_reached;
  result.joint_hint_termination_reason = joint.hint_termination_reason;
  result.joint_hints_requested = joint.hints_requested;
  result.joint_hints_effective = joint.hints_effective;
  result.joint_hint_source = joint.hint_source;
  result.joint_hint_scope = joint.hint_scope;
  result.joint_hint_complete_for_basic_model =
      joint.hint_complete_for_basic_model;
  result.joint_hint_complete_for_full_model =
      joint.hint_complete_for_full_model;
  result.joint_hinted_layer_variable_count = joint.hinted_layer_variable_count;
  result.joint_hinted_operation_variable_count =
      joint.hinted_operation_variable_count;
  result.joint_hinted_scalar_variable_count =
      joint.hinted_scalar_variable_count;
  result.joint_hinted_auxiliary_variable_count =
      joint.hinted_auxiliary_variable_count;
  result.joint_hinted_total_variable_count = joint.hinted_total_variable_count;
  result.joint_auxiliary_variable_count = joint.auxiliary_variable_count;
  result.joint_fifo_ordering_requested = joint.fifo_ordering_requested;
  result.joint_fifo_ordering_effective = joint.fifo_ordering_effective;
  result.joint_fifo_constraint_count = joint.fifo_constraint_count;
  result.joint_fallback_available = joint.fallback_available;
  result.joint_fallback_source = joint.fallback_source;
  result.joint_solution_source = joint.solution_source;
  result.joint_fallback_used = joint.fallback_used;
  result.schedule = std::move(joint.schedule);
  result.machine_orders = std::move(joint.machine_orders);
  result.machine_predecessors = std::move(joint.machine_predecessors);
  result.cp_sat_models_solved =
      BfsCpSatModelsSolved(result.bfs) + joint.cp_sat_models_solved;
  result.proven_global_optimal =
      result.makespan_ticks > 0 &&
      result.makespan_ticks == result.analytical_global_lower_bound;
  result.global_certificate =
      result.proven_global_optimal ? "analytical_bound_closure" : "";
  result.diagnostic = std::move(joint.diagnostic);
  result.worker_balance_constraint = joint.worker_balance_constraint;
  result.pressure_pruning_stats = joint.pressure_pruning_stats;
  result.activation_cap_constraints = joint.activation_cap_constraints;
  result.search_stats_enabled = options.search_stats != nullptr;
  if (options.search_stats != nullptr) {
    options.search_stats->algorithm = "slackpipe";
    result.search_stats = *options.search_stats;
  }

  if (!result.split.empty()) {
    ValidateSplit(instance, result.split);
    result.mode_validation_passed = SplitSatisfiesSlackPipeMode(
        instance, result.split, result.bfs.split, options);
    if (!result.mode_validation_passed) {
      throw Error(PartitionRestrictionViolationMessage(instance, result.split,
                                                       restriction));
    }
    result.final_worker_layers = WorkerLayerTotals(instance, result.split);
    result.worker_layer_differences = WorkerLayerDifferences(
        result.final_worker_layers, result.baseline_worker_layers);
    result.worker_balance_l1 = WorkerBalanceL1(result.worker_layer_differences);
    result.worker_balance_max_deviation =
        WorkerBalanceMaxDeviation(result.worker_layer_differences);
  }
  result.total_wall_time_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  result.wall_time_seconds = result.total_wall_time_seconds;
  result.timing.total_seconds = result.total_wall_time_seconds;
  return result;
}

bool IsSlackPipeSolverAvailable() { return true; }

}  // namespace slackpipe

#endif
