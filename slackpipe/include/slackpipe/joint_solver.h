#pragma once

#include <string>
#include <optional>
#include <utility>
#include <vector>

#include "slackpipe/bfs_solver.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/lifecycle.h"
#include "slackpipe/partition_restriction.h"
#include "slackpipe/pressure_pruning.h"
#include "slackpipe/heuristic_hint.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/timing.h"
#include "slackpipe/worker_balance.h"

namespace slackpipe {

struct JointOptimizerOptions {
  double time_limit_seconds = 0.0;
  int num_workers = 1;
  int random_seed = 1;
  bool require_optimal = true;
  bool log_search_progress = false;
  bool fifo_ordering = true;
  bool symmetry_break_f0_fifo = true;
  std::string bfs_method = "auto";
  HeuristicHintOptions heuristic_options;
  std::uint64_t enumeration_threshold = 100000;
  bool use_bfs_hints = true;
  std::optional<bool> worker_balance_pruning;
  std::optional<std::string> incumbent_method;
  bool incumbent_bound = true;
  std::optional<bool> incumbent_hints;
  bool incumbent_fallback = true;
  double worker_balance_tolerance_percent = -1.0;
  std::optional<Index> worker_balance_tolerance_layers;
  PressurePruningOptions pressure_pruning;
  std::optional<PartitionRestriction> partition_restriction;
  std::optional<BfsSplitOptimizationResult> bfs_incumbent_override;
  ActivationAnalysisOptions activation_options;
  SearchStats* search_stats = nullptr;
  LifecycleCallback lifecycle;
};

struct JointOptimizationResult {
  std::vector<Tick> split;
  Tick makespan_ticks = 0;
  double solver_objective_ticks = 0.0;
  double best_bound_ticks = 0.0;
  Index workload_constraint_count = 0;  // W worker bounds + one capacity bound when built.
  std::string status;
  bool proven_optimal = false;
  double wall_time_seconds = 0.0;
  SolverPhaseTiming timing;
  double incumbent_budget_seconds = 0.0;
  double incumbent_model_build_seconds = 0.0;
  double incumbent_solve_seconds = 0.0;
  std::string incumbent_status;
  double joint_budget_seconds = 0.0;
  double joint_model_build_seconds = 0.0;
  double joint_solve_seconds = 0.0;
  std::string joint_status;
  std::string incumbent_method_requested;
  std::string incumbent_method_effective;
  std::string bfs_incumbent_method_requested;
  std::string bfs_incumbent_method_effective;
  std::string incumbent_source = "none";
  bool incumbent_feasible = false;
  Tick incumbent_primary_objective = 0;
  double incumbent_hybrid_min_slack = 0.0;
  Tick incumbent_baseline_primary_objective = 0;
  double incumbent_baseline_hybrid_min_slack = 0.0;
  bool incumbent_improved_over_baseline = false;
  std::vector<double> incumbent_hybrid_stage_scores;
  std::vector<Index> incumbent_hybrid_bottleneck_stages;
  std::string horizon_source = "none";
  double hint_budget_seconds = 0.0;
  double hint_elapsed_seconds = 0.0;
  Index hint_iterations = 0;
  Index hint_candidates_generated = 0;
  Index hint_candidates_simulated = 0;
  Index hint_partition_moves_accepted = 0;
  Index hint_interleaving_moves_accepted = 0;
  bool hint_deadline_reached = false;
  std::string hint_termination_reason;
  bool hints_requested = false;
  bool hints_effective = false;
  std::string hint_source = "none";
  std::string hint_scope = "none";
  bool hint_complete_for_basic_model = false;
  bool hint_complete_for_full_model = false;
  Index hinted_layer_variable_count = 0;
  Index hinted_operation_variable_count = 0;
  Index hinted_scalar_variable_count = 0;
  Index hinted_auxiliary_variable_count = 0;
  Index hinted_total_variable_count = 0;
  Index auxiliary_variable_count = 0;
  bool worker_balance_pruning_requested = false;
  bool worker_balance_pruning_effective = false;
  bool raw_bound_globally_valid = true;
  std::optional<double> worker_balance_tolerance_requested_percent;
  std::optional<Index> worker_balance_tolerance_requested_layers;
  std::string incumbent_method_requested_normalized = "slack";
  bool incumbent_bound_requested = true;
  bool incumbent_bound_effective = false;
  Tick incumbent_bound_horizon = 0;
  bool incumbent_hints_requested = true;
  bool incumbent_hints_effective = false;
  bool incumbent_found = false;
  bool incumbent_valid = false;
  Tick incumbent_makespan = 0;
  bool fallback_enabled = true;
  bool solver_solution_available = false;
  bool final_solution_available = false;
  double time_to_first_cpsat_feasible_seconds = 0.0;
  Tick first_cpsat_feasible_objective = 0;
  std::string final_solution_source = "none";
  std::string no_solution_reason;
  double global_time_limit_seconds = 0.0;
  bool fallback_available = false;
  std::string fallback_source = "none";
  std::string solution_source;
  bool fallback_used = false;
  Index cp_sat_models_solved = 0;
  double deterministic_time = 0.0;
  Index branches = 0;
  Index conflicts = 0;
  double time_to_first_feasible_seconds = 0.0;
  double time_to_best_incumbent_seconds = 0.0;
  Tick first_feasible_objective = 0;
  Index incumbent_improvement_count = 0;
  std::vector<std::pair<double, Tick>> incumbent_trace;
  bool fifo_ordering_requested = true;
  bool fifo_ordering_effective = true;
  Index fifo_constraint_count = 0;
  bool symmetry_break_f0_fifo = true;
  int num_workers = 1;
  ScheduleSolution schedule;
  MachineOrders machine_orders;
  MachinePredecessors machine_predecessors;
  BfsSplitOptimizationResult bfs_incumbent;
  std::string diagnostic;
  WorkerBalanceConstraintResult worker_balance_constraint;
  PressurePruningStats pressure_pruning_stats;
  ActivationCapConstraintMetadata activation_cap_constraints;
  bool search_stats_enabled = false;
  SearchStats search_stats;
  std::optional<CanonicalResultMetadata> canonical;
};

[[nodiscard]] JointOptimizationResult OptimizeJointSplitAndScheduleCpSat(
    const Instance& instance, const JointOptimizerOptions& options = {});
[[nodiscard]] JointOptimizationResult OptimizeScheduleForFixedSplitCpSat(
    const Instance& instance, const std::vector<Tick>& fixed_split,
    const JointOptimizerOptions& options = {});
[[nodiscard]] JointOptimizationResult
BuildScheduleOnlyFixedSplitDeadlineFallback(
    const Instance& instance, const std::vector<Tick>& fixed_split,
    const JointOptimizerOptions& options, double elapsed_seconds,
    const std::string& reason);
[[nodiscard]] std::string RequestedIncumbentMethodName(
    const JointOptimizerOptions& options);
[[nodiscard]] bool IncumbentHintsRequested(
    const JointOptimizerOptions& options);
[[nodiscard]] bool IncumbentFallbackRequested(
    const JointOptimizerOptions& options);
[[nodiscard]] bool WorkerBalancePruningRequested(
    const JointOptimizerOptions& options);
[[nodiscard]] bool WorkerBalanceToleranceProvided(
    const JointOptimizerOptions& options);
[[nodiscard]] WorkerBalanceConstraintResult ComputeJointWorkerBalanceConstraint(
    const Instance& instance, const JointOptimizerOptions& options);
void ValidateJointOptimizerMechanismOptions(
    const JointOptimizerOptions& options);
[[nodiscard]] bool CpSatPartitionRestrictionAcceptsSplitForTesting(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction);

[[nodiscard]] bool IsJointOptimizerAvailable();

}  // namespace slackpipe
