#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "slackpipe/bfs_solver.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/partition_restriction.h"
#include "slackpipe/heuristic_hint.h"
#include "slackpipe/search_stats.h"
#include "slackpipe/timing.h"
#include "slackpipe/lifecycle.h"
#include "slackpipe/worker_balance.h"
#include "slackpipe/pressure_pruning.h"
#include "slackpipe/result_schema.h"

namespace slackpipe {

struct SlackPipeOptions {
  SlackPipeSplitMode split_mode = SlackPipeSplitMode::kGlobal;
  Index move_budget = 2;
  bool move_budget_provided = false;
  std::optional<Index> per_stage_delta;
  Index worker_move_budget = 2;
  bool worker_move_budget_provided = false;
  std::optional<Index> per_worker_delta;
  double time_limit_seconds = 0.0;
  int num_workers = 1;
  int random_seed = 1;
  bool require_optimal = true;
  bool log_search_progress = false;
  bool fifo_ordering = true;
  bool symmetry_break_f0_fifo = true;
  std::string bfs_method = "auto";
  HeuristicHintOptions heuristic_options;
  bool use_bfs_hints = true;
  std::uint64_t enumeration_threshold = 100000;
  double worker_balance_tolerance_percent = -1.0;
  std::optional<Index> worker_balance_tolerance_layers;
  PressurePruningOptions pressure_pruning;
  ActivationAnalysisOptions activation_options;
  SearchStats* search_stats = nullptr;
  LifecycleCallback lifecycle;
};

struct SlackPipeResult {
  BfsSplitOptimizationResult bfs;
  std::vector<Tick> split;
  Tick makespan_ticks = 0;
  double solver_objective_ticks = 0.0;
  double best_bound_ticks = 0.0;
  std::string status;
  std::string joint_status;
  bool proven_optimal = false;
  double wall_time_seconds = 0.0;
  SolverPhaseTiming timing;
  double deterministic_time = 0.0;
  Index branches = 0;
  Index conflicts = 0;
  double time_to_first_feasible_seconds = 0.0;
  double time_to_best_incumbent_seconds = 0.0;
  Tick first_feasible_objective = 0;
  Index incumbent_improvement_count = 0;
  std::vector<std::pair<double, Tick>> incumbent_trace;
  double reference_budget_seconds = 0.0;
  double reference_solve_seconds = 0.0;
  double joint_remaining_budget_seconds = 0.0;
  std::string joint_incumbent_method_requested;
  std::string joint_incumbent_method_effective;
  std::string joint_bfs_incumbent_method_requested;
  std::string joint_bfs_incumbent_method_effective;
  std::string joint_incumbent_source = "none";
  bool joint_incumbent_feasible = false;
  Tick joint_incumbent_primary_objective = 0;
  double joint_incumbent_hybrid_min_slack = 0.0;
  Tick joint_incumbent_baseline_primary_objective = 0;
  double joint_incumbent_baseline_hybrid_min_slack = 0.0;
  bool joint_incumbent_improved_over_baseline = false;
  std::vector<double> joint_incumbent_hybrid_stage_scores;
  std::vector<Index> joint_incumbent_hybrid_bottleneck_stages;
  std::string joint_horizon_source = "none";
  double joint_hint_budget_seconds = 0.0;
  double joint_hint_elapsed_seconds = 0.0;
  Index joint_hint_iterations = 0;
  Index joint_hint_candidates_generated = 0;
  Index joint_hint_candidates_simulated = 0;
  Index joint_hint_partition_moves_accepted = 0;
  Index joint_hint_interleaving_moves_accepted = 0;
  bool joint_hint_deadline_reached = false;
  std::string joint_hint_termination_reason;
  bool joint_hints_requested = false;
  bool joint_hints_effective = false;
  std::string joint_hint_source = "none";
  std::string joint_hint_scope = "none";
  bool joint_hint_complete_for_basic_model = false;
  bool joint_hint_complete_for_full_model = false;
  Index joint_hinted_layer_variable_count = 0;
  Index joint_hinted_operation_variable_count = 0;
  Index joint_hinted_scalar_variable_count = 0;
  Index joint_hinted_auxiliary_variable_count = 0;
  Index joint_hinted_total_variable_count = 0;
  Index joint_auxiliary_variable_count = 0;
  bool joint_fifo_ordering_requested = true;
  bool joint_fifo_ordering_effective = true;
  Index joint_fifo_constraint_count = 0;
  bool joint_fallback_available = false;
  std::string joint_fallback_source = "none";
  std::string joint_solution_source = "none";
  bool joint_fallback_used = false;
  ScheduleSolution schedule;
  MachineOrders machine_orders;
  MachinePredecessors machine_predecessors;
  std::string algorithm = "slackpipe";
  std::string initial_split_method = "optimized-bfs";
  std::vector<Tick> initial_uniform_split;
  Tick initial_uniform_bfs_makespan = 0;
  Index cp_sat_models_solved = 0;
  std::string split_mode;
  std::string effective_split_mode;
  Index move_budget = 0;
  std::optional<Index> per_stage_delta;
  Index worker_move_budget = 0;
  std::optional<Index> per_worker_delta;
  bool mode_validation_passed = false;
  std::vector<Tick> baseline_worker_layers;
  std::vector<Tick> final_worker_layers;
  std::vector<Tick> worker_layer_differences;
  Tick worker_balance_l1 = 0;
  Tick worker_balance_max_deviation = 0;
  Tick analytical_global_lower_bound = 0;
  double total_wall_time_seconds = 0.0;
  bool proven_global_optimal = false;
  std::string global_certificate;
  std::string diagnostic;
  WorkerBalanceConstraintResult worker_balance_constraint;
  PressurePruningStats pressure_pruning_stats;
  ActivationCapConstraintMetadata activation_cap_constraints;
  bool search_stats_enabled = false;
  SearchStats search_stats;
  std::optional<CanonicalResultMetadata> canonical;
};

[[nodiscard]] std::string ToString(SlackPipeSplitMode mode);
[[nodiscard]] SlackPipeSplitMode ParseSlackPipeSplitMode(
    const std::string& text);
[[nodiscard]] Tick AnalyticalGlobalLowerBound(const Instance& instance);
[[nodiscard]] std::vector<Tick> UniformSplit(const Instance& instance);
[[nodiscard]] std::vector<Tick> LoadBalancedSplit(const Instance& instance);
[[nodiscard]] std::vector<std::vector<Index>> StagesOnWorkers(
    const Instance& instance);
[[nodiscard]] std::vector<Tick> WorkerLayerTotals(
    const Instance& instance, const std::vector<Tick>& split);
[[nodiscard]] std::vector<Tick> WorkerLayerDifferences(
    const std::vector<Tick>& worker_layers,
    const std::vector<Tick>& baseline_worker_layers);
[[nodiscard]] Tick WorkerBalanceL1(const std::vector<Tick>& differences);
[[nodiscard]] Tick WorkerBalanceMaxDeviation(
    const std::vector<Tick>& differences);
void ValidateSlackPipeOptions(const SlackPipeOptions& options);
[[nodiscard]] PartitionRestriction MakePartitionRestriction(
    const Instance& instance, const std::vector<Tick>& reference_split,
    const SlackPipeOptions& options);
void ValidatePartitionRestriction(const Instance& instance,
                                  const PartitionRestriction& restriction);
[[nodiscard]] bool SplitSatisfiesPartitionRestriction(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction);
[[nodiscard]] std::string PartitionRestrictionViolationMessage(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction);
[[nodiscard]] bool SplitSatisfiesSlackPipeMode(
    const Instance& instance, const std::vector<Tick>& split,
    const std::vector<Tick>& baseline_split, const SlackPipeOptions& options);

[[nodiscard]] SlackPipeResult SolveCanonicalSlackPipe(
    const Instance& instance, const SlackPipeOptions& options = {});

[[nodiscard]] bool IsSlackPipeSolverAvailable();

}  // namespace slackpipe
