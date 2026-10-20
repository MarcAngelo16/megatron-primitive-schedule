#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "slackpipe/lifecycle.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/schedule.h"
#include "slackpipe/search_stats.h"
#include "slackpipe/timing.h"

namespace slackpipe {

struct BfsSplitOptimizerOptions {
  std::uint64_t enumeration_threshold = 100000;
  double time_limit_seconds = 0.0;
  int num_workers = 1;
  int random_seed = 1;
  bool require_optimal = true;
  bool log_search_progress = false;
  std::string fixed_order_partition_backend = "auto";
  std::optional<std::vector<Tick>> fixed_order_partition_incumbent;
  ActivationAnalysisOptions activation_options;
  ActivationCapModelDebugDump *activation_cap_model_debug_dump = nullptr;
  SearchStats *search_stats = nullptr;
  LifecycleCallback lifecycle;
};

struct BfsSplitOptimizationResult {
  std::vector<Tick> split;
  Tick makespan_ticks = 0;
  std::string status;
  Tick best_bound_ticks = 0;
  double solver_objective_ticks = 0.0;
  double wall_time_seconds = 0.0;
  SolverPhaseTiming timing;
  Index branches = 0;
  Index conflicts = 0;
  std::string method;
  std::uint64_t checked_splits = 0;
  std::uint64_t enumeration_candidates_total = 0;
  std::uint64_t enumeration_candidates_valid_schedule = 0;
  std::uint64_t enumeration_candidates_cap_feasible = 0;
  std::uint64_t enumeration_candidates_cap_rejected = 0;
  bool enumeration_proved_optimal = false;
  std::string optimality_proof_source = "none";
  std::string solver_status_raw;
  bool proven_optimal = false;
  bool fallback_available = false;
  bool fallback_used = false;
  std::string fallback_reason;
  std::string fallback_source = "none";
  std::string solution_source = "none";
  std::string returned_solution_source = "none";
  std::string fixed_order_partition_backend_requested;
  std::string fixed_order_partition_backend_effective;
  std::uint64_t estimated_partition_count = 0;
  bool estimated_partition_count_available = false;
  std::uint64_t enumeration_safety_threshold = 0;
  bool cp_sat_launched = false;
  Index cp_sat_models_solved = 0;
  double deterministic_time = 0.0;
  double time_to_first_feasible_seconds = 0.0;
  double time_to_best_incumbent_seconds = 0.0;
  Tick first_feasible_objective = 0;
  Index incumbent_improvement_count = 0;
  std::vector<std::pair<double, Tick>> incumbent_trace;
  ScheduleSolution schedule;
  MachineOrders machine_orders;
  std::string diagnostic;
  // Versioned diagnostics from the optional Heuristic Algorithm only.
  std::string heuristic_metadata_json;
  ActivationCapConstraintMetadata activation_cap_constraints;
  double hybrid_min_slack = 0.0;
  std::vector<double> hybrid_stage_scores;
  std::vector<Index> hybrid_bottleneck_stages;
  Tick baseline_primary_objective = 0;
  double baseline_hybrid_min_slack = 0.0;
  bool improved_over_baseline = false;
  double hint_budget_seconds = 0.0;
  double hint_elapsed_seconds = 0.0;
  Index hint_iterations = 0;
  Index hint_candidates_generated = 0;
  Index hint_candidates_simulated = 0;
  Index hint_partition_moves_accepted = 0;
  Index hint_interleaving_moves_accepted = 0;
  bool hint_deadline_reached = false;
  std::string hint_termination_reason;
  bool search_stats_enabled = false;
  SearchStats search_stats;
  std::optional<CanonicalResultMetadata> canonical;
};

using SplitVisitor = std::function<void(const std::vector<Tick> &)>;

[[nodiscard]] std::uint64_t CountValidSplitsCapped(const Instance &instance,
                                                   std::uint64_t cap);
void EnumerateValidSplits(const Instance &instance,
                          const SplitVisitor &visitor);

[[nodiscard]] BfsSplitOptimizationResult OptimizeBfsSplitEnumerate(
    const Instance &instance, const BfsSplitOptimizerOptions &options = {});
[[nodiscard]] BfsSplitOptimizationResult OptimizeBfsSplitCpSat(
    const Instance &instance, const BfsSplitOptimizerOptions &options = {});
[[nodiscard]] BfsSplitOptimizationResult OptimizeBfsSplitAuto(
    const Instance &instance, const BfsSplitOptimizerOptions &options = {});

void ApplyBfsCanonicalEvaluationPolicy(const Instance &instance,
                                       BfsSplitOptimizationResult &result,
                                       Tick raw_cp_sat_objective,
                                       SearchStats *search_stats = nullptr);

[[nodiscard]] bool IsCpSatBfsSplitOptimizerAvailable();

}  // namespace slackpipe
