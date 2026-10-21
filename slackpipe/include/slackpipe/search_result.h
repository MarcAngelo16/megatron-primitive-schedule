#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/schedule.h"
#include "slackpipe/timing.h"

namespace slackpipe {

// Method-neutral outcome. Solver status/objective and deterministic replay are
// separate: no schedule is fabricated when the solver has no feasible solution.
struct SearchResult {
  std::string method;
  std::string requested_command;
  int random_seed = 1;
  std::string status;
  std::string solver_status_raw;
  std::string optimality_proof_source = "none";
  bool proven_optimal = false;
  std::vector<Tick> split;
  ScheduleSolution schedule;
  MachineOrders machine_orders;
  Tick makespan_ticks = 0;
  double solver_objective_ticks = 0;
  double best_bound_ticks = 0;
  bool solver_bound_available = false;
  bool cp_sat_launched = false;
  bool final_solution_available = false;
  std::string solution_source = "none";
  std::string no_solution_reason;
  double requested_seconds = 0;
  double wall_time_seconds = 0;
  int num_workers = 1;
  SolverPhaseTiming timing;
  Index branches = 0, conflicts = 0, cp_sat_models_solved = 0;
  double deterministic_time = 0;
  Tick conservative_horizon = 0;
  Index workload_constraint_count = 0, fifo_constraint_count = 0;
  Index external_hint_count = 0;
  Index replay_cap_rejections = 0;
  double time_to_first_feasible_seconds = 0, time_to_best_incumbent_seconds = 0;
  Tick first_feasible_objective = 0;
  Index incumbent_improvement_count = 0;
  std::vector<std::pair<double, Tick>> incumbent_trace;
  Tick baseline_primary_objective = 0;
  bool improved_over_baseline = false;
  Index evaluations = 0, candidates_generated = 0, iterations = 0;
  Index partition_moves_accepted = 0, order_moves_accepted = 0;
  bool deadline_reached = false;
  std::string termination_reason;
  std::string heuristic_metadata_json;
  std::string method_diagnostics_json;
  std::string diagnostic;
  ActivationCapConstraintMetadata activation_cap_constraints;
  std::optional<ActivationAnalysisResult> activation_analysis;
};

}  // namespace slackpipe
