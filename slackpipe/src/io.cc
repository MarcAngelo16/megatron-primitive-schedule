#include "slackpipe/io.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/evaluation_method.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

#include "slackpipe/result_schema.h"

#include <iomanip>
#include "slackpipe/workload_bound.h"
#include "slackpipe/result_validator.h"
namespace slackpipe {
namespace {
std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (unsigned char ch : text) {
    if (ch == '"' || ch == '\\') out << '\\';
    if (ch < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<unsigned>(ch) << std::dec;
    } else {
      out << ch;
    }
  }
  return out.str();
}

void WriteTickArray(std::ostringstream &out, const std::vector<Tick> &values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << values[i];
  }
  out << "]";
}

void WriteWorkerOrdersJson(std::ostringstream &out, const Instance &instance,
                           const MachineOrders &orders,
                           const std::string &indent, bool trailing_comma) {
  out << indent << "\"worker_local_operation_order\": [\n";
  for (std::size_t w = 0; w < orders.size(); ++w) {
    out << indent << "  [";
    for (std::size_t i = 0; i < orders[w].size(); ++i) {
      if (i != 0) out << ", ";
      out << "\"" << JsonEscape(OperationName(instance, orders[w][i])) << "\"";
    }
    out << "]";
    if (w + 1 != orders.size()) out << ",";
    out << "\n";
  }
  out << indent << "]";
  if (trailing_comma) out << ",";
  out << "\n";
}

void WriteScheduleMetrics(std::ostringstream &out, const Instance &instance,
                          const ScheduleSolution &schedule,
                          const std::string &indent, bool trailing_comma,
                          bool fifo_ordering = true) {
  const ScheduleMetrics metrics =
      ComputeScheduleMetrics(instance, schedule, fifo_ordering);
  out << indent
      << "\"simulated_iteration_time\": " << metrics.simulated_iteration_time
      << ",\n";
  out << indent << "\"total_useful_work\": " << metrics.total_useful_work
      << ",\n";
  out << indent << "\"pipeline_utilization\": " << metrics.pipeline_utilization
      << ",\n";
  out << indent << "\"per_worker_busy_time\": ";
  WriteTickArray(out, metrics.per_worker_busy_time);
  out << ",\n";
  out << indent << "\"per_worker_idle_time\": ";
  WriteTickArray(out, metrics.per_worker_idle_time);
  out << ",\n";
  out << indent << "\"maximum_worker_load\": " << metrics.max_worker_load
      << ",\n";
  out << indent << "\"pipeline_fill_time\": " << metrics.pipeline_fill_time
      << ",\n";
  out << indent << "\"pipeline_drain_time\": " << metrics.pipeline_drain_time
      << ",\n";
  out << indent << "\"communication_blocked_time\": ";
  if (metrics.communication_blocked_time) {
    out << *metrics.communication_blocked_time;
  } else {
    out << "null";
  }
  if (trailing_comma) out << ",";
  out << "\n";
}
}
std::string ToJson(const Instance &instance, const ScheduleSolution &schedule) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"microbatches\": " << instance.microbatches << ",\n";
  out << "  \"stages\": " << instance.stages << ",\n";
  out << "  \"workers\": " << instance.workers << ",\n";
  out << "  \"makespan\": " << schedule.makespan << ",\n";
  out << "  \"ok\": " << (schedule.ok() ? "true" : "false") << ",\n";
  out << "  \"split\": [";
  for (std::size_t i = 0; i < schedule.split.size(); ++i) {
    if (i != 0) out << ", ";
    out << schedule.split[i];
  }
  out << "],\n";
  out << "  \"errors\": [";
  for (std::size_t i = 0; i < schedule.validation_errors.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(schedule.validation_errors[i]) << "\"";
  }
  out << "],\n";
  WriteScheduleMetrics(out, instance, schedule, "  ", true);
  WriteWorkerOrdersJson(out, instance, schedule.orders, "  ", true);
  out << "  \"operations\": [\n";
  for (std::size_t i = 0; i < schedule.operations_by_id.size(); ++i) {
    const auto &op = schedule.operations_by_id[i];
    const OperationView view = DecodeOperation(instance, op.id);
    out << "    {\"id\": " << op.id.value << ", \"name\": \""
        << OperationName(instance, op.id)
        << "\", \"microbatch\": " << view.microbatch
        << ", \"chain_index\": " << view.chain_index
        << ", \"stage\": " << view.stage << ", \"kind\": \""
        << (view.backward ? "B" : "F") << "\", \"worker\": " << op.worker
        << ", \"start\": " << op.start << ", \"end\": " << op.end
        << ", \"duration\": " << op.duration << "}";
    if (i + 1 != schedule.operations_by_id.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::string ToCsv(const Instance &instance, const ScheduleSolution &schedule) {
  std::ostringstream out;
  out << "id,name,microbatch,chain_index,stage,kind,worker,start,end,"
         "duration\n";
  for (const auto &op : schedule.operations_by_id) {
    const OperationView view = DecodeOperation(instance, op.id);
    out << op.id.value << "," << OperationName(instance, op.id) << ","
        << view.microbatch << "," << view.chain_index << "," << view.stage
        << "," << (view.backward ? "B" : "F") << "," << op.worker << ","
        << op.start << "," << op.end << "," << op.duration << "\n";
  }
  return out.str();
}

std::string ToOrdersText(const Instance &instance,
                         const MachineOrders &orders) {
  std::ostringstream out;
  for (std::size_t w = 0; w < orders.size(); ++w) {
    out << "worker " << w << ":";
    for (OperationId id : orders[w]) out << " " << OperationName(instance, id);
    out << "\n";
  }
  return out.str();
}

std::string ToSvg(const Instance &instance, const ScheduleSolution &schedule) {
  const Tick scale_den = std::max<Tick>(1, schedule.makespan);
  const int width = 1000;
  const int row_height = 36;
  const int left = 90;
  const int height = static_cast<int>(instance.workers) * row_height + 40;
  std::ostringstream out;
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width + left
      << "\" height=\"" << height << "\" viewBox=\"0 0 " << width + left << " "
      << height << "\">\n";
  out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  for (Index w = 0; w < instance.workers; ++w) {
    const int y = 25 + static_cast<int>(w) * row_height;
    out << "<text x=\"8\" y=\"" << y + 18
        << "\" font-family=\"monospace\" font-size=\"12\">worker " << w
        << "</text>\n";
    out << "<line x1=\"" << left << "\" y1=\"" << y + 24 << "\" x2=\""
        << width + left << "\" y2=\"" << y + 24 << "\" stroke=\"#ddd\"/>\n";
  }
  for (const auto &op : schedule.operations_by_id) {
    const OperationView view = DecodeOperation(instance, op.id);
    const int x = left + static_cast<int>(
                             (op.start * static_cast<Tick>(width)) / scale_den);
    const int bar_width = std::max<int>(
        1, static_cast<int>(((op.end - op.start) * static_cast<Tick>(width)) /
                            scale_den));
    const int y = 25 + static_cast<int>(op.worker) * row_height;
    const char *color = view.backward ? "rgb(249,204,167)" : "rgb(106,149,206)";
    const int text_x = x + bar_width / 2;
    const int text_y = y + 10;
    out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << bar_width
        << "\" height=\"20\" fill=\"" << color << "\" stroke=\"#222\"/>\n";
    out << "<text x=\"" << text_x << "\" y=\"" << text_y
        << "\" font-family=\"monospace\" font-size=\"11\""
        << " text-anchor=\"middle\" dominant-baseline=\"central\""
        << " fill=\"#111\" pointer-events=\"none\">" << view.microbatch
        << "</text>\n";
    out << "<title>" << OperationName(instance, op.id) << " [" << op.start
        << "," << op.end << "]</title>\n";
  }
  out << "</svg>\n";
  return out.str();
}

void WriteTextFile(const std::string &path, const std::string &contents) {
  const std::filesystem::path final_path(path);
  const std::filesystem::path tmp_path =
      final_path.string() + ".tmp." + std::to_string(::getpid());
  std::ofstream out(tmp_path);
  if (!out) throw Error("failed to open output file: " + path);
  out << contents;
  out.close();
  if (!out) throw Error("failed to write output file: " + path);
  std::filesystem::rename(tmp_path, final_path);
}
std::string ToJson(const Instance& instance, const SearchResult& result) {
  const bool feasible = result.final_solution_available;
  if (feasible && !ValidateScheduleSolutionIndependent(instance, result.schedule).passed)
    throw Error("cannot serialize an invalid search result");
  std::ostringstream out;
  out << std::setprecision(17) << "{\n";
  auto text = [&](const char* key, const std::string& value) {
    out << "  \"" << key << "\": \"" << JsonEscape(value) << "\",\n";
  };
  auto flag = [&](const char* key, bool value) {
    out << "  \"" << key << "\": " << (value ? "true" : "false") << ",\n";
  };
  auto number = [&](const char* key, auto value) {
    out << "  \"" << key << "\": " << value << ",\n";
  };
  number("schema_version", 2);
  number("evaluation_method_version", 2);
  text("method_id", result.method);
  text("method", result.method);
  text("executor", "slackpipe");
  text("requested_command", result.requested_command);
  number("random_seed", result.random_seed);
  if (const auto* definition = FindEvaluationMethod(result.method)) {
    text("display_name", definition->display_name);
    text("method_contract_hash", EvaluationMethodContractHash(*definition));
  }
  text("status", result.status);
  if (result.cp_sat_launched || result.method == "cpsat-no-hint")
    text("solver_status_raw", result.solver_status_raw);
  else out << "  \"solver_status_raw\": null,\n";
  flag("final_solution_available", feasible);
  const bool solver_solution = result.cp_sat_launched &&
      (result.solver_status_raw == "FEASIBLE" || result.solver_status_raw == "OPTIMAL");
  flag("solver_solution_available", solver_solution);
  flag("fallback_used", false);
  flag("external_incumbent_available", false);
  flag("cp_sat_launched", result.cp_sat_launched);
  flag("optimal", result.proven_optimal);
  text("solution_source", result.solution_source);
  text("no_solution_reason", result.no_solution_reason);
  text("optimality_proof_source", result.optimality_proof_source);
  text("proof_scope", result.proven_optimal
       ? "modeled_partition_and_fifo_worker_order_problem; not GPU optimality" : "none");
  number("requested_time_limit_seconds", result.requested_seconds);
  number("wall_time_seconds", result.wall_time_seconds);
  number("solver_threads", result.num_workers);
  number("model_build_seconds", result.timing.model_build_seconds);
  number("solver_seconds", result.timing.solver_seconds);
  number("conservative_horizon", result.conservative_horizon);
  number("external_hint_count", result.external_hint_count);
  number("workload_constraint_count", result.workload_constraint_count);
  number("fifo_constraint_count", result.fifo_constraint_count);
  number("cp_sat_models_solved", result.cp_sat_models_solved);
  number("replay_cap_rejections", result.replay_cap_rejections);
  number("branches", result.branches);
  number("conflicts", result.conflicts);
  out << "  \"solver_progress\": ";
  if (result.cp_sat_launched) {
    out << "{\"deterministic_time\":" << result.deterministic_time
        << ",\"first_feasible_seconds\":";
    if (result.incumbent_trace.empty()) out << "null";
    else out << result.time_to_first_feasible_seconds;
    out << ",\"best_incumbent_seconds\":";
    if (result.incumbent_trace.empty()) out << "null";
    else out << result.time_to_best_incumbent_seconds;
    out << ",\"improvements\":[";
    for (std::size_t i = 0; i < result.incumbent_trace.size(); ++i) {
      if (i) out << ",";
      out << "{\"seconds\":" << result.incumbent_trace[i].first
          << ",\"raw_objective_ticks\":" << result.incumbent_trace[i].second << "}";
    }
    out << "]}";
  } else {
    out << "null";
  }
  out << ",\n";
  out << "  \"solver_objective_ticks\": ";
  if (solver_solution) out << result.solver_objective_ticks;
  else out << "null";
  out << ",\n  \"best_bound_ticks\": ";
  if (result.solver_bound_available) out << result.best_bound_ticks;
  else out << "null";
  out << ",\n  \"makespan_ticks\": ";
  if (feasible) out << result.makespan_ticks;
  else out << "null";
  out << ",\n";
  out << "  \"heuristic_certificate_bound_ticks\": ";
  if (result.method == "slack-heuristic") out << result.best_bound_ticks;
  else out << "null";
  out << ",\n";
  const auto bounds = ComputeLowerBoundReport(instance,
      result.solver_bound_available ? std::optional<double>(result.best_bound_ticks) : std::nullopt,
      feasible ? std::optional<Tick>(result.makespan_ticks) : std::nullopt);
  out << "  \"lower_bound_report\": " << LowerBoundReportJson(bounds) << ",\n";
  number("evaluations", result.evaluations);
  number("iterations", result.iterations);
  text("termination_reason", result.termination_reason);
  text("diagnostic", result.diagnostic);
  out << "  \"heuristic\": " << (result.heuristic_metadata_json.empty() ? "null" : result.heuristic_metadata_json) << ",\n";
  out << "  \"method_diagnostics\": " << (result.method_diagnostics_json.empty() ? "null" : result.method_diagnostics_json) << ",\n";
  text("git_commit", CurrentGitCommit().value_or("unknown"));
  flag("git_dirty", CurrentGitDirty().value_or(true));
  text("timestamp_utc", CurrentTimestampUtc());
  text("build_type", CurrentBuildType().value_or("unknown"));
  text("cost_profile_hash", instance.cost_profile_hash);
  text("cost_profile_path", instance.cost_profile_path);
  text("model_manifest_hash", instance.model_manifest_hash);
  text("cost_units", instance.HasCostProfile() ? instance.cost_profile_units : "abstract_ticks");
  number("total_layers", instance.total_layers);
  number("min_layers", instance.min_layers);
  number("backward_cost_ratio_numerator", instance.backward_ratio_num);
  number("backward_cost_ratio_denominator", 1);
  number("forward_cost_ratio_numerator", instance.backward_ratio_den);
  number("forward_cost_ratio_denominator", 1);
  text("communication_model", "constant_inter_worker_delay");
  number("communication_ticks", instance.communication_ticks);
  out << "  \"stage_to_worker_mapping\": [";
  for (Index s = 0; s < instance.stages; ++s) {
    if (s) out << ",";
    out << "{\"stage\":" << s << ",\"worker\":" << instance.WorkerForStage(s) << "}";
  }
  out << "],\n";
  if (instance.HasCostProfile()) {
    text("cost_profile_schema_version", instance.cost_profile_schema_version);
    out << "  \"duration_model_ticks\": {";
    bool comma = false;
    for (const auto& entry : std::vector<std::pair<std::string, std::vector<Tick>>>{
          {"forward_slope", instance.profile_forward_slope_ticks},
          {"backward_slope", instance.profile_backward_slope_ticks},
          {"forward_bias", instance.profile_forward_bias_ticks},
          {"backward_bias", instance.profile_backward_bias_ticks},
          {"prefix_forward", instance.profile_prefix_forward_ticks},
          {"prefix_backward", instance.profile_prefix_backward_ticks},
          {"role_forward_bias", instance.profile_role_forward_bias_ticks},
          {"role_backward_bias", instance.profile_role_backward_bias_ticks}}) {
      if (comma) out << ",";
      comma = true;
      out << "\"" << entry.first << "\": ";
      WriteTickArray(out, entry.second);
    }
    out << "},\n";
  }
  out << "  \"activation_analysis\": " <<
      (result.activation_analysis ? ActivationAnalysisJson(*result.activation_analysis) : "null") << ",\n";
  flag("activation_cap_constraints_added", result.activation_cap_constraints.constraints_added);
  if (feasible) {
    out << "  \"result_validation\": " <<
        ResultValidationToJson(ValidateScheduleSolutionIndependent(instance, result.schedule), "  ") << ",\n";
    const std::string schedule = ToJson(instance, result.schedule);
    out << schedule.substr(2);
  } else {
    number("microbatches", instance.microbatches);
    number("stages", instance.stages);
    number("workers", instance.workers);
    out << "  \"result_validation\": null,\n  \"ok\": false,\n"
        << "  \"makespan\": null,\n  \"split\": null,\n"
        << "  \"worker_local_operation_order\": null,\n  \"operations\": []\n}\n";
  }
  return out.str();
}

}  // namespace slackpipe
