#include "slackpipe/io.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

#include "slackpipe/pressure_pruning.h"
#include "slackpipe/result_schema.h"

namespace slackpipe {

namespace {
std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') out << '\\';
    out << ch;
  }
  return out.str();
}

void WriteTheoretical(std::ostringstream &out, bool available,
                      std::int64_t value) {
  if (available) {
    out << value;
  } else {
    out << "null";
  }
}

void WriteEnumeratedCount(std::ostringstream &out, bool enumerated,
                          std::int64_t value) {
  if (enumerated) {
    out << value;
  } else {
    out << "null";
  }
}

template <typename T>
void WriteOptionalNumber(std::ostringstream &out,
                         const std::optional<T> &value) {
  if (value) {
    out << *value;
  } else {
    out << "null";
  }
}

void WriteWorkerPartitionStats(std::ostringstream &out,
                               const SearchStats &stats,
                               const std::string &indent) {
  out << "{\n";
  out << indent << "  \"enabled\": "
      << (stats.worker_partitions_enabled ? "true" : "false") << ",\n";
  out << indent << "  \"reason\": \""
      << JsonEscape(stats.worker_partitions_reason) << "\",\n";
  out << indent << "  \"theoretical\": ";
  WriteTheoretical(out, stats.worker_partitions_theoretical_available,
                   stats.worker_partitions_theoretical);
  out << ",\n";
  out << indent << "  \"theoretical_available\": "
      << (stats.worker_partitions_theoretical_available ? "true" : "false")
      << ",\n";
  out << indent << "  \"enumerated\": "
      << (stats.worker_partitions_enumerated ? "true" : "false") << ",\n";
  out << indent << "  \"visited\": ";
  WriteEnumeratedCount(out, stats.worker_partitions_enumerated,
                       stats.worker_partitions_visited);
  out << ",\n";
  out << indent << "  \"pruned\": ";
  WriteEnumeratedCount(out, stats.worker_partitions_enumerated,
                       stats.worker_partitions_pruned);
  out << ",\n";
  out << indent << "  \"kept\": ";
  WriteEnumeratedCount(out, stats.worker_partitions_enumerated,
                       stats.worker_partitions_kept);
  out << "\n" << indent << "}";
}

void WriteStagePartitionStats(std::ostringstream &out, const SearchStats &stats,
                              const std::string &indent) {
  out << "{\n";
  out << indent << "  \"theoretical\": ";
  WriteTheoretical(out, stats.stage_partitions_theoretical_available,
                   stats.stage_partitions_theoretical);
  out << ",\n";
  out << indent << "  \"theoretical_available\": "
      << (stats.stage_partitions_theoretical_available ? "true" : "false")
      << ",\n";
  out << indent << "  \"note\": \"" << JsonEscape(stats.stage_partitions_note)
      << "\",\n";
  out << indent << "  \"enumerated\": "
      << (stats.stage_partitions_enumerated ? "true" : "false") << ",\n";
  out << indent << "  \"visited\": ";
  WriteEnumeratedCount(out, stats.stage_partitions_enumerated,
                       stats.stage_partitions_visited);
  out << ",\n";
  out << indent << "  \"pruned\": ";
  WriteEnumeratedCount(out, stats.stage_partitions_enumerated,
                       stats.stage_partitions_pruned);
  out << ",\n";
  out << indent << "  \"kept\": ";
  WriteEnumeratedCount(out, stats.stage_partitions_enumerated,
                       stats.stage_partitions_kept);
  out << "\n" << indent << "}";
}

void WriteInterleaveStats(std::ostringstream &out, const SearchStats &stats,
                          const std::string &indent) {
  out << "{\n";
  out << indent << "  \"theoretical\": ";
  WriteTheoretical(out, stats.interleave_orders_theoretical_available,
                   stats.interleave_orders_theoretical);
  out << ",\n";
  out << indent << "  \"theoretical_available\": "
      << (stats.interleave_orders_theoretical_available ? "true" : "false")
      << ",\n";
  out << indent << "  \"note\": \"" << JsonEscape(stats.interleave_orders_note)
      << "\",\n";
  out << indent << "  \"enumerated\": "
      << (stats.interleave_orders_enumerated ? "true" : "false") << ",\n";
  out << indent << "  \"visited\": ";
  WriteEnumeratedCount(out, stats.interleave_orders_enumerated,
                       stats.interleave_orders_visited);
  out << ",\n";
  out << indent << "  \"deduplicated\": ";
  WriteEnumeratedCount(out, stats.interleave_orders_enumerated,
                       stats.interleave_orders_deduplicated);
  out << ",\n";
  out << indent << "  \"evaluated\": ";
  WriteEnumeratedCount(out, stats.interleave_orders_enumerated,
                       stats.interleave_orders_evaluated);
  out << ",\n";
  out << indent << "  \"kept\": ";
  WriteEnumeratedCount(out, stats.interleave_orders_enumerated,
                       stats.interleave_orders_kept);
  out << "\n" << indent << "}";
}

void WriteTickArray(std::ostringstream &out, const std::vector<Tick> &values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << values[i];
  }
  out << "]";
}

void WriteDoubleArray(std::ostringstream &out,
                      const std::vector<double> &values) {
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

void WriteWorkerAggregateLayerReport(std::ostringstream &out,
                                     const Instance &instance,
                                     const std::vector<Tick> &split,
                                     const std::string &indent,
                                     bool trailing_comma) {
  if (split.empty()) return;
  const std::vector<Tick> loads = WorkerLayerTotals(instance, split);
  Tick min_load = loads.empty() ? 0 : loads.front();
  Tick max_load = loads.empty() ? 0 : loads.front();
  for (Tick load : loads) {
    min_load = std::min(min_load, load);
    max_load = std::max(max_load, load);
  }
  out << indent << "\"stage_partition\": ";
  WriteTickArray(out, split);
  out << ",\n";
  out << indent << "\"worker_aggregate_layer_loads\": ";
  WriteTickArray(out, loads);
  out << ",\n";
  out << indent << "\"worker_balance\": {\n";
  out << indent << "  \"max_load\": " << max_load << ",\n";
  out << indent << "  \"min_load\": " << min_load << ",\n";
  out << indent << "  \"imbalance\": " << (max_load - min_load) << "\n";
  out << indent << "}";
  if (trailing_comma) out << ",";
  out << "\n";
}

void WriteWorkerBalanceConstraint(
    std::ostringstream &out, const WorkerBalanceConstraintResult &constraint,
    const Instance &instance, const std::vector<Tick> &split,
    const std::string &indent, bool trailing_comma) {
  out << indent << "\"worker_balance_constraint\": {\n";
  out << indent << "  \"enabled\": " << (constraint.enabled ? "true" : "false");
  if (constraint.enabled) {
    const std::vector<Tick> loads = split.empty()
                                        ? std::vector<Tick>{}
                                        : WorkerLayerTotals(instance, split);
    Tick min_load = loads.empty() ? 0 : loads.front();
    Tick max_load = loads.empty() ? 0 : loads.front();
    for (Tick load : loads) {
      min_load = std::min(min_load, load);
      max_load = std::max(max_load, load);
    }
    out << ",\n";
    out << indent << "  \"tolerance_percent\": ";
    if (constraint.tolerance_percent >= 0.0) {
      out << constraint.tolerance_percent;
    } else {
      out << "null";
    }
    out << ",\n";
    out << indent << "  \"tolerance_layers\": " << constraint.tolerance_layers
        << ",\n";
    out << indent << "  \"lower_bound\": " << constraint.lower_bound << ",\n";
    out << indent << "  \"upper_bound\": " << constraint.upper_bound << ",\n";
    out << indent << "  \"worker_aggregate_layer_loads\": ";
    WriteTickArray(out, loads);
    out << ",\n";
    out << indent << "  \"max_load\": " << max_load << ",\n";
    out << indent << "  \"min_load\": " << min_load << ",\n";
    out << indent << "  \"imbalance\": " << (max_load - min_load) << "\n";
    out << indent << "}";
  } else {
    out << "\n" << indent << "}";
  }
  if (trailing_comma) out << ",";
  out << "\n";
}

void WriteSearchStatsObject(std::ostringstream &out, const SearchStats &stats,
                            const std::string &indent) {
  const std::string nested = indent + "  ";
  out << "{\n";
  out << nested << "\"algorithm\": \"" << JsonEscape(stats.algorithm)
      << "\",\n";
  out << nested << "\"enumerative_search\": "
      << (stats.enumerative_search ? "true" : "false") << ",\n";
  out << nested << "\"worker_level_partition\": ";
  WriteWorkerPartitionStats(out, stats, nested);
  out << ",\n";
  out << nested << "\"stage_level_partition\": ";
  WriteStagePartitionStats(out, stats, nested);
  out << ",\n";
  out << nested << "\"interleave_reordering\": ";
  WriteInterleaveStats(out, stats, nested);
  out << ",\n";
  out << nested << "\"candidate_schedules\": {\n";
  out << nested << "  \"extracted\": " << stats.candidate_schedules_extracted
      << ",\n";
  out << nested << "  \"deterministically_evaluated\": "
      << stats.candidate_schedules_deterministically_evaluated << ",\n";
  out << nested << "  \"accepted\": " << stats.candidate_schedules_accepted
      << ",\n";
  out << nested << "  \"rejected\": " << stats.candidate_schedules_rejected
      << "\n";
  out << nested << "}";
  if (stats.cp_sat_available) {
    out << ",\n";
    out << nested << "\"cp_sat\": {\n";
    out << nested << "  \"status\": \"" << JsonEscape(stats.cp_sat_status)
        << "\",\n";
    out << nested << "  \"objective\": " << stats.cp_sat_objective << ",\n";
    out << nested << "  \"best_bound\": " << stats.cp_sat_best_bound << ",\n";
    out << nested << "  \"branches\": " << stats.cp_sat_branches << ",\n";
    out << nested << "  \"conflicts\": " << stats.cp_sat_conflicts << ",\n";
    out << nested
        << "  \"wall_time_seconds\": " << stats.cp_sat_wall_time_seconds
        << ",\n";
    out << nested
        << "  \"deterministic_time\": " << stats.cp_sat_deterministic_time
        << "\n";
    out << nested << "}";
  }
  out << "\n" << indent << "}";
}
}  // namespace

std::string ToJson(const Instance &instance, const ScheduleSolution &schedule,
                   const CanonicalResultMetadata &canonical,
                   const SearchStats *search_stats) {
  std::string json = search_stats == nullptr
                         ? ToJson(instance, schedule)
                         : ToJson(instance, schedule, search_stats);
  const std::string marker = "{\n";
  const std::size_t insert_at = json.find(marker);
  if (insert_at == std::string::npos) return json;
  json.insert(insert_at + marker.size(),
              CanonicalResultTopLevelJsonFields(canonical, "  ", true));
  return json;
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

std::string ToJson(const Instance &instance, const ScheduleSolution &schedule,
                   const SearchStats *search_stats) {
  if (search_stats == nullptr) return ToJson(instance, schedule);
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
  out << "  \"search_stats\": ";
  WriteSearchStatsObject(out, *search_stats, "  ");
  out << ",\n";
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

std::string ToJson(const Instance &instance, const SearchStats &search_stats) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"config\": {\n";
  out << "    \"B\": " << instance.microbatches << ",\n";
  out << "    \"N\": " << instance.stages << ",\n";
  out << "    \"J\": " << instance.workers << ",\n";
  out << "    \"L\": " << instance.total_layers << "\n";
  out << "  },\n";
  out << "  \"search_stats\": ";
  WriteSearchStatsObject(out, search_stats, "  ");
  out << "\n";
  out << "}\n";
  return out.str();
}

std::string SearchStatsSummary(const SearchStats &search_stats) {
  std::ostringstream out;
  out << "search_stats: enumerative_search="
      << (search_stats.enumerative_search ? "true" : "false")
      << "; stage_partitions enumerated="
      << (search_stats.stage_partitions_enumerated ? "true" : "false");
  if (search_stats.stage_partitions_enumerated) {
    out << " visited=" << search_stats.stage_partitions_visited
        << " pruned=" << search_stats.stage_partitions_pruned
        << " kept=" << search_stats.stage_partitions_kept;
  }
  out << "; interleave_orders enumerated="
      << (search_stats.interleave_orders_enumerated ? "true" : "false");
  if (search_stats.interleave_orders_enumerated) {
    out << " visited=" << search_stats.interleave_orders_visited
        << " deduplicated=" << search_stats.interleave_orders_deduplicated
        << " evaluated=" << search_stats.interleave_orders_evaluated
        << " kept=" << search_stats.interleave_orders_kept;
  }
  out << "; candidate_schedules extracted="
      << search_stats.candidate_schedules_extracted << " evaluated="
      << search_stats.candidate_schedules_deterministically_evaluated
      << " accepted=" << search_stats.candidate_schedules_accepted
      << " rejected=" << search_stats.candidate_schedules_rejected;
  if (search_stats.cp_sat_available) {
    out << "; cp_sat status=" << search_stats.cp_sat_status
        << " branches=" << search_stats.cp_sat_branches
        << " conflicts=" << search_stats.cp_sat_conflicts;
  }
  return out.str();
}

void WritePressurePruningStats(std::ostringstream &out,
                               const PressurePruningStats &stats,
                               const std::string &indent, bool trailing_comma) {
  out << indent << "\"pressure_pruning\": {\n";
  out << indent << "  \"enabled\": " << (stats.enabled ? "true" : "false")
      << ",\n";
  out << indent << "  \"generation_mode\": \""
      << JsonEscape(stats.generation_mode) << "\",\n";
  out << indent << "  \"generated_partitions\": " << stats.generated_partitions
      << ",\n";
  out << indent << "  \"partitions_before\": " << stats.partitions_before
      << ",\n";
  out << indent << "  \"partitions_after\": " << stats.partitions_after
      << ",\n";
  out << indent << "  \"anchor_splits_added\": " << stats.anchor_splits_added
      << ",\n";
  out << indent << "  \"incumbent_split_included\": "
      << (stats.incumbent_split_included ? "true" : "false") << ",\n";
  out << indent << "  \"uniform_split_included\": "
      << (stats.uniform_split_included ? "true" : "false") << ",\n";
  out << indent << "  \"cost_balanced_split_included\": "
      << (stats.cost_balanced_split_included ? "true" : "false") << ",\n";
  out << indent
      << "  \"selected_splits_total\": " << stats.selected_splits_total
      << ",\n";
  out << indent << "  \"beam_width\": " << stats.beam_width << ",\n";
  out << indent << "  \"branch_width\": " << stats.branch_width << ",\n";
  out << indent << "  \"exhaustive_enumeration_skipped\": "
      << (stats.exhaustive_enumeration_skipped ? "true" : "false") << ",\n";
  out << indent << "  \"best_pressure\": " << stats.best_pressure << ",\n";
  out << indent << "  \"pressure_cutoff\": ";
  if (stats.pressure_cutoff_available) {
    out << stats.pressure_cutoff;
  } else {
    out << "null";
  }
  out << ",\n";
  out << indent << "  \"predecessor_candidates_before\": "
      << stats.predecessor_candidates_before << ",\n";
  out << indent << "  \"predecessor_candidates_after\": "
      << stats.predecessor_candidates_after << ",\n";
  out << indent << "  \"predecessor_note\": \""
      << JsonEscape(stats.predecessor_note) << "\"\n";
  out << indent << "}";
  if (trailing_comma) out << ",";
  out << "\n";
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

std::string ToJson(const Instance &instance,
                   const BfsSplitOptimizationResult &result) {
  std::ostringstream out;
  out << "{\n";
  if (result.canonical) {
    out << CanonicalResultTopLevelJsonFields(*result.canonical, "  ", true);
  }
  out << "  \"algorithm\": \"" << (result.method == "heuristic" ? "heuristic" :
      "optimize-bfs") << "\",\n";
  if (!result.heuristic_metadata_json.empty())
    out << "  \"heuristic\": " << result.heuristic_metadata_json << ",\n";
  out << "  \"method\": \"" << JsonEscape(result.method) << "\",\n";
  out << "  \"status\": \"" << JsonEscape(result.status) << "\",\n";
  out << "  \"solver_status_raw\": \""
      << JsonEscape(result.solver_status_raw.empty() ? result.status
                                                     : result.solver_status_raw)
      << "\",\n";
  out << "  \"proven_optimal\": " << (result.proven_optimal ? "true" : "false")
      << ",\n";
  out << "  \"split\": [";
  for (std::size_t i = 0; i < result.split.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.split[i];
  }
  out << "],\n";
  WriteWorkerAggregateLayerReport(out, instance, result.split, "  ", true);
  WriteScheduleMetrics(out, instance, result.schedule, "  ", true);
  WriteWorkerOrdersJson(out, instance, result.machine_orders, "  ", true);
  out << "  \"makespan_ticks\": " << result.makespan_ticks << ",\n";
  out << "  \"ratio_num\": " << instance.backward_ratio_num << ",\n";
  out << "  \"ratio_den\": " << instance.backward_ratio_den << ",\n";
  out << "  \"best_bound_ticks\": " << result.best_bound_ticks << ",\n";
  out << "  \"solver_objective_ticks\": " << result.solver_objective_ticks
      << ",\n";
  out << "  \"wall_time_seconds\": " << result.wall_time_seconds << ",\n";
  out << "  \"branches\": " << result.branches << ",\n";
  out << "  \"conflicts\": " << result.conflicts << ",\n";
  out << "  \"checked_splits\": " << result.checked_splits << ",\n";
  if (result.method == "heuristic") {
    out << "  \"ok\": " << (result.schedule.ok() ? "true" : "false") << ",\n"
        << "  \"reported_status\": \"" << JsonEscape(result.status) << "\",\n"
        << "  \"hint_budget_seconds\": " << result.hint_budget_seconds << ",\n"
        << "  \"hint_elapsed_seconds\": " << result.hint_elapsed_seconds << ",\n"
        << "  \"hint_iterations\": " << result.hint_iterations << ",\n"
        << "  \"hint_candidates_generated\": " << result.hint_candidates_generated << ",\n"
        << "  \"hint_candidates_simulated\": " << result.hint_candidates_simulated << ",\n"
        << "  \"hint_deadline_reached\": " << (result.hint_deadline_reached ? "true" : "false") << ",\n"
        << "  \"hint_termination_reason\": \"" << JsonEscape(result.hint_termination_reason) << "\",\n";
    out << "  \"stage_to_worker_mapping\": [";
    for (Index s = 0; s < instance.stages; ++s) {
      if (s) out << ", ";
      out << "{\"stage\": " << s << ", \"worker\": " << instance.WorkerForStage(s) << "}";
    }
    out << "],\n";
    // Self-contained evaluator inputs, not a new calibration profile format.
    out << "  \"duration_model_ticks\": {";
    auto array = [&](const char* name, const std::vector<Tick>& values, bool comma) {
      if (comma) out << ", ";
      out << '"' << name << "\": [";
      for (std::size_t k = 0; k < values.size(); ++k) { if (k) out << ", "; out << values[k]; }
      out << ']';
    };
    array("forward_slope", instance.profile_forward_slope_ticks, false);
    array("backward_slope", instance.profile_backward_slope_ticks, true);
    array("forward_bias", instance.profile_forward_bias_ticks, true);
    array("backward_bias", instance.profile_backward_bias_ticks, true);
    array("prefix_forward", instance.profile_prefix_forward_ticks, true);
    array("prefix_backward", instance.profile_prefix_backward_ticks, true);
    array("role_forward_bias", instance.profile_role_forward_bias_ticks, true);
    array("role_backward_bias", instance.profile_role_backward_bias_ticks, true);
    out << "},\n"
        << "  \"cost_profile_hash\": \"" << JsonEscape(instance.cost_profile_hash) << "\",\n"
        << "  \"cost_profile_schema_version\": \"" << JsonEscape(instance.cost_profile_schema_version) << "\",\n";
  }
  out << "  \"enumeration_candidates_total\": "
      << result.enumeration_candidates_total << ",\n";
  out << "  \"enumeration_candidates_valid_schedule\": "
      << result.enumeration_candidates_valid_schedule << ",\n";
  out << "  \"enumeration_candidates_cap_feasible\": "
      << result.enumeration_candidates_cap_feasible << ",\n";
  out << "  \"enumeration_candidates_cap_rejected\": "
      << result.enumeration_candidates_cap_rejected << ",\n";
  out << "  \"enumeration_proved_optimal\": "
      << (result.enumeration_proved_optimal ? "true" : "false") << ",\n";
  out << "  \"optimality_proof_source\": \""
      << JsonEscape(result.optimality_proof_source) << "\",\n";
  out << "  \"fixed_order_partition_backend_requested\": \""
      << JsonEscape(result.fixed_order_partition_backend_requested) << "\",\n";
  out << "  \"fixed_order_partition_backend_effective\": \""
      << JsonEscape(result.fixed_order_partition_backend_effective) << "\",\n";
  out << "  \"estimated_partition_count\": " << result.estimated_partition_count
      << ",\n";
  out << "  \"estimated_partition_count_available\": "
      << (result.estimated_partition_count_available ? "true" : "false")
      << ",\n";
  out << "  \"enumeration_safety_threshold\": "
      << result.enumeration_safety_threshold << ",\n";
  out << "  \"cp_sat_launched\": "
      << (result.cp_sat_launched ? "true" : "false") << ",\n";
  out << "  \"cp_sat_models_solved\": " << result.cp_sat_models_solved << ",\n";
  out << "  \"fallback_available\": "
      << (result.fallback_available ? "true" : "false") << ",\n";
  out << "  \"fallback_used\": " << (result.fallback_used ? "true" : "false")
      << ",\n";
  out << "  \"fallback_reason\": \"" << JsonEscape(result.fallback_reason)
      << "\",\n";
  out << "  \"fallback_source\": \"" << JsonEscape(result.fallback_source)
      << "\",\n";
  out << "  \"solution_source\": \"" << JsonEscape(result.solution_source)
      << "\",\n";
  out << "  \"returned_solution_source\": \""
      << JsonEscape(result.returned_solution_source) << "\",\n";
  out << "  \"deterministic_time\": " << result.deterministic_time << ",\n";
  out << "  \"time_to_first_feasible_seconds\": "
      << result.time_to_first_feasible_seconds << ",\n";
  out << "  \"time_to_best_incumbent_seconds\": "
      << result.time_to_best_incumbent_seconds << ",\n";
  out << "  \"first_feasible_objective\": " << result.first_feasible_objective
      << ",\n";
  out << "  \"incumbent_improvement_count\": "
      << result.incumbent_improvement_count << ",\n";
  out << "  \"incumbent_trace\": [";
  for (std::size_t i = 0; i < result.incumbent_trace.size(); ++i) {
    if (i != 0) out << ", ";
    out << "{\"time_seconds\": " << result.incumbent_trace[i].first
        << ", \"objective\": " << result.incumbent_trace[i].second << "}";
  }
  out << "],\n";
  out << "  \"microbatches\": " << instance.microbatches << ",\n";
  out << "  \"stages\": " << instance.stages << ",\n";
  out << "  \"workers\": " << instance.workers << ",\n";
  out << "  \"total_layers\": " << instance.total_layers << ",\n";
  out << "  \"min_layers\": " << instance.min_layers << ",\n";
  out << "  \"communication_ticks\": " << instance.communication_ticks << ",\n";
  out << "  \"diagnostic\": \"" << JsonEscape(result.diagnostic) << "\",\n";
  if (result.search_stats_enabled) {
    out << "  \"search_stats\": ";
    WriteSearchStatsObject(out, result.search_stats, "  ");
    out << ",\n";
  }
  out << "  \"operations\": [\n";
  for (std::size_t i = 0; i < result.schedule.operations_by_id.size(); ++i) {
    const auto &op = result.schedule.operations_by_id[i];
    const OperationView view = DecodeOperation(instance, op.id);
    out << "    {\"id\": " << op.id.value << ", \"name\": \""
        << OperationName(instance, op.id)
        << "\", \"microbatch\": " << view.microbatch
        << ", \"chain_index\": " << view.chain_index
        << ", \"stage\": " << view.stage << ", \"kind\": \""
        << (view.backward ? "B" : "F") << "\", \"worker\": " << op.worker
        << ", \"start\": " << op.start << ", \"end\": " << op.end
        << ", \"duration\": " << op.duration << "}";
    if (i + 1 != result.schedule.operations_by_id.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::string ToJson(const Instance &instance,
                   const JointOptimizationResult &result) {
  std::ostringstream out;
  out << "{\n";
  if (result.canonical) {
    out << CanonicalResultTopLevelJsonFields(*result.canonical, "  ", true);
  }
  out << "  \"algorithm\": \"optimize-joint\",\n";
  out << "  \"workload_constraint_count\": " << result.workload_constraint_count << ",\n";
  auto bound_report = result.canonical ? result.canonical->lower_bound_report : ComputeLowerBoundReport(
      instance, result.best_bound_ticks,
      (result.status == "OPTIMAL" || result.status == "FEASIBLE") && result.makespan_ticks > 0
          ? std::optional<Tick>(result.makespan_ticks) : std::nullopt,
      result.raw_bound_globally_valid);
  if (result.canonical && (!result.canonical->outcome.feasible.value_or(false) ||
      !result.canonical->outcome.makespan || *result.canonical->outcome.makespan <= 0)) {
    bound_report.effective_relative_gap = std::nullopt;
  }
  out << "  \"lower_bound_report\": " << LowerBoundReportJson(bound_report) << ",\n";
  out << "  \"status\": \"" << JsonEscape(result.status) << "\",\n";
  out << "  \"proven_optimal\": " << (result.proven_optimal ? "true" : "false")
      << ",\n";
  out << "  \"split\": [";
  for (std::size_t i = 0; i < result.split.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.split[i];
  }
  out << "],\n";
  WriteWorkerAggregateLayerReport(out, instance, result.split, "  ", true);
  WriteWorkerBalanceConstraint(out, result.worker_balance_constraint, instance,
                               result.split, "  ", true);
  out << "  \"worker_balance_pruning_requested\": "
      << (result.worker_balance_pruning_requested ? "true" : "false") << ",\n";
  out << "  \"worker_balance_pruning_effective\": "
      << (result.worker_balance_pruning_effective ? "true" : "false") << ",\n";
  out << "  \"worker_balance_tolerance_requested\": {\n";
  out << "    \"percent\": ";
  WriteOptionalNumber(out, result.worker_balance_tolerance_requested_percent);
  out << ",\n";
  out << "    \"layers\": ";
  WriteOptionalNumber(out, result.worker_balance_tolerance_requested_layers);
  out << "\n";
  out << "  },\n";
  WritePressurePruningStats(out, result.pressure_pruning_stats, "  ", true);
  WriteScheduleMetrics(out, instance, result.schedule, "  ", true,
                       result.fifo_ordering_effective);
  WriteWorkerOrdersJson(out, instance, result.machine_orders, "  ", true);
  out << "  \"makespan_ticks\": " << result.makespan_ticks << ",\n";
  out << "  \"solver_objective_ticks\": " << result.solver_objective_ticks
      << ",\n";
  out << "  \"best_bound_ticks\": " << result.best_bound_ticks << ",\n";
  out << "  \"bfs_incumbent_split\": [";
  for (std::size_t i = 0; i < result.bfs_incumbent.split.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.bfs_incumbent.split[i];
  }
  out << "],\n";
  out << "  \"bfs_incumbent_makespan_ticks\": "
      << result.bfs_incumbent.makespan_ticks << ",\n";
  if (result.bfs_incumbent.method == "heuristic") {
    out << "  \"early_hint\": " << ToJson(instance, result.bfs_incumbent) << ",\n";
  }
  out << "  \"improvement_over_bfs_ticks\": ";
  if (result.bfs_incumbent.makespan_ticks > 0) {
    out << result.bfs_incumbent.makespan_ticks - result.makespan_ticks;
  } else {
    out << "null";
  }
  out << ",\n";
  out << "  \"wall_time_seconds\": " << result.wall_time_seconds << ",\n";
  out << "  \"global_time_limit_s\": " << result.global_time_limit_seconds
      << ",\n";
  out << "  \"total_wall_time_s\": " << result.wall_time_seconds << ",\n";
  out << "  \"incumbent_budget_seconds\": " << result.incumbent_budget_seconds
      << ",\n";
  out << "  \"incumbent_budget_s\": " << result.incumbent_budget_seconds
      << ",\n";
  out << "  \"incumbent_time_s\": " << result.timing.incumbent_seconds << ",\n";
  out << "  \"incumbent_model_build_seconds\": "
      << result.incumbent_model_build_seconds << ",\n";
  out << "  \"incumbent_solve_seconds\": " << result.incumbent_solve_seconds
      << ",\n";
  out << "  \"incumbent_status\": \"" << JsonEscape(result.incumbent_status)
      << "\",\n";
  out << "  \"joint_budget_seconds\": " << result.joint_budget_seconds << ",\n";
  out << "  \"cpsat_budget_s\": " << result.joint_budget_seconds << ",\n";
  out << "  \"joint_model_build_seconds\": " << result.joint_model_build_seconds
      << ",\n";
  out << "  \"model_build_time_s\": " << result.timing.model_build_seconds
      << ",\n";
  out << "  \"joint_solve_seconds\": " << result.joint_solve_seconds << ",\n";
  out << "  \"cpsat_wall_time_s\": " << result.timing.ortools_wall_time_seconds
      << ",\n";
  out << "  \"joint_status\": \"" << JsonEscape(result.joint_status) << "\",\n";
  out << "  \"solver_status\": \"" << JsonEscape(result.joint_status)
      << "\",\n";
  out << "  \"best_objective\": " << result.solver_objective_ticks << ",\n";
  out << "  \"best_bound\": " << result.best_bound_ticks << ",\n";
  out << "  \"optimality_gap\": ";
  if (result.makespan_ticks > 0 && result.best_bound_ticks > 0.0) {
    WriteOptionalNumber(out, slackpipe::RelativeGap(result.makespan_ticks,
                                                    result.best_bound_ticks));
  } else {
    out << "null";
  }
  out << ",\n";
  out << "  \"incumbent_method_requested\": \""
      << JsonEscape(result.incumbent_method_requested) << "\",\n";
  out << "  \"incumbent_method_requested_normalized\": \""
      << JsonEscape(result.incumbent_method_requested_normalized) << "\",\n";
  out << "  \"incumbent_method_effective\": \""
      << JsonEscape(result.incumbent_method_effective) << "\",\n";
  out << "  \"bfs_incumbent_method_requested\": \""
      << JsonEscape(result.bfs_incumbent_method_requested) << "\",\n";
  out << "  \"bfs_incumbent_method_effective\": \""
      << JsonEscape(result.bfs_incumbent_method_effective) << "\",\n";
  out << "  \"incumbent_source\": \"" << JsonEscape(result.incumbent_source)
      << "\",\n";
  out << "  \"incumbent_feasible\": "
      << (result.incumbent_feasible ? "true" : "false") << ",\n";
  out << "  \"incumbent_found\": "
      << (result.incumbent_found ? "true" : "false") << ",\n";
  out << "  \"incumbent_valid\": "
      << (result.incumbent_valid ? "true" : "false") << ",\n";
  out << "  \"incumbent_makespan\": " << result.incumbent_makespan << ",\n";
  out << "  \"incumbent_primary_objective\": "
      << result.incumbent_primary_objective << ",\n";
  out << "  \"incumbent_hybrid_min_slack\": "
      << result.incumbent_hybrid_min_slack << ",\n";
  out << "  \"incumbent_baseline_primary_objective\": "
      << result.incumbent_baseline_primary_objective << ",\n";
  out << "  \"incumbent_baseline_hybrid_min_slack\": "
      << result.incumbent_baseline_hybrid_min_slack << ",\n";
  out << "  \"incumbent_improved_over_baseline\": "
      << (result.incumbent_improved_over_baseline ? "true" : "false") << ",\n";
  out << "  \"incumbent_hybrid_stage_scores\": ";
  WriteDoubleArray(out, result.incumbent_hybrid_stage_scores);
  out << ",\n";
  out << "  \"incumbent_hybrid_bottleneck_stages\": ";
  WriteTickArray(out, result.incumbent_hybrid_bottleneck_stages);
  out << ",\n";
  out << "  \"horizon_source\": \"" << JsonEscape(result.horizon_source)
      << "\",\n";
  out << "  \"incumbent_bound_requested\": "
      << (result.incumbent_bound_requested ? "true" : "false") << ",\n";
  out << "  \"incumbent_bound_effective\": "
      << (result.incumbent_bound_effective ? "true" : "false") << ",\n";
  out << "  \"incumbent_bound_horizon\": " << result.incumbent_bound_horizon
      << ",\n";
  out << "  \"hint_budget_seconds\": " << result.hint_budget_seconds << ",\n";
  out << "  \"hint_elapsed_seconds\": " << result.hint_elapsed_seconds << ",\n";
  out << "  \"hint_iterations\": " << result.hint_iterations << ",\n";
  out << "  \"hint_candidates_generated\": " << result.hint_candidates_generated
      << ",\n";
  out << "  \"hint_candidates_simulated\": " << result.hint_candidates_simulated
      << ",\n";
  out << "  \"hint_partition_moves_accepted\": "
      << result.hint_partition_moves_accepted << ",\n";
  out << "  \"hint_interleaving_moves_accepted\": "
      << result.hint_interleaving_moves_accepted << ",\n";
  out << "  \"hint_deadline_reached\": "
      << (result.hint_deadline_reached ? "true" : "false") << ",\n";
  out << "  \"hint_termination_reason\": \""
      << JsonEscape(result.hint_termination_reason) << "\",\n";
  out << "  \"hints_requested\": "
      << (result.hints_requested ? "true" : "false") << ",\n";
  out << "  \"incumbent_hints_requested\": "
      << (result.incumbent_hints_requested ? "true" : "false") << ",\n";
  out << "  \"hints_effective\": "
      << (result.hints_effective ? "true" : "false") << ",\n";
  out << "  \"incumbent_hints_effective\": "
      << (result.incumbent_hints_effective ? "true" : "false") << ",\n";
  out << "  \"hint_source\": \"" << JsonEscape(result.hint_source) << "\",\n";
  out << "  \"hint_scope\": \"" << JsonEscape(result.hint_scope) << "\",\n";
  out << "  \"hint_complete_for_basic_model\": "
      << (result.hint_complete_for_basic_model ? "true" : "false") << ",\n";
  out << "  \"hint_complete_for_full_model\": "
      << (result.hint_complete_for_full_model ? "true" : "false") << ",\n";
  out << "  \"hinted_layer_variable_count\": "
      << result.hinted_layer_variable_count << ",\n";
  out << "  \"hinted_operation_variable_count\": "
      << result.hinted_operation_variable_count << ",\n";
  out << "  \"hinted_scalar_variable_count\": "
      << result.hinted_scalar_variable_count << ",\n";
  out << "  \"hinted_auxiliary_variable_count\": "
      << result.hinted_auxiliary_variable_count << ",\n";
  out << "  \"hinted_total_variable_count\": "
      << result.hinted_total_variable_count << ",\n";
  out << "  \"auxiliary_variable_count\": " << result.auxiliary_variable_count
      << ",\n";
  out << "  \"fallback_available\": "
      << (result.fallback_available ? "true" : "false") << ",\n";
  out << "  \"fallback_enabled\": "
      << (result.fallback_enabled ? "true" : "false") << ",\n";
  out << "  \"fallback_source\": \"" << JsonEscape(result.fallback_source)
      << "\",\n";
  out << "  \"solution_source\": \"" << JsonEscape(result.solution_source)
      << "\",\n";
  out << "  \"solver_solution_available\": "
      << (result.solver_solution_available ? "true" : "false") << ",\n";
  out << "  \"final_solution_available\": "
      << (result.final_solution_available ? "true" : "false") << ",\n";
  out << "  \"final_solution_source\": \""
      << JsonEscape(result.final_solution_source) << "\",\n";
  out << "  \"no_solution_reason\": \"" << JsonEscape(result.no_solution_reason)
      << "\",\n";
  out << "  \"fallback_used\": " << (result.fallback_used ? "true" : "false")
      << ",\n";
  out << "  \"cp_sat_models_solved\": " << result.cp_sat_models_solved << ",\n";
  out << "  \"deterministic_time\": " << result.deterministic_time << ",\n";
  out << "  \"branches\": " << result.branches << ",\n";
  out << "  \"num_branches\": " << result.branches << ",\n";
  out << "  \"conflicts\": " << result.conflicts << ",\n";
  out << "  \"num_conflicts\": " << result.conflicts << ",\n";
  out << "  \"time_to_first_feasible_seconds\": "
      << result.time_to_first_feasible_seconds << ",\n";
  out << "  \"time_to_first_cpsat_feasible_solution\": "
      << result.time_to_first_cpsat_feasible_seconds << ",\n";
  out << "  \"time_to_first_cpsat_feasible_seconds\": "
      << result.time_to_first_cpsat_feasible_seconds << ",\n";
  out << "  \"first_cpsat_feasible_objective\": "
      << result.first_cpsat_feasible_objective << ",\n";
  out << "  \"time_to_best_incumbent_seconds\": "
      << result.time_to_best_incumbent_seconds << ",\n";
  out << "  \"first_feasible_objective\": " << result.first_feasible_objective
      << ",\n";
  out << "  \"incumbent_improvement_count\": "
      << result.incumbent_improvement_count << ",\n";
  out << "  \"incumbent_trace\": [";
  for (std::size_t i = 0; i < result.incumbent_trace.size(); ++i) {
    if (i != 0) out << ", ";
    out << "{\"time_seconds\": " << result.incumbent_trace[i].first
        << ", \"objective\": " << result.incumbent_trace[i].second << "}";
  }
  out << "],\n";
  out << "  \"num_workers\": " << result.num_workers << ",\n";
  out << "  \"fifo_ordering_requested\": "
      << (result.fifo_ordering_requested ? "true" : "false") << ",\n";
  out << "  \"fifo_ordering_effective\": "
      << (result.fifo_ordering_effective ? "true" : "false") << ",\n";
  out << "  \"fifo_constraint_count\": " << result.fifo_constraint_count
      << ",\n";
  out << "  \"symmetry_break_f0_fifo\": "
      << (result.symmetry_break_f0_fifo ? "true" : "false") << ",\n";
  out << "  \"ratio_num\": " << instance.backward_ratio_num << ",\n";
  out << "  \"ratio_den\": " << instance.backward_ratio_den << ",\n";
  out << "  \"diagnostic\": \"" << JsonEscape(result.diagnostic) << "\",\n";
  if (result.search_stats_enabled) {
    out << "  \"search_stats\": ";
    WriteSearchStatsObject(out, result.search_stats, "  ");
    out << ",\n";
  }
  out << "  \"operations\": [\n";
  for (std::size_t i = 0; i < result.schedule.operations_by_id.size(); ++i) {
    const auto &op = result.schedule.operations_by_id[i];
    const OperationView view = DecodeOperation(instance, op.id);
    out << "    {\"id\": " << op.id.value << ", \"name\": \""
        << OperationName(instance, op.id)
        << "\", \"microbatch\": " << view.microbatch
        << ", \"chain_index\": " << view.chain_index
        << ", \"stage\": " << view.stage << ", \"kind\": \""
        << (view.backward ? "B" : "F") << "\", \"worker\": " << op.worker
        << ", \"start\": " << op.start << ", \"end\": " << op.end
        << ", \"duration\": " << op.duration << "}";
    if (i + 1 != result.schedule.operations_by_id.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::string ToJson(const Instance &instance,
                   const AlternatingOptimizationResult &result) {
  std::ostringstream out;
  out << "{\n";
  if (result.canonical) {
    out << CanonicalResultTopLevelJsonFields(*result.canonical, "  ", true);
  }
  out << "  \"algorithm\": \"" << JsonEscape(result.method) << "\",\n";
  out << "  \"method\": \"" << JsonEscape(result.method) << "\",\n";
  out << "  \"status\": \"" << JsonEscape(result.status) << "\",\n";
  out << "  \"solver_status_raw\": \"" << JsonEscape(result.solver_status_raw)
      << "\",\n";
  out << "  \"proven_optimal\": " << (result.proven_optimal ? "true" : "false")
      << ",\n";
  out << "  \"initial_split\": ";
  WriteTickArray(out, result.initial_split);
  out << ",\n";
  out << "  \"initial_makespan_ticks\": " << result.initial_makespan << ",\n";
  out << "  \"intermediate_partition_only_makespan\": "
      << result.intermediate_partition_only_makespan << ",\n";
  out << "  \"split\": ";
  WriteTickArray(out, result.split);
  out << ",\n";
  WriteWorkerAggregateLayerReport(out, instance, result.split, "  ", true);
  WriteScheduleMetrics(out, instance, result.schedule, "  ", true);
  WriteWorkerOrdersJson(out, instance, result.machine_orders, "  ", true);
  out << "  \"makespan_ticks\": " << result.makespan_ticks << ",\n";
  out << "  \"best_bound_ticks\": " << result.best_bound_ticks << ",\n";
  out << "  \"wall_time_seconds\": " << result.wall_time_seconds << ",\n";
  out << "  \"cp_sat_models_solved\": " << result.cp_sat_models_solved << ",\n";
  out << "  \"alternating_max_rounds\": " << result.alternating_max_rounds
      << ",\n";
  out << "  \"alternating_completed_rounds\": "
      << result.alternating_completed_rounds << ",\n";
  out << "  \"alternating_convergence_reason\": \""
      << JsonEscape(result.alternating_convergence_reason) << "\",\n";
  out << "  \"returned_solution_source\": \""
      << JsonEscape(result.returned_solution_source) << "\",\n";
  out << "  \"fallback_used\": " << (result.fallback_used ? "true" : "false")
      << ",\n";
  out << "  \"fallback_reason\": \"" << JsonEscape(result.fallback_reason)
      << "\",\n";
  out << "  \"diagnostic\": \"" << JsonEscape(result.diagnostic) << "\",\n";
  out << "  \"operations\": [\n";
  for (std::size_t i = 0; i < result.schedule.operations_by_id.size(); ++i) {
    const auto &op = result.schedule.operations_by_id[i];
    const OperationView view = DecodeOperation(instance, op.id);
    out << "    {\"id\": " << op.id.value << ", \"name\": \""
        << OperationName(instance, op.id)
        << "\", \"microbatch\": " << view.microbatch
        << ", \"chain_index\": " << view.chain_index
        << ", \"stage\": " << view.stage << ", \"kind\": \""
        << (view.backward ? "B" : "F") << "\", \"worker\": " << op.worker
        << ", \"start\": " << op.start << ", \"end\": " << op.end
        << ", \"duration\": " << op.duration << "}";
    if (i + 1 != result.schedule.operations_by_id.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::string ToJson(const Instance &instance, const SlackPipeResult &result) {
  std::ostringstream out;
  out << "{\n";
  auto bound_report = result.canonical ? result.canonical->lower_bound_report : ComputeLowerBoundReport(
      instance, result.best_bound_ticks,
      (result.status == "OPTIMAL" || result.status == "FEASIBLE") && result.makespan_ticks > 0
          ? std::optional<Tick>(result.makespan_ticks) : std::nullopt,
      result.effective_split_mode == "global" && !result.worker_balance_constraint.enabled);
  if (result.canonical && (!result.canonical->outcome.feasible.value_or(false) ||
      !result.canonical->outcome.makespan || *result.canonical->outcome.makespan <= 0)) {
    bound_report.effective_relative_gap = std::nullopt;
  }
  out << "  \"lower_bound_report\": " << LowerBoundReportJson(bound_report) << ",\n";
  if (result.canonical) {
    out << CanonicalResultTopLevelJsonFields(*result.canonical, "  ", true);
  }
  out << "  \"algorithm\": \"" << JsonEscape(result.algorithm) << "\",\n";
  out << "  \"initial_split_method\": \""
      << JsonEscape(result.initial_split_method) << "\",\n";
  out << "  \"initial_uniform_split\": [";
  for (std::size_t i = 0; i < result.initial_uniform_split.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.initial_uniform_split[i];
  }
  out << "],\n";
  out << "  \"initial_uniform_bfs_makespan\": "
      << result.initial_uniform_bfs_makespan << ",\n";
  out << "  \"cp_sat_models_solved\": " << result.cp_sat_models_solved << ",\n";
  out << "  \"split_mode\": \"" << JsonEscape(result.split_mode) << "\",\n";
  out << "  \"effective_split_mode\": \""
      << JsonEscape(result.effective_split_mode) << "\",\n";
  out << "  \"move_budget\": " << result.move_budget << ",\n";
  out << "  \"per_stage_delta\": ";
  if (result.per_stage_delta) {
    out << *result.per_stage_delta;
  } else {
    out << "null";
  }
  out << ",\n";
  out << "  \"worker_move_budget\": " << result.worker_move_budget << ",\n";
  out << "  \"per_worker_delta\": ";
  if (result.per_worker_delta) {
    out << *result.per_worker_delta;
  } else {
    out << "null";
  }
  out << ",\n";
  out << "  \"mode_validation_passed\": "
      << (result.mode_validation_passed ? "true" : "false") << ",\n";
  out << "  \"baseline_worker_layers\": [";
  for (std::size_t i = 0; i < result.baseline_worker_layers.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.baseline_worker_layers[i];
  }
  out << "],\n";
  out << "  \"bfs_split\": [";
  for (std::size_t i = 0; i < result.bfs.split.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.bfs.split[i];
  }
  out << "],\n";
  out << "  \"bfs_makespan_ticks\": " << result.bfs.makespan_ticks << ",\n";
  if (result.bfs.method == "heuristic") {
    out << "  \"early_hint\": " << ToJson(instance, result.bfs) << ",\n";
  }
  out << "  \"analytical_global_lower_bound\": "
      << result.analytical_global_lower_bound << ",\n";
  out << "  \"split\": [";
  for (std::size_t i = 0; i < result.split.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.split[i];
  }
  out << "],\n";
  WriteWorkerAggregateLayerReport(out, instance, result.split, "  ", true);
  WriteWorkerBalanceConstraint(out, result.worker_balance_constraint, instance,
                               result.split, "  ", true);
  WritePressurePruningStats(out, result.pressure_pruning_stats, "  ", true);
  WriteScheduleMetrics(out, instance, result.schedule, "  ", true,
                       result.joint_fifo_ordering_effective);
  WriteWorkerOrdersJson(out, instance, result.machine_orders, "  ", true);
  out << "  \"makespan_ticks\": " << result.makespan_ticks << ",\n";
  out << "  \"status\": \"" << JsonEscape(result.status) << "\",\n";
  out << "  \"joint_status\": \"" << JsonEscape(result.joint_status) << "\",\n";
  out << "  \"proven_optimal\": " << (result.proven_optimal ? "true" : "false")
      << ",\n";
  out << "  \"solver_objective_ticks\": " << result.solver_objective_ticks
      << ",\n";
  out << "  \"best_bound_ticks\": " << result.best_bound_ticks << ",\n";
  out << "  \"time_to_first_feasible_seconds\": "
      << result.time_to_first_feasible_seconds << ",\n";
  out << "  \"time_to_best_incumbent_seconds\": "
      << result.time_to_best_incumbent_seconds << ",\n";
  out << "  \"first_feasible_objective\": " << result.first_feasible_objective
      << ",\n";
  out << "  \"incumbent_improvement_count\": "
      << result.incumbent_improvement_count << ",\n";
  out << "  \"incumbent_trace\": [";
  for (std::size_t i = 0; i < result.incumbent_trace.size(); ++i) {
    if (i != 0) out << ", ";
    out << "{\"time_seconds\": " << result.incumbent_trace[i].first
        << ", \"objective\": " << result.incumbent_trace[i].second << "}";
  }
  out << "],\n";
  out << "  \"reference_budget_seconds\": " << result.reference_budget_seconds
      << ",\n";
  out << "  \"reference_solve_seconds\": " << result.reference_solve_seconds
      << ",\n";
  out << "  \"joint_remaining_budget_seconds\": "
      << result.joint_remaining_budget_seconds << ",\n";
  out << "  \"joint_incumbent_method_requested\": \""
      << JsonEscape(result.joint_incumbent_method_requested) << "\",\n";
  out << "  \"joint_incumbent_method_effective\": \""
      << JsonEscape(result.joint_incumbent_method_effective) << "\",\n";
  out << "  \"joint_bfs_incumbent_method_requested\": \""
      << JsonEscape(result.joint_bfs_incumbent_method_requested) << "\",\n";
  out << "  \"joint_bfs_incumbent_method_effective\": \""
      << JsonEscape(result.joint_bfs_incumbent_method_effective) << "\",\n";
  out << "  \"joint_incumbent_source\": \""
      << JsonEscape(result.joint_incumbent_source) << "\",\n";
  out << "  \"joint_incumbent_feasible\": "
      << (result.joint_incumbent_feasible ? "true" : "false") << ",\n";
  out << "  \"joint_incumbent_primary_objective\": "
      << result.joint_incumbent_primary_objective << ",\n";
  out << "  \"joint_incumbent_hybrid_min_slack\": "
      << result.joint_incumbent_hybrid_min_slack << ",\n";
  out << "  \"joint_incumbent_baseline_primary_objective\": "
      << result.joint_incumbent_baseline_primary_objective << ",\n";
  out << "  \"joint_incumbent_baseline_hybrid_min_slack\": "
      << result.joint_incumbent_baseline_hybrid_min_slack << ",\n";
  out << "  \"joint_incumbent_improved_over_baseline\": "
      << (result.joint_incumbent_improved_over_baseline ? "true" : "false")
      << ",\n";
  out << "  \"joint_incumbent_hybrid_stage_scores\": ";
  WriteDoubleArray(out, result.joint_incumbent_hybrid_stage_scores);
  out << ",\n";
  out << "  \"joint_incumbent_hybrid_bottleneck_stages\": ";
  WriteTickArray(out, result.joint_incumbent_hybrid_bottleneck_stages);
  out << ",\n";
  out << "  \"joint_horizon_source\": \""
      << JsonEscape(result.joint_horizon_source) << "\",\n";
  out << "  \"joint_hint_budget_seconds\": " << result.joint_hint_budget_seconds
      << ",\n";
  out << "  \"joint_hint_elapsed_seconds\": "
      << result.joint_hint_elapsed_seconds << ",\n";
  out << "  \"joint_hint_iterations\": " << result.joint_hint_iterations
      << ",\n";
  out << "  \"joint_hint_candidates_generated\": "
      << result.joint_hint_candidates_generated << ",\n";
  out << "  \"joint_hint_candidates_simulated\": "
      << result.joint_hint_candidates_simulated << ",\n";
  out << "  \"joint_hint_partition_moves_accepted\": "
      << result.joint_hint_partition_moves_accepted << ",\n";
  out << "  \"joint_hint_interleaving_moves_accepted\": "
      << result.joint_hint_interleaving_moves_accepted << ",\n";
  out << "  \"joint_hint_deadline_reached\": "
      << (result.joint_hint_deadline_reached ? "true" : "false") << ",\n";
  out << "  \"joint_hint_termination_reason\": \""
      << JsonEscape(result.joint_hint_termination_reason) << "\",\n";
  out << "  \"joint_hints_requested\": "
      << (result.joint_hints_requested ? "true" : "false") << ",\n";
  out << "  \"joint_hints_effective\": "
      << (result.joint_hints_effective ? "true" : "false") << ",\n";
  out << "  \"joint_hint_source\": \"" << JsonEscape(result.joint_hint_source)
      << "\",\n";
  out << "  \"joint_hint_scope\": \"" << JsonEscape(result.joint_hint_scope)
      << "\",\n";
  out << "  \"joint_hint_complete_for_basic_model\": "
      << (result.joint_hint_complete_for_basic_model ? "true" : "false")
      << ",\n";
  out << "  \"joint_hint_complete_for_full_model\": "
      << (result.joint_hint_complete_for_full_model ? "true" : "false")
      << ",\n";
  out << "  \"joint_hinted_layer_variable_count\": "
      << result.joint_hinted_layer_variable_count << ",\n";
  out << "  \"joint_hinted_operation_variable_count\": "
      << result.joint_hinted_operation_variable_count << ",\n";
  out << "  \"joint_hinted_scalar_variable_count\": "
      << result.joint_hinted_scalar_variable_count << ",\n";
  out << "  \"joint_hinted_auxiliary_variable_count\": "
      << result.joint_hinted_auxiliary_variable_count << ",\n";
  out << "  \"joint_hinted_total_variable_count\": "
      << result.joint_hinted_total_variable_count << ",\n";
  out << "  \"joint_auxiliary_variable_count\": "
      << result.joint_auxiliary_variable_count << ",\n";
  out << "  \"joint_fifo_ordering_requested\": "
      << (result.joint_fifo_ordering_requested ? "true" : "false") << ",\n";
  out << "  \"joint_fifo_ordering_effective\": "
      << (result.joint_fifo_ordering_effective ? "true" : "false") << ",\n";
  out << "  \"joint_fifo_constraint_count\": "
      << result.joint_fifo_constraint_count << ",\n";
  out << "  \"joint_fallback_available\": "
      << (result.joint_fallback_available ? "true" : "false") << ",\n";
  out << "  \"joint_fallback_source\": \""
      << JsonEscape(result.joint_fallback_source) << "\",\n";
  out << "  \"joint_solution_source\": \""
      << JsonEscape(result.joint_solution_source) << "\",\n";
  out << "  \"joint_fallback_used\": "
      << (result.joint_fallback_used ? "true" : "false") << ",\n";
  out << "  \"final_worker_layers\": [";
  for (std::size_t i = 0; i < result.final_worker_layers.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.final_worker_layers[i];
  }
  out << "],\n";
  out << "  \"worker_layer_differences\": [";
  for (std::size_t i = 0; i < result.worker_layer_differences.size(); ++i) {
    if (i != 0) out << ", ";
    out << result.worker_layer_differences[i];
  }
  out << "],\n";
  out << "  \"worker_balance_l1\": " << result.worker_balance_l1 << ",\n";
  out << "  \"worker_balance_max_deviation\": "
      << result.worker_balance_max_deviation << ",\n";
  out << "  \"total_wall_time_seconds\": " << result.total_wall_time_seconds
      << ",\n";
  out << "  \"model_build_seconds\": " << result.timing.model_build_seconds
      << ",\n";
  out << "  \"solver_seconds\": " << result.timing.solver_seconds << ",\n";
  out << "  \"ortools_wall_time_seconds\": "
      << result.timing.ortools_wall_time_seconds << ",\n";
  out << "  \"extraction_seconds\": " << result.timing.extraction_seconds
      << ",\n";
  out << "  \"canonicalization_seconds\": "
      << result.timing.canonicalization_seconds << ",\n";
  out << "  \"total_seconds\": " << result.timing.total_seconds << ",\n";
  out << "  \"proven_global_optimal\": "
      << (result.proven_global_optimal ? "true" : "false") << ",\n";
  out << "  \"global_certificate\": \"" << JsonEscape(result.global_certificate)
      << "\",\n";
  out << "  \"diagnostic\": \"" << JsonEscape(result.diagnostic) << "\",\n";
  if (result.search_stats_enabled) {
    out << "  \"search_stats\": ";
    WriteSearchStatsObject(out, result.search_stats, "  ");
    out << ",\n";
  }
  out << "  \"ratio_num\": " << instance.backward_ratio_num << ",\n";
  out << "  \"ratio_den\": " << instance.backward_ratio_den << ",\n";
  out << "  \"operations\": [\n";
  for (std::size_t i = 0; i < result.schedule.operations_by_id.size(); ++i) {
    const auto &op = result.schedule.operations_by_id[i];
    const OperationView view = DecodeOperation(instance, op.id);
    out << "    {\"id\": " << op.id.value << ", \"name\": \""
        << OperationName(instance, op.id)
        << "\", \"microbatch\": " << view.microbatch
        << ", \"chain_index\": " << view.chain_index
        << ", \"stage\": " << view.stage << ", \"kind\": \""
        << (view.backward ? "B" : "F") << "\", \"worker\": " << op.worker
        << ", \"start\": " << op.start << ", \"end\": " << op.end
        << ", \"duration\": " << op.duration << "}";
    if (i + 1 != result.schedule.operations_by_id.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
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

}  // namespace slackpipe
