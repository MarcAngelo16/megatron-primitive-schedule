// Diagnostic only. This executable never enters the Algorithm-1 tuning loop.
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

#include "slackpipe/breadth_first.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/io.h"
#include "slackpipe/octopipe_tuner.h"
#include "slackpipe/one_f_one_b.h"
#include "slackpipe/instance.h"

namespace {
using namespace slackpipe;

void Array(std::ostream& out, const std::vector<Tick>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

void State(std::ostream& out, const OctoPipeState& state, const OctoPipeState& initial) {
  out << "{\"makespan\":" << state.schedule.makespan << ",\"split\":";
  Array(out, state.schedule.split);
  std::vector<Tick> placement;
  for (Index s = 0; s < state.instance.stages; ++s) placement.push_back(state.instance.WorkerForStage(s));
  out << ",\"stage_to_worker\":";
  Array(out, placement);
  const bool predecessors_changed =
      ExtractMachinePredecessors(state.instance, state.schedule.orders) !=
      ExtractMachinePredecessors(initial.instance, initial.schedule.orders);
  out << ",\"predecessor_ordering_changed\":" << std::boolalpha << predecessors_changed
      << ",\"worker_order_changed\":" << (state.schedule.orders != initial.schedule.orders)
      << ",\"num_stages\":" << state.instance.stages
      << ",\"unique_fb_operations\":" << state.schedule.operations_by_id.size()
      << ",\"worker_order_ids\":[";
  for (std::size_t w = 0; w < state.schedule.orders.size(); ++w) {
    if (w) out << ',';
    std::vector<Tick> ids;
    for (auto op : state.schedule.orders[w]) ids.push_back(op.value);
    Array(out, ids);
  }
  out << "]}";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Instance instance;
    instance.microbatches = 8; instance.stages = 8;
    instance.workers = 4; instance.total_layers = 64;
    instance.backward_ratio_num = 2;
    OctoPipeOptions options;
    std::string output_prefix;
    std::string initialization = kUniformBreadthFirstMethod;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--help") {
        std::cout << "Diagnostic only; one independent invocation per phase.\n"
                  << "--B --N --J --L --min-layers --ratio-num --ratio-den --communication\n"
                  << "--initialization uniform-breadth-first|uniform-interleaved-1f1b (CLI aliases accepted)\n"
                  << "--octopipe-candidates-per-iteration (default 64) --output-prefix PATH\n"
                  << "Placement is enabled; no time truncation, iteration loop, or phase fallback.\n";
        return 0;
      }
      if (i + 1 >= argc) throw Error("missing value for " + flag);
      const std::string value = argv[++i];
      if (flag == "--output-prefix") { output_prefix = value; continue; }
      if (flag == "--initialization") {
        initialization = CanonicalizeEvaluationMethodName(value);
        if (initialization != kUniformBreadthFirstMethod &&
            initialization != kUniformInterleavedOneFOneBMethod) {
          throw Error("diagnostic requires an existing fixed uniform initializer");
        }
        continue;
      }
      std::size_t used = 0;
      const auto number = std::stoll(value, &used);
      if (used != value.size()) throw Error("invalid integer for " + flag);
      if (flag == "--B") instance.microbatches = number;
      else if (flag == "--N") instance.stages = number;
      else if (flag == "--J") instance.workers = number;
      else if (flag == "--L") instance.total_layers = number;
      else if (flag == "--min-layers") instance.min_layers = number;
      else if (flag == "--ratio-num") instance.backward_ratio_num = number;
      else if (flag == "--ratio-den") instance.backward_ratio_den = number;
      else if (flag == "--communication") instance.communication_ticks = number;
      else if (flag == "--octopipe-candidates-per-iteration") options.candidates_per_iteration = number;
      else throw Error("unknown diagnostic flag: " + flag);
    }
    instance.Validate();
    const bool breadth_first = initialization == kUniformBreadthFirstMethod;
    const auto orders = breadth_first ? BreadthFirstOrders(instance) : InterleavedOneFOneBOrders(instance);
    const auto initial = EvaluateOctoPipeCandidate(instance, UniformSplit(instance), orders);
    if (!initial) throw Error("invalid diagnostic initializer");
    const auto metrics = ComputeOctoPipeBubbleMetrics(instance, initial->schedule);
    const Tick t_layer = MinimumOctoPipeLayerComputeCost(instance);
    // This executable accepts only the abstract uniform cost model, not profiles.
    const Tick t_layer_min_op = std::min(instance.backward_ratio_den, instance.backward_ratio_num);
    const auto selected = SelectOctoPipePhase(metrics, t_layer);
    std::vector<Tick> busy(instance.workers, 0);
    for (Index w = 0; w < instance.workers; ++w) {
      for (auto id : initial->schedule.orders[w]) {
        const auto& operation = initial->schedule.operations_by_id[id.value];
        busy[w] = CheckedAdd(busy[w], operation.end - operation.start, "diagnostic busy time");
      }
      if (CheckedAdd(busy[w], metrics.bubble[w], "diagnostic timeline") != initial->schedule.makespan) {
        throw Error("busy time plus bubbles must cover each worker timeline");
      }
    }
    std::ostringstream out;
    out << "{\n\"diagnostic_only\":true,\"placement_tuning_enabled\":true,"
        << "\"initialization\":\"" << initialization << "\","
        << "\"generator\":\"" << (breadth_first ? "BreadthFirstOrders" : "InterleavedOneFOneBOrders") << "\","
        << "\"num_microbatches\":" << instance.microbatches
        << ",\"num_workers\":" << instance.workers << ",\"num_layers\":" << instance.total_layers
        << ",\"ratio_num\":" << instance.backward_ratio_num
        << ",\"ratio_den\":" << instance.backward_ratio_den
        << ",\"communication_ticks\":" << instance.communication_ticks
        << ",\"scheduling_candidate_limit\":" << options.candidates_per_iteration
        << ",\n\"initial\":";
    State(out, *initial, *initial);
    out << ",\n\"initial_metrics\":{\"delta_b\":" << metrics.delta_b
        << ",\"t_layer\":" << t_layer << ",\"boundary_bubble\":" << metrics.boundary_sum
        << ",\"residual_bubble\":" << metrics.residual_sum
        << ",\"t_layer_min_op_analysis_only\":" << t_layer_min_op
        << ",\"delta_exceeds_total\":" << (metrics.delta_b > t_layer)
        << ",\"delta_exceeds_min_op\":" << (metrics.delta_b > t_layer_min_op)
        << ",\"boundary_exceeds_residual\":" << (metrics.boundary_sum > metrics.residual_sum)
        << ",\"busy_time\":";
    Array(out, busy);
    out << ",\"boundary\":"; Array(out, metrics.boundary);
    out << ",\"residual\":"; Array(out, metrics.residual);
    out << ",\"total_bubble\":"; Array(out, metrics.bubble);
    out << ",\"equal_worker_busy\":" << std::all_of(busy.begin(), busy.end(), [&](Tick v) { return v == busy.front(); })
        << ",\"equal_worker_bubble\":" << (metrics.delta_b == 0) << "},"
        << "\n\"min_op_selected_phase_analysis_only\":\"" << ToString(SelectOctoPipePhase(metrics, t_layer_min_op)) << "\","
        << "\n\"algorithm1_selected_phase\":\"" << ToString(selected) << "\",\n\"phases\":[\n";
    bool first = true;
    for (auto phase : {OctoPipePhase::kPartition, OctoPipePhase::kFixedStagePlacement,
                       OctoPipePhase::kScheduleFBOnly}) {
      const auto phase_initial = *initial;
      const auto report = DiagnoseOctoPipePhase(phase_initial, phase, options);
      if (!first) out << ",\n";
      first = false;
      const bool improved = report.improving > 0;
      const Tick resulting = improved ? report.best_valid->schedule.makespan : report.initial_makespan;
      const Tick gain = report.initial_makespan - resulting;
      out << "{\"phase\":\"" << ToString(phase) << "\",\"initial_makespan\":" << report.initial_makespan
          << ",\"generated\":" << report.generated << ",\"valid\":" << report.valid
          << ",\"improving\":" << report.improving << ",\"best_valid\":";
      if (report.best_valid) State(out, *report.best_valid, *initial); else out << "null";
      out << ",\"best_improving_makespan\":";
      if (improved) out << resulting; else out << "null";
      out << ",\"best_valid_improvement_ticks\":";
      if (report.best_valid) out << report.initial_makespan - report.best_valid->schedule.makespan;
      else out << "null";
      out << ",\"best_valid_improvement_percent\":";
      if (report.best_valid) out << 100.0 * (report.initial_makespan - report.best_valid->schedule.makespan)
                                     / report.initial_makespan;
      else out << "null";
      out << ",\"resulting_diagnostic_makespan\":" << resulting
          << ",\"improvement_ticks\":" << gain
          << ",\"improvement_percent\":" << 100.0 * gain / report.initial_makespan << '}';
    }
    out << "\n]}\n";
    if (!output_prefix.empty()) WriteTextFile(output_prefix + ".json", out.str());
    std::cout << out.str();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "forced-phase diagnostic failed: " << error.what() << '\n';
    return 2;
  }
}
