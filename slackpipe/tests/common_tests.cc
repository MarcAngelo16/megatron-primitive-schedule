#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/cost_profile.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/interleaving_stats.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/one_f_one_b.h"
#include "slackpipe/operation.h"
#include "slackpipe/plan_export.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/instance.h"
#include "gtest/gtest.h"

namespace {
void ExpectSamePositionFifoRespected(const slackpipe::Instance& instance,
                                    const slackpipe::MachineOrders& orders) {
  ASSERT_EQ(orders.size(), static_cast<std::size_t>(instance.workers));
  for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
    std::vector<slackpipe::Index> last(slackpipe::OperationPositionCount(instance), -1);
    for (auto id : orders[worker]) {
      const auto view = slackpipe::DecodeOperation(instance, id);
      ASSERT_EQ(view.worker, worker);
      EXPECT_TRUE(last[view.chain_index] < view.microbatch);
      last[view.chain_index] = view.microbatch;
    }
  }
}
slackpipe::Instance BaseInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 3;
  instance.workers = 2;
  instance.total_layers = 9;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  return instance;
}

slackpipe::Instance SanityInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 8;
  instance.stages = 4;
  instance.workers = 4;
  instance.total_layers = 8;
  instance.min_layers = 1;
  return instance;
}

std::size_t CountSubstring(const std::string &text, const std::string &needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::vector<slackpipe::OperationId> SameWorkerBasePredecessors(
    const slackpipe::Instance &instance, slackpipe::OperationId id) {
  const slackpipe::OperationView view =
      slackpipe::DecodeOperation(instance, id);
  std::vector<slackpipe::OperationId> predecessors;
  const std::optional<slackpipe::OperationId> data =
      slackpipe::DataPredecessor(instance, id);
  if (data &&
      slackpipe::DecodeOperation(instance, *data).worker == view.worker) {
    predecessors.push_back(*data);
  }
  const std::optional<slackpipe::OperationId> fifo =
      slackpipe::FifoPredecessor(instance, id);
  if (fifo &&
      slackpipe::DecodeOperation(instance, *fifo).worker == view.worker) {
    predecessors.push_back(*fifo);
  }
  return predecessors;
}

std::vector<std::vector<slackpipe::OperationId>> WorkerLocalExtensions(
    const slackpipe::Instance &instance, slackpipe::Index worker) {
  std::vector<slackpipe::OperationId> worker_ops;
  for (slackpipe::Index id = 0; id < instance.OperationCount(); ++id) {
    const slackpipe::OperationId op{id};
    if (slackpipe::DecodeOperation(instance, op).worker == worker) {
      worker_ops.push_back(op);
    }
  }

  std::vector<char> placed(static_cast<std::size_t>(instance.OperationCount()),
                           false);
  std::vector<slackpipe::OperationId> current;
  std::vector<std::vector<slackpipe::OperationId>> extensions;
  std::function<void()> visit = [&]() {
    if (current.size() == worker_ops.size()) {
      extensions.push_back(current);
      return;
    }
    for (slackpipe::OperationId candidate : worker_ops) {
      if (placed[static_cast<std::size_t>(candidate.value)]) continue;
      bool ready = true;
      for (slackpipe::OperationId predecessor :
           SameWorkerBasePredecessors(instance, candidate)) {
        if (!placed[static_cast<std::size_t>(predecessor.value)]) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;
      placed[static_cast<std::size_t>(candidate.value)] = true;
      current.push_back(candidate);
      visit();
      current.pop_back();
      placed[static_cast<std::size_t>(candidate.value)] = false;
    }
  };
  visit();
  return extensions;
}

std::vector<slackpipe::MachineOrders> ExhaustiveMachineOrders(
    const slackpipe::Instance &instance) {
  std::vector<std::vector<std::vector<slackpipe::OperationId>>>
      extensions_by_worker;
  for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
    extensions_by_worker.push_back(WorkerLocalExtensions(instance, worker));
  }

  slackpipe::MachineOrders current(static_cast<std::size_t>(instance.workers));
  std::vector<slackpipe::MachineOrders> orders;
  std::function<void(slackpipe::Index)> choose_worker =
      [&](slackpipe::Index worker) {
        if (worker == instance.workers) {
          orders.push_back(current);
          return;
        }
        for (const std::vector<slackpipe::OperationId> &extension :
             extensions_by_worker[static_cast<std::size_t>(worker)]) {
          current[static_cast<std::size_t>(worker)] = extension;
          choose_worker(worker + 1);
        }
      };
  choose_worker(0);
  return orders;
}

std::vector<slackpipe::SerializedWorkerPredecessorRecord> DerivedRecords(
    const slackpipe::Instance &instance,
    const slackpipe::MachineOrders &orders) {
  const slackpipe::MachinePredecessors predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders);
  std::vector<slackpipe::SerializedWorkerPredecessorRecord> records;
  for (const auto &[id_value, predecessor] : predecessors) {
    const slackpipe::OperationId id{id_value};
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, id);
    records.push_back(slackpipe::SerializedWorkerPredecessorRecord{
        id, slackpipe::OperationName(instance, id), predecessor,
        slackpipe::OperationName(instance, predecessor), view.worker});
  }
  return records;
}

slackpipe::SerializedOperationRecord OperationRecord(
    const slackpipe::Instance &instance, slackpipe::Index microbatch,
    slackpipe::Index position, slackpipe::Tick start, slackpipe::Tick end,
    const std::vector<slackpipe::Tick> &split) {
  const slackpipe::OperationId id =
      slackpipe::EncodeOperation(instance, microbatch, position);
  const slackpipe::OperationView view =
      slackpipe::DecodeOperation(instance, id);
  slackpipe::SerializedOperationRecord record;
  record.id = id;
  record.name = slackpipe::OperationName(instance, id);
  record.microbatch = view.microbatch;
  record.operation_position = view.chain_index;
  record.stage = view.stage;
  record.phase = view.backward ? "B" : "F";
  record.worker = view.worker;
  record.start = start;
  record.end = end;
  record.duration = instance.Duration(view.stage, view.backward, split);
  return record;
}

slackpipe::ResultValidationInput ManualValidatorInput() {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 0;
  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);

  slackpipe::ResultValidationInput input;
  input.instance = instance;
  input.selected_partition = split;
  input.worker_orders = orders;
  input.derived_worker_predecessors = DerivedRecords(instance, orders);
  input.require_serialized_worker_predecessors = true;
  input.serialized_operations =
      std::vector<slackpipe::SerializedOperationRecord>{
          OperationRecord(instance, 0, 0, 0, 2, split),
          OperationRecord(instance, 0, 1, 2, 4, split),
          OperationRecord(instance, 0, 2, 4, 8, split),
          OperationRecord(instance, 0, 3, 8, 12, split)};
  input.require_serialized_operations = true;
  input.reported_makespan = 12;
  input.feasible_claimed = true;
  input.solver_status_raw = "FEASIBLE";
  input.reported_status = "FEASIBLE";
  input.method_name = "test";
  return input;
}

slackpipe::ResultValidationInput EvaluatedValidatorInput(
    const slackpipe::Instance &instance,
    const std::vector<slackpipe::Tick> &split,
    const slackpipe::MachineOrders &orders) {
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  slackpipe::ResultValidationInput input =
      slackpipe::ValidationInputFromSchedule(instance, evaluated.schedule,
                                             "FEASIBLE", "test");
  input.derived_worker_predecessors = DerivedRecords(instance, orders);
  input.require_serialized_worker_predecessors = true;
  input.require_serialized_operations = true;
  return input;
}

slackpipe::Instance OneStageActivationInstance(slackpipe::Index microbatches) {
  slackpipe::Instance instance;
  instance.microbatches = microbatches;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 1;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  return instance;
}

slackpipe::OperationId OneStageOp(const slackpipe::Instance &instance,
                                  slackpipe::Index microbatch, bool backward) {
  return slackpipe::EncodeOperation(instance, microbatch, backward ? 1 : 0);
}
}

TEST(OperationId, UsesCanonicalOperationPositionIndexing) {
  const slackpipe::Instance instance = SanityInstance();
  EXPECT_EQ(slackpipe::OperationPositionCount(instance), 8);

  for (slackpipe::Index n = 0; n < 8; ++n) {
    const slackpipe::OperationId id =
        slackpipe::EncodeOperation(instance, 5, n);
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, id);
    EXPECT_EQ(view.chain_index, n);
    EXPECT_EQ(view.stage, n < 4 ? n : 7 - n);
    EXPECT_EQ(view.worker, view.stage % instance.workers);
  }
}

TEST(ActivationAnalyzer, OneZeroLengthLifetimeConsumesNoCapacity) {
  const slackpipe::Instance instance = OneStageActivationInstance(1);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 0, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, slackpipe::ActivationAnalysisOptions{});

  EXPECT_TRUE(analysis.passed);
  EXPECT_EQ(analysis.total_activation_lifetimes, 1);
  EXPECT_EQ(analysis.zero_length_activation_lifetimes, 1);
  ASSERT_EQ(analysis.per_worker.size(), std::size_t{1});
  EXPECT_EQ(analysis.per_worker[0].peak_live_activation_count, 0);
  EXPECT_EQ(analysis.per_worker[0].peak_activation_units, 0);
  EXPECT_FALSE(analysis.per_worker[0].peak_activation_bytes);
}

TEST(ActivationAnalyzer, ReleaseBeforeAcquireAtEqualTimestamp) {
  const slackpipe::Instance instance = OneStageActivationInstance(2);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 0, true), OneStageOp(instance, 1, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, slackpipe::ActivationAnalysisOptions{});

  ASSERT_EQ(analysis.per_worker.size(), std::size_t{1});
  EXPECT_EQ(analysis.per_worker[0].peak_live_activation_count, 1);
  EXPECT_EQ(analysis.per_worker[0].peak_activation_units, 1);
  EXPECT_EQ(analysis.per_worker[0].time_of_peak, 1);
  EXPECT_EQ(analysis.per_worker[0].activation_unit_time_area, 3);
}

TEST(ActivationAnalyzer, SameProblemDifferentOrdersCanChangePeakMemory) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  const std::vector<slackpipe::Tick> split{1};

  slackpipe::MachineOrders low_memory(1);
  low_memory[0] = {
      OneStageOp(instance, 0, false), OneStageOp(instance, 0, true),
      OneStageOp(instance, 1, false), OneStageOp(instance, 2, false),
      OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult low =
      slackpipe::EvaluateSchedule(instance, split, low_memory);
  ASSERT_TRUE(low.schedule.ok());

  slackpipe::MachineOrders high_memory(1);
  high_memory[0] = {
      OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
      OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
      OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult high =
      slackpipe::EvaluateSchedule(instance, split, high_memory);
  ASSERT_TRUE(high.schedule.ok());

  const slackpipe::ActivationAnalysisResult low_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, low.schedule, slackpipe::ActivationAnalysisOptions{});
  slackpipe::ActivationAnalysisOptions trace_options;
  trace_options.emit_activation_trace = true;
  const slackpipe::ActivationAnalysisResult high_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, high.schedule, trace_options);

  EXPECT_EQ(low_analysis.per_worker[0].peak_activation_units, 1);
  EXPECT_EQ(high_analysis.per_worker[0].peak_activation_units, 2);
  EXPECT_EQ(high_analysis.per_worker[0].peak_live_activation_count, 2);
  EXPECT_FALSE(high_analysis.event_trace.empty());
}

TEST(ActivationAnalyzer, LinearUnitsExplicitUnitsAndBytes) {
  slackpipe::Instance instance = OneStageActivationInstance(3);
  instance.total_layers = 3;
  const std::vector<slackpipe::Tick> split{3};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
               OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::ActivationAnalysisOptions linear;
  linear.activation_units_per_layer = 2;
  linear.activation_bytes_per_unit = 2.5;
  const slackpipe::ActivationAnalysisResult linear_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, linear);
  EXPECT_EQ(linear_analysis.per_worker[0].peak_activation_units, 12);
  ASSERT_TRUE(linear_analysis.per_worker[0].peak_activation_bytes);
  EXPECT_TRUE(std::abs(*linear_analysis.per_worker[0].peak_activation_bytes -
                       30.0) <= 1e-9);
  EXPECT_TRUE(
      std::abs(*linear_analysis.global.maximum_worker_peak_activation_bytes -
               30.0) <= 1e-9);

  slackpipe::ActivationAnalysisOptions explicit_units;
  explicit_units.model = slackpipe::ActivationModel::kExplicitStageUnits;
  explicit_units.explicit_stage_activation_units = {7};
  const slackpipe::ActivationAnalysisResult explicit_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, explicit_units);
  EXPECT_EQ(explicit_analysis.per_worker[0].peak_activation_units, 14);
}

TEST(ActivationAnalyzer, ExplicitCapViolationDiagnosticsAndUniformRatio) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
               OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {1};
  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, options);

  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_FALSE(*analysis.activation_cap_satisfied);
  ASSERT_TRUE(analysis.activation_cap_violation);
  EXPECT_EQ(analysis.activation_cap_violation->worker, 0);
  EXPECT_EQ(analysis.activation_cap_violation->cap_units, 1);
  EXPECT_EQ(analysis.activation_cap_violation->observed_peak_units, 2);
  EXPECT_EQ(analysis.activation_cap_violation->first_violating_timestamp, 2);
  EXPECT_EQ(analysis.activation_cap_violation->live_activations.size(),
            std::size_t{2});
  EXPECT_FALSE(analysis.activation_peak_ratio_to_uniform
                   .maximum_worker_peak_units_ratio);
  EXPECT_FALSE(analysis.activation_peak_ratio_to_uniform.warnings.empty());
}

TEST(ActivationAnalyzer, ResolvesScalarAndVectorCaps) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 2;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> scalar =
      slackpipe::ResolveExplicitActivationCap(instance, {5});
  ASSERT_EQ(scalar.size(), std::size_t{2});
  EXPECT_EQ(scalar[0], 5);
  EXPECT_EQ(scalar[1], 5);

  const std::vector<slackpipe::Tick> vector =
      slackpipe::ResolveExplicitActivationCap(instance, {3, 7});
  ASSERT_EQ(vector.size(), std::size_t{2});
  EXPECT_EQ(vector[0], 3);
  EXPECT_EQ(vector[1], 7);
  EXPECT_THROW(
      (void)slackpipe::ResolveExplicitActivationCap(instance, {1, 2, 3}),
      slackpipe::Error);
}

TEST(ActivationAnalyzer, EnforcedCapRequiresConfiguredCap) {
  const slackpipe::Instance instance = OneStageActivationInstance(1);
  slackpipe::ActivationAnalysisOptions options;
  options.enforce_activation_cap = true;

  EXPECT_THROW(slackpipe::ValidateActivationOptions(instance, options),
               slackpipe::Error);
}

TEST(ActivationCapSolverSupport, ReportsUnsupportedVariableDemandCombination) {
  if (slackpipe::ActivationCapSolverSupport() ==
      slackpipe::ActivationCapSolverSupportLevel::kVariableDemands) {
    return;
  }
  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {10};
  options.enforce_activation_cap = true;
  options.model = slackpipe::ActivationModel::kLinearInStageLayers;

  EXPECT_FALSE(slackpipe::ActivationCapSolverCanEnforce(options, true));
  const std::string reason =
      slackpipe::ActivationCapSolverUnsupportedReason(options, true);
  if (slackpipe::ActivationCapSolverSupport() ==
      slackpipe::ActivationCapSolverSupportLevel::kNone) {
    EXPECT_TRUE(reason.find("cumulative") != std::string::npos);
  } else {
    EXPECT_TRUE(reason.find("variable cumulative") != std::string::npos);
  }
}

TEST(ActivationCapSolverSupport, ReportsNoSupportWithoutOrTools) {
  if (slackpipe::OrToolsCompiledIn()) {
    return;
  }
  EXPECT_FALSE(slackpipe::CumulativeConstraintCompiledIn());
  EXPECT_FALSE(slackpipe::VariableCumulativeDemandCompiledIn());
  EXPECT_EQ(slackpipe::ToString(slackpipe::ActivationCapSolverSupport()),
            std::string("none"));
  EXPECT_EQ(slackpipe::ActivationCapSolverSupportLevelForBuild(),
            std::string("none"));

  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {10};
  options.enforce_activation_cap = true;
  EXPECT_FALSE(slackpipe::ActivationCapSolverCanEnforce(options, true));
  EXPECT_TRUE(slackpipe::ActivationCapSolverUnsupportedReason(options, true)
                  .find("cumulative") != std::string::npos);
}

TEST(ActivationAnalyzer, UniformBaselineDerivationIsStable) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kUniformBaseline;

  const slackpipe::ActivationUniformBaseline first =
      slackpipe::DeriveUniformActivationBaseline(instance, options);
  const slackpipe::ActivationUniformBaseline second =
      slackpipe::DeriveUniformActivationBaseline(instance, options);

  EXPECT_TRUE(first.errors.empty());
  ASSERT_EQ(first.partition.size(), std::size_t{1});
  EXPECT_EQ(first.partition[0], 1);
  ASSERT_EQ(first.cap_units_per_worker.size(), std::size_t{1});
  EXPECT_EQ(first.cap_units_per_worker[0], 0);
  EXPECT_FALSE(first.cap_derivation_hash.empty());
  EXPECT_EQ(first.cap_derivation_hash, second.cap_derivation_hash);
  EXPECT_EQ(first.baseline_run_id, second.baseline_run_id);
}

TEST(ActivationAnalyzer, PeakRatioToUniformUsesNonzeroDenominators) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, slackpipe::ActivationAnalysisOptions{});

  ASSERT_TRUE(analysis.activation_peak_ratio_to_uniform
                  .maximum_worker_peak_units_ratio);
  EXPECT_TRUE(std::abs(*analysis.activation_peak_ratio_to_uniform
                            .maximum_worker_peak_units_ratio -
                       1.0) <= 1e-9);
  ASSERT_EQ(analysis.activation_peak_ratio_to_uniform
                .per_worker_peak_units_ratio.size(),
            std::size_t{2});
  ASSERT_TRUE(
      analysis.activation_peak_ratio_to_uniform.per_worker_peak_units_ratio[0]);
  EXPECT_TRUE(std::abs(*analysis.activation_peak_ratio_to_uniform
                            .per_worker_peak_units_ratio[0] -
                       1.0) <= 1e-9);
}

TEST(OperationId, MapsBackwardOperationsToStageWorkers) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const std::vector<slackpipe::Index> expected_stage{0, 1, 2, 3, 4, 5, 6, 7,
                                                     7, 6, 5, 4, 3, 2, 1, 0};
  const std::vector<slackpipe::Index> expected_worker{0, 1, 2, 3, 0, 1, 2, 3,
                                                      3, 2, 1, 0, 3, 2, 1, 0};
  for (slackpipe::Index n = 0; n < 16; ++n) {
    EXPECT_EQ(slackpipe::StageForOperationPosition(instance, n),
              expected_stage[static_cast<std::size_t>(n)]);
    EXPECT_EQ(slackpipe::WorkerForOperationPosition(instance, n),
              expected_worker[static_cast<std::size_t>(n)]);
  }
}

TEST(OperationId, BackwardOperationWorkerDoesNotUseRawModulo) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const slackpipe::Index n = 8;
  EXPECT_EQ(slackpipe::StageForOperationPosition(instance, n), 7);
  EXPECT_EQ(slackpipe::WorkerForOperationPosition(instance, n), 3);
  EXPECT_EQ(n % instance.workers, 0);
  EXPECT_NE(slackpipe::WorkerForOperationPosition(instance, n),
            n % instance.workers);
}

TEST(BreadthFirst, ProducesOneOperationPerWorkerListEntry) {
  const slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  std::set<slackpipe::Index> seen;
  for (const auto &order : orders) {
    for (slackpipe::OperationId id : order) seen.insert(id.value);
  }
  EXPECT_EQ(static_cast<slackpipe::Index>(seen.size()),
            instance.OperationCount());
}

TEST(BreadthFirst, UsesBreadthFirstWavefrontNotExactInterleavedOneFOneB) {
  slackpipe::Instance instance;
  instance.microbatches = 3;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  ASSERT_EQ(orders.size(), std::size_t{1});

  std::vector<std::string> names;
  for (slackpipe::OperationId id : orders.front()) {
    names.push_back(slackpipe::OperationName(instance, id));
  }
  const std::vector<std::string> exact_1f1b_prefix{"F0(b0)", "F1(b0)", "B1(b0)",
                                                   "F0(b1)"};
  EXPECT_NE(std::vector<std::string>(names.begin(), names.begin() + 4),
            exact_1f1b_prefix);
  EXPECT_TRUE(slackpipe::FindEvaluationMethod("eval-bfs") == nullptr);
}

TEST(InterleavedOneFOneB, HasExactOneFOneBPrefixAndDiffersFromBreadthFirst) {
  slackpipe::Instance instance;
  instance.microbatches = 3;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const slackpipe::MachineOrders orders =
      slackpipe::InterleavedOneFOneBOrders(instance);
  ASSERT_EQ(orders.size(), std::size_t{1});

  std::vector<std::string> names;
  for (slackpipe::OperationId id : orders.front()) {
    names.push_back(slackpipe::OperationName(instance, id));
  }
  const std::vector<std::string> exact_1f1b_prefix{"F0(b0)", "F1(b0)", "B1(b0)",
                                                   "F0(b1)"};
  const std::vector<std::string> actual_prefix(names.begin(),
                                               names.begin() + 4);
  EXPECT_TRUE(actual_prefix == exact_1f1b_prefix);
  EXPECT_NE(orders, slackpipe::BreadthFirstOrders(instance));
}

TEST(InterleavedOneFOneB, ValidatesSmallCyclicCases) {
  std::vector<slackpipe::Instance> cases;
  {
    slackpipe::Instance instance;
    instance.microbatches = 4;
    instance.stages = 2;
    instance.workers = 2;
    instance.total_layers = 16;
    instance.min_layers = 1;
    cases.push_back(instance);
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 8;
    instance.stages = 4;
    instance.workers = 4;
    instance.total_layers = 32;
    instance.min_layers = 1;
    cases.push_back(instance);
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 4;
    instance.stages = 4;
    instance.workers = 2;
    instance.total_layers = 32;
    instance.min_layers = 1;
    cases.push_back(instance);
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 8;
    instance.stages = 8;
    instance.workers = 4;
    instance.total_layers = 64;
    instance.min_layers = 1;
    cases.push_back(instance);
  }

  for (const slackpipe::Instance &instance : cases) {
    const std::vector<slackpipe::Tick> split =
        slackpipe::UniformSplit(instance);
    const slackpipe::MachineOrders orders =
        slackpipe::InterleavedOneFOneBOrders(instance);
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, split, orders);
    ASSERT_TRUE(evaluated.schedule.ok());
    std::set<slackpipe::Index> seen;
    for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
      for (slackpipe::OperationId id :
           orders[static_cast<std::size_t>(worker)]) {
        const slackpipe::OperationView view =
            slackpipe::DecodeOperation(instance, id);
        EXPECT_EQ(view.worker, worker);
        seen.insert(id.value);
      }
    }
    EXPECT_EQ(static_cast<slackpipe::Index>(seen.size()),
              instance.OperationCount());
    ExpectSamePositionFifoRespected(instance, orders);
    slackpipe::ResultValidationInput input =
        slackpipe::ValidationInputFromSchedule(instance, evaluated.schedule,
                                               "FEASIBLE", "1f1b-test");
    input.derived_worker_predecessors = DerivedRecords(instance, orders);
    input.require_serialized_worker_predecessors = true;
    input.require_serialized_operations = true;
    ASSERT_TRUE(input.reported_makespan.has_value());
    EXPECT_EQ(*input.reported_makespan, evaluated.schedule.makespan);
    const slackpipe::ResultValidationResult validation =
        slackpipe::ValidateResult(input);
    EXPECT_TRUE(validation.passed);
    const slackpipe::ResultValidationResult independent_validation =
        slackpipe::ValidateScheduleSolutionIndependent(
            instance, evaluated.schedule, "FEASIBLE", "1f1b-test");
    EXPECT_TRUE(independent_validation.passed);
    const slackpipe::ActivationAnalysisResult activation =
        slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
            instance, evaluated.schedule,
            slackpipe::ActivationAnalysisOptions{});
    EXPECT_TRUE(activation.passed);
    EXPECT_EQ(activation.total_activation_lifetimes,
              instance.microbatches * instance.stages);
  }
}

TEST(Deadline, RemainingTimeIsMonotonicAndNonNegative) {
  slackpipe::Deadline deadline(0.02);
  const double first = deadline.remaining_seconds();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  const double second = deadline.remaining_seconds();
  EXPECT_TRUE(first >= second);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  EXPECT_EQ(deadline.remaining_seconds(), 0.0);
  EXPECT_TRUE(deadline.expired());
}

TEST(Deadline, ChildPhaseCannotExceedParentRemaining) {
  slackpipe::Deadline deadline(1.0);
  const double remaining = deadline.remaining_seconds();
  const double child = deadline.clamp_solver_limit(10.0);
  EXPECT_TRUE(child <= remaining);
  EXPECT_TRUE(child > 0.0);

  slackpipe::Deadline unbounded(0.0);
  EXPECT_FALSE(unbounded.bounded());
  EXPECT_EQ(unbounded.clamp_solver_limit(0.0), 0.0);
  EXPECT_EQ(unbounded.remaining_seconds_for_reporting(), 0.0);
}

TEST(ResultValidator, AcceptsValidHandBfsSchedule) {
  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateResult(ManualValidatorInput());
  EXPECT_TRUE(validation.passed);
  ASSERT_TRUE(validation.reconstructed_makespan.has_value());
  ASSERT_TRUE(validation.serialized_makespan.has_value());
  EXPECT_EQ(*validation.reconstructed_makespan, 12);
  EXPECT_EQ(*validation.serialized_makespan, 12);
  EXPECT_EQ(validation.data_edge_count, 3);
  EXPECT_EQ(validation.fifo_edge_count, 0);
  EXPECT_EQ(validation.worker_edge_count, 1);
}

TEST(ResultValidator, AcceptsValidNonBfsWorkerOrder) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{2};
  slackpipe::MachineOrders orders(1);
  orders[0] = {slackpipe::EncodeOperation(instance, 0, 0),
               slackpipe::EncodeOperation(instance, 1, 0),
               slackpipe::EncodeOperation(instance, 0, 1),
               slackpipe::EncodeOperation(instance, 1, 1)};

  slackpipe::ResultValidationInput input;
  input.instance = instance;
  input.selected_partition = split;
  input.worker_orders = orders;
  input.derived_worker_predecessors = DerivedRecords(instance, orders);
  input.require_serialized_worker_predecessors = true;
  input.reported_makespan = 12;
  input.feasible_claimed = true;

  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateResult(input);
  EXPECT_TRUE(validation.passed);
  EXPECT_EQ(validation.worker_edge_count, 1);
  ASSERT_TRUE(validation.reconstructed_makespan.has_value());
  EXPECT_EQ(*validation.reconstructed_makespan, 12);
}

TEST(ResultValidator, RejectsPartitionErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::nullopt;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code, "partition_missing");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::vector<slackpipe::Tick>{4};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_length_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::vector<slackpipe::Tick>{0, 4};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_empty_stage");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::vector<slackpipe::Tick>{2, 3};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_sum_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.fixed_partition_reference = std::vector<slackpipe::Tick>{3, 1};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_fixed_value_mismatch");
  }
}

TEST(ResultValidator, RejectsOperationInventoryErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().pop_back();
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code, "operation_missing");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().push_back(
        input.worker_orders->front().front());
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_duplicate");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().push_back(
        slackpipe::EncodeOperation(input.instance, 0, 1));
    input.worker_orders->at(1).erase(input.worker_orders->at(1).begin());
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "worker_order_wrong_worker");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().front() = slackpipe::OperationId{999};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code, "operation_unknown");
  }
}

TEST(ResultValidator, RejectsSerializedOperationIdentityAndDurationErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).name = "F0(b0)";
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_name_malformed");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).stage = 0;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_stage_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).phase = "B";
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_phase_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).worker = 0;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_worker_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).duration = 7;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_duration_mismatch");
  }
}

TEST(ResultValidator, RejectsDependencyTimingErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).start = 1;
    input.serialized_operations->at(1).end = 3;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "data_dependency_violation");
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 2;
    instance.stages = 1;
    instance.workers = 1;
    instance.total_layers = 2;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    const std::vector<slackpipe::Tick> split{2};
    const slackpipe::MachineOrders orders =
        slackpipe::BreadthFirstOrders(instance);
    slackpipe::ResultValidationInput input =
        EvaluatedValidatorInput(instance, split, orders);
    input.serialized_operations->at(2).start = 1;
    input.serialized_operations->at(2).end = 3;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "fifo_dependency_violation");
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 2;
    instance.stages = 1;
    instance.workers = 1;
    instance.total_layers = 2;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    const std::vector<slackpipe::Tick> split{2};
    slackpipe::MachineOrders orders(1);
    orders[0] = {slackpipe::EncodeOperation(instance, 0, 0),
                 slackpipe::EncodeOperation(instance, 1, 0),
                 slackpipe::EncodeOperation(instance, 0, 1),
                 slackpipe::EncodeOperation(instance, 1, 1)};
    slackpipe::ResultValidationInput input =
        EvaluatedValidatorInput(instance, split, orders);
    input.serialized_operations->at(1).start = 3;
    input.serialized_operations->at(1).end = 7;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "worker_serialization_violation");
  }
}

TEST(ResultValidator, RejectsCyclesAndBadDerivedPredecessors) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{2};
  slackpipe::MachineOrders cyclic_orders(1);
  cyclic_orders[0] = {slackpipe::EncodeOperation(instance, 0, 1),
                      slackpipe::EncodeOperation(instance, 0, 0)};

  slackpipe::ResultValidationInput cyclic;
  cyclic.instance = instance;
  cyclic.selected_partition = split;
  cyclic.worker_orders = cyclic_orders;
  cyclic.reported_makespan = 6;
  cyclic.feasible_claimed = true;

  const slackpipe::ResultValidationResult first =
      slackpipe::ValidateResult(cyclic);
  const slackpipe::ResultValidationResult second =
      slackpipe::ValidateResult(cyclic);
  EXPECT_EQ(first.error_code, "dependency_cycle");
  ASSERT_EQ(first.cycle_witness.size(), second.cycle_witness.size());
  for (std::size_t i = 0; i < first.cycle_witness.size(); ++i) {
    EXPECT_EQ(first.cycle_witness[i], second.cycle_witness[i]);
  }

  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.derived_worker_predecessors->at(0).predecessor_id =
        input.derived_worker_predecessors->at(0).operation_id;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "malformed_self_edge");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.derived_worker_predecessors->clear();
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "derived_predecessor_mismatch");
  }
}

TEST(ResultValidator, RejectsCommunicationAndObjectiveErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.communication_model = "alpha_beta";
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "unsupported_communication_model");
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 1;
    instance.stages = 2;
    instance.workers = 2;
    instance.total_layers = 4;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    instance.communication_ticks = 3;
    const std::vector<slackpipe::Tick> split{2, 2};
    const slackpipe::MachineOrders orders =
        slackpipe::BreadthFirstOrders(instance);
    slackpipe::ResultValidationInput input =
        EvaluatedValidatorInput(instance, split, orders);
    input.serialized_operations->at(1).start = 2;
    input.serialized_operations->at(1).end = 4;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "data_dependency_violation");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.reported_makespan = 13;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "reported_makespan_mismatch");
  }
}

TEST(ResultValidator, HandlesFallbackAndInfeasibleSemantics) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.solver_status_raw = "UNKNOWN";
    input.reported_status = "FEASIBLE";
    EXPECT_TRUE(slackpipe::ValidateResult(input).passed);
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.solver_status_raw = "NOT_RUN";
    input.reported_status = "FEASIBLE";
    EXPECT_TRUE(slackpipe::ValidateResult(input).passed);
  }
  {
    slackpipe::ResultValidationInput input;
    input.instance = ManualValidatorInput().instance;
    input.feasible_claimed = false;
    const slackpipe::ResultValidationResult validation =
        slackpipe::ValidateResult(input);
    EXPECT_TRUE(validation.passed);
    EXPECT_FALSE(validation.warnings.empty());
  }
  {
    slackpipe::ResultValidationInput input;
    input.instance = ManualValidatorInput().instance;
    input.feasible_claimed = false;
    input.reported_makespan = 12;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "infeasible_has_makespan");
  }
}

TEST(ResultValidator, CrossChecksSmallSchedulesWithDagEvaluatorInTestsOnly) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{2, 2};
  const std::vector<slackpipe::MachineOrders> orders =
      ExhaustiveMachineOrders(instance);

  int checked = 0;
  for (const slackpipe::MachineOrders &order : orders) {
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, split, order);
    if (!evaluated.schedule.ok()) continue;
    slackpipe::ResultValidationInput input =
        slackpipe::ValidationInputFromSchedule(instance, evaluated.schedule,
                                               "FEASIBLE", "cross-check");
    input.derived_worker_predecessors = DerivedRecords(instance, order);
    input.require_serialized_worker_predecessors = true;
    input.require_serialized_operations = true;
    const slackpipe::ResultValidationResult validation =
        slackpipe::ValidateResult(input);
    ASSERT_TRUE(validation.passed);
    ASSERT_TRUE(validation.reconstructed_makespan.has_value());
    EXPECT_EQ(*validation.reconstructed_makespan, evaluated.schedule.makespan);
    if (++checked == 3) break;
  }
  EXPECT_EQ(checked, 3);
}

TEST(Evaluator, AppliesFifoPerOperationPosition) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const slackpipe::MachinePredecessors empty_pred_n;
  const slackpipe::EvaluationResult result =
      slackpipe::EvaluateScheduleWithPredecessors(
          instance, std::vector<slackpipe::Tick>{2, 2}, empty_pred_n);
  ASSERT_TRUE(result.schedule.ok());

  for (slackpipe::Index n = 0; n < 2 * instance.stages; ++n) {
    const slackpipe::OperationId prev =
        slackpipe::EncodeOperation(instance, 0, n);
    const slackpipe::OperationId next =
        slackpipe::EncodeOperation(instance, 1, n);
    const auto &prev_op = result.schedule.operations_by_id[prev.value];
    const auto &next_op = result.schedule.operations_by_id[next.value];
    EXPECT_TRUE(next_op.start >= prev_op.end);
  }

  const slackpipe::OperationId b0_f1 =
      slackpipe::EncodeOperation(instance, 0, 1);
  const slackpipe::OperationId b1_f0 =
      slackpipe::EncodeOperation(instance, 1, 0);
  const auto &op_b0_f1 = result.schedule.operations_by_id[b0_f1.value];
  const auto &op_b1_f0 = result.schedule.operations_by_id[b1_f0.value];
  EXPECT_TRUE(op_b1_f0.start < op_b0_f1.end);
}

TEST(Evaluator, FifoOffAllowsSamePositionWorkerSerialization) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 1;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {
      slackpipe::EncodeOperation(instance, 1, 0),
      slackpipe::EncodeOperation(instance, 0, 0),
      slackpipe::EncodeOperation(instance, 0, 1),
      slackpipe::EncodeOperation(instance, 1, 1),
  };

  EXPECT_THROW((void)slackpipe::ExtractMachinePredecessors(instance, orders), slackpipe::Error);
  EXPECT_FALSE(slackpipe::EvaluateSchedule(instance, split, orders).schedule.ok());
  const slackpipe::MachinePredecessors fifo_off_predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders, false);
  EXPECT_FALSE(fifo_off_predecessors.empty());

  const slackpipe::EvaluationResult fifo_off =
      slackpipe::EvaluateSchedule(instance, split, orders, false);
  ASSERT_TRUE(fifo_off.schedule.ok());
  const slackpipe::ResultValidationResult fifo_off_validation =
      slackpipe::ValidateScheduleSolutionIndependent(
          instance, fifo_off.schedule, "FEASIBLE", "fifo-off-test", false);
  EXPECT_TRUE(fifo_off_validation.passed);
  EXPECT_EQ(fifo_off_validation.fifo_edge_count, 0);
  EXPECT_TRUE(fifo_off_validation.worker_edge_count > 0);

  const slackpipe::ResultValidationResult fifo_on_validation =
      slackpipe::ValidateScheduleSolutionIndependent(
          instance, fifo_off.schedule, "FEASIBLE", "fifo-on-test");
  EXPECT_FALSE(fifo_on_validation.passed);
  EXPECT_EQ(fifo_on_validation.error_code, "worker_order_reverses_fifo");
}

TEST(Evaluator, MachinePredecessorsAreSameWorkerDifferentPosition) {
  const slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::MachinePredecessors predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders);

  for (const auto &[id_value, predecessor] : predecessors) {
    const slackpipe::OperationView current =
        slackpipe::DecodeOperation(instance, slackpipe::OperationId{id_value});
    const slackpipe::OperationView pred =
        slackpipe::DecodeOperation(instance, predecessor);
    EXPECT_EQ(pred.worker, current.worker);
    EXPECT_NE(pred.chain_index, current.chain_index);
    EXPECT_FALSE(slackpipe::IsDataPredecessor(
        instance, predecessor, slackpipe::OperationId{id_value}));
  }
}

TEST(Evaluator, ComputesPrimaryAblationMetrics) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 3;

  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());
  EXPECT_EQ(evaluated.schedule.makespan, 28);

  const slackpipe::ScheduleMetrics metrics =
      slackpipe::ComputeScheduleMetrics(instance, evaluated.schedule);
  EXPECT_EQ(metrics.simulated_iteration_time, 28);
  EXPECT_EQ(metrics.total_useful_work, 32);
  ASSERT_EQ(metrics.per_worker_busy_time.size(), std::size_t{2});
  ASSERT_EQ(metrics.per_worker_idle_time.size(), std::size_t{2});
  EXPECT_EQ(metrics.per_worker_busy_time[0], 16);
  EXPECT_EQ(metrics.per_worker_busy_time[1], 16);
  EXPECT_EQ(metrics.per_worker_idle_time[0], 12);
  EXPECT_EQ(metrics.per_worker_idle_time[1], 12);
  EXPECT_EQ(metrics.max_worker_load, 16);
  EXPECT_EQ(metrics.pipeline_utilization, 32.0 / (2.0 * 28.0));
  ASSERT_TRUE(metrics.communication_blocked_time.has_value());
  EXPECT_TRUE(*metrics.communication_blocked_time > 0);
}

TEST(Evaluator, ProfileUtilizationUsesActualOperationDurations) {
  slackpipe::Instance instance;
  instance.microbatches = instance.stages = instance.workers = 2;
  instance.total_layers = 4;
  instance.profile_forward_slope_ticks = {1, 3};
  instance.profile_backward_slope_ticks = {2, 7};
  instance.profile_forward_bias_ticks = {0, 0};
  instance.profile_backward_bias_ticks = {0, 0};
  const auto evaluated = slackpipe::EvaluateSchedule(
      instance, {2, 2}, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());
  const auto metrics = slackpipe::ComputeScheduleMetrics(instance, evaluated.schedule);
  EXPECT_EQ(metrics.total_useful_work, 52);
  EXPECT_EQ(metrics.per_worker_busy_time[0], 12);
  EXPECT_EQ(metrics.per_worker_busy_time[1], 40);
  EXPECT_EQ(metrics.pipeline_utilization, 52.0 / (2.0 * evaluated.schedule.makespan));
}

TEST(Evaluator, RejectsCombinedPredNCycle) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::OperationId b1_f0 =
      slackpipe::EncodeOperation(instance, 1, 0);
  const slackpipe::OperationId b0_f1 =
      slackpipe::EncodeOperation(instance, 0, 1);

  slackpipe::MachinePredecessors first;
  first.emplace(b0_f1.value, b1_f0);
  EXPECT_TRUE(
      slackpipe::EvaluateScheduleWithPredecessors(instance, split, first)
          .schedule.ok());

  slackpipe::MachinePredecessors second;
  second.emplace(b1_f0.value, b0_f1);
  EXPECT_TRUE(
      slackpipe::EvaluateScheduleWithPredecessors(instance, split, second)
          .schedule.ok());

  slackpipe::MachinePredecessors combined;
  combined.emplace(b0_f1.value, b1_f0);
  combined.emplace(b1_f0.value, b0_f1);
  const slackpipe::EvaluationResult result =
      slackpipe::EvaluateScheduleWithPredecessors(instance, split, combined);
  EXPECT_FALSE(result.schedule.ok());
}

TEST(PredNCandidates, PrunesToNeighboringOperationPositions) {
  const slackpipe::Instance instance = SanityInstance();
  const slackpipe::OperationId target =
      slackpipe::EncodeOperation(instance, 5, 4);
  const std::vector<slackpipe::OperationId> candidates =
      slackpipe::PredNCandidates(instance, target);

  std::set<slackpipe::Index> candidate_microbatches;
  for (slackpipe::OperationId candidate : candidates) {
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, candidate);
    EXPECT_EQ(view.chain_index, 3);
    candidate_microbatches.insert(view.microbatch);
  }

  const std::vector<slackpipe::Index> actual(candidate_microbatches.begin(),
                                             candidate_microbatches.end());
  const std::vector<slackpipe::Index> expected{0, 1, 2, 3, 4, 6, 7};
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(InterleavingStats, CountsAllAdjacentDirectionPairs) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;

  slackpipe::ScheduleSolution schedule;
  schedule.split = {2, 2};
  schedule.makespan = 8;
  schedule.operations_by_id.resize(
      static_cast<std::size_t>(instance.OperationCount()));

  auto set_op = [&](slackpipe::Index microbatch, slackpipe::Index chain_index,
                    slackpipe::Tick start, slackpipe::Tick end) {
    const slackpipe::OperationId id =
        slackpipe::EncodeOperation(instance, microbatch, chain_index);
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, id);
    schedule.operations_by_id[static_cast<std::size_t>(id.value)] =
        slackpipe::ScheduledOperation{id, start, end, end - start, view.worker};
  };

  set_op(0, 0, 0, 1);  // F0
  set_op(0, 1, 1, 2);  // F1 => F_to_F
  set_op(0, 2, 2, 3);  // B1 => F_to_B
  set_op(1, 1, 3, 4);  // F1 => B_to_F
  set_op(1, 2, 4, 5);  // B1 => F_to_B
  set_op(1, 3, 5, 6);  // B0 => B_to_B
  set_op(1, 0, 6, 7);  // F0 => B_to_F
  set_op(0, 3, 7, 8);  // B0 => F_to_B

  const slackpipe::InterleavingStats stats =
      slackpipe::CollectInterleavingStats(instance, schedule);
  ASSERT_EQ(stats.events.size(), std::size_t{7});

  std::vector<std::string> kinds;
  std::vector<bool> switches;
  for (const slackpipe::InterleavingEvent &event : stats.events) {
    kinds.push_back(event.kind);
    switches.push_back(event.is_direction_switch);
    EXPECT_EQ(event.gap_ticks, 0);
  }

  const std::vector<std::string> expected_kinds{
      "F_to_F", "F_to_B", "B_to_F", "F_to_B", "B_to_B", "B_to_F", "F_to_B"};
  const std::vector<bool> expected_switches{false, true, true, true,
                                            false, true, true};
  ASSERT_EQ(kinds.size(), expected_kinds.size());
  ASSERT_EQ(switches.size(), expected_switches.size());
  for (std::size_t i = 0; i < expected_kinds.size(); ++i) {
    EXPECT_EQ(kinds[i], expected_kinds[i]);
    EXPECT_EQ(switches[i], expected_switches[i]);
  }

  EXPECT_EQ(stats.events[0].prev_stage_position, 0);
  EXPECT_EQ(stats.events[0].next_stage_position, 1);
  EXPECT_EQ(stats.events[0].prev_order_index, 0);
  EXPECT_EQ(stats.events[0].next_order_index, 1);
  EXPECT_EQ(stats.events[0].num_worker_ops, 8);
  EXPECT_EQ(stats.events[0].prev_order_fraction, 0.0);
  EXPECT_EQ(stats.events[0].next_order_fraction, 1.0 / 7.0);
  EXPECT_FALSE(stats.events[0].forward_b.has_value());
  EXPECT_FALSE(stats.events[0].backward_b.has_value());

  EXPECT_EQ(stats.events[6].prev_order_index, 6);
  EXPECT_EQ(stats.events[6].next_order_index, 7);
  EXPECT_EQ(stats.events[6].num_worker_ops, 8);
  EXPECT_EQ(stats.events[6].prev_order_fraction, 6.0 / 7.0);
  EXPECT_EQ(stats.events[6].next_order_fraction, 1.0);

  ASSERT_TRUE(stats.events[1].forward_n.has_value());
  EXPECT_EQ(*stats.events[1].forward_n, 1);
  ASSERT_TRUE(stats.events[1].forward_stage_position.has_value());
  EXPECT_EQ(*stats.events[1].forward_stage_position, 1);
  ASSERT_TRUE(stats.events[1].backward_n.has_value());
  EXPECT_EQ(*stats.events[1].backward_n, 1);
  ASSERT_TRUE(stats.events[1].backward_stage_position.has_value());
  EXPECT_EQ(*stats.events[1].backward_stage_position, 1);

  EXPECT_EQ(stats.events[4].kind, "B_to_B");
  EXPECT_FALSE(stats.events[4].is_direction_switch);
  EXPECT_FALSE(stats.events[4].forward_n.has_value());
  EXPECT_FALSE(stats.events[4].backward_n.has_value());

  const std::string csv = slackpipe::ToInterleavingEventsCsv(
      instance, slackpipe::InterleavingRunMetadata{"OK", 8, std::nullopt},
      stats);
  EXPECT_TRUE(csv.find("is_direction_switch") != std::string::npos);
  EXPECT_TRUE(csv.find("prev_stage_position") != std::string::npos);
  EXPECT_TRUE(csv.find("next_stage_position") != std::string::npos);
  EXPECT_TRUE(csv.find("prev_order_index") != std::string::npos);
  EXPECT_TRUE(csv.find("next_order_index") != std::string::npos);
  EXPECT_TRUE(csv.find("num_worker_ops") != std::string::npos);
  EXPECT_TRUE(csv.find("prev_order_fraction") != std::string::npos);
  EXPECT_TRUE(csv.find("next_order_fraction") != std::string::npos);
  EXPECT_TRUE(csv.find("F_to_F") != std::string::npos);
  EXPECT_TRUE(csv.find("F_to_B") != std::string::npos);
  EXPECT_TRUE(csv.find("B_to_F") != std::string::npos);
  EXPECT_TRUE(csv.find("B_to_B") != std::string::npos);

  const std::string summary_csv = slackpipe::ToInterleavingSummaryCsv(
      instance, slackpipe::InterleavingRunMetadata{"OK", 8, std::nullopt},
      stats);
  EXPECT_TRUE(summary_csv.find("prev_stage_position") != std::string::npos);
  EXPECT_TRUE(summary_csv.find("next_stage_position") != std::string::npos);
}

TEST(OrToolsCapabilities, ReportsCumulativeSupportConsistently) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  EXPECT_TRUE(slackpipe::OrToolsCompiledIn());
  EXPECT_TRUE(slackpipe::CumulativeConstraintCompiledIn());
  EXPECT_TRUE(slackpipe::ActivationCapCumulativeConstraintSupported());
  EXPECT_EQ(slackpipe::CumulativeConstraintCompiledIn(),
            slackpipe::ActivationCapCumulativeConstraintSupported());
  EXPECT_EQ(slackpipe::VariableCumulativeDemandCompiledIn(),
            slackpipe::ActivationCapVariableCumulativeDemandSupported());
  EXPECT_EQ(slackpipe::ActivationCapSolverSupportLevelForBuild(),
            slackpipe::ToString(slackpipe::ActivationCapSolverSupport()));
  EXPECT_NE(slackpipe::ActivationCapSolverSupportLevelForBuild(),
            std::string("none"));
}

TEST(MegatronPlanExport, EmitsRequiredSchemaAndStructuredOperations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;

  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const std::string json = slackpipe::ToMegatronSlackPipePlanJson(
      instance, evaluated.schedule, "FEASIBLE");

  EXPECT_NE(json.find("\"schema_version\": \"slackpipe.plan.v1\""),
            std::string::npos);
  EXPECT_NE(json.find("\"num_microbatches\": 4"), std::string::npos);
  EXPECT_NE(json.find("\"num_stages\": 4"), std::string::npos);
  EXPECT_NE(json.find("\"num_workers\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"num_layers\": 8"), std::string::npos);
  EXPECT_NE(json.find("\"layer_split\": [2, 2, 2, 2]"), std::string::npos);
  EXPECT_NE(json.find("\"stage_to_worker\": [0, 1, 0, 1]"), std::string::npos);
  EXPECT_NE(json.find("\"solver_status\": \"FEASIBLE\""), std::string::npos);
  EXPECT_NE(json.find("\"predicted_makespan\": 36"), std::string::npos);
  EXPECT_NE(json.find("\"forward_costs\": [2, 2, 2, 2]"), std::string::npos);
  EXPECT_NE(json.find("\"backward_costs\": [2, 2, 2, 2]"), std::string::npos);
  EXPECT_NE(json.find("{\"kind\": \"F\", \"microbatch\": 0, \"stage\": 0}"),
            std::string::npos);
  EXPECT_NE(json.find("{\"kind\": \"B\", \"microbatch\": 3, \"stage\": 0}"),
            std::string::npos);
  EXPECT_EQ(
      CountSubstring(json, "{\"kind\":"),
      static_cast<std::size_t>(2 * instance.microbatches * instance.stages));
}

TEST(MegatronPlanExport, RejectsDuplicateOrMissingOperations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::ScheduleSolution duplicate = evaluated.schedule;
  duplicate.orders[0].push_back(duplicate.orders[0].front());
  EXPECT_THROW(
      slackpipe::ValidateMegatronSlackPipePlanExport(instance, duplicate),
      slackpipe::Error);

  slackpipe::ScheduleSolution missing = evaluated.schedule;
  missing.orders[0].pop_back();
  EXPECT_THROW(
      slackpipe::ValidateMegatronSlackPipePlanExport(instance, missing),
      slackpipe::Error);
}

TEST(MegatronPlanExport, AtomicWriteLeavesNoFinalFileOnValidationFailure) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const std::filesystem::path final_path =
      std::filesystem::temp_directory_path() /
      "slackpipe_plan_export_failure.plan.json";
  const std::filesystem::path tmp_path = final_path.string() + ".tmp";
  std::error_code ignored;
  std::filesystem::remove(final_path, ignored);
  std::filesystem::remove(tmp_path, ignored);

  slackpipe::ScheduleSolution invalid;
  invalid.split = slackpipe::UniformSplit(instance);
  invalid.orders.assign(static_cast<std::size_t>(instance.workers), {});

  EXPECT_THROW(slackpipe::WriteMegatronSlackPipePlanFile(
                   final_path.string(), instance, invalid, "FEASIBLE"),
               slackpipe::Error);
  EXPECT_FALSE(std::filesystem::exists(final_path));
  EXPECT_FALSE(std::filesystem::exists(tmp_path));
}

TEST(CostProfile, AppliesSharedSlopeAndStageBiasDurations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;

  const std::filesystem::path profile_path =
      std::filesystem::temp_directory_path() /
      "slackpipe_cost_profile_unit.json";
  std::ofstream out(profile_path);
  out << "{\n"
      << "  \"schema_version\": \"slackpipe.cost_profile.v1\",\n"
      << "  \"calibration_partition\": [1, 4, 7, 4],\n"
      << "  \"a_fwd\": 0.25,\n"
      << "  \"a_bwd\": 0.5,\n"
      << "  \"bias_fwd\": [0.01, 0.02, 0.03, 0.04],\n"
      << "  \"bias_bwd\": [0.05, 0.06, 0.07, 0.08]\n"
      << "}\n";
  out.close();
  ASSERT_TRUE(out.good());

  slackpipe::ApplyCostProfileFile(instance, profile_path.string());

  EXPECT_TRUE(instance.HasCostProfile());
  const std::vector<slackpipe::Tick> split{1, 2, 3, 10};
  EXPECT_EQ(instance.Duration(0, false, split), 260);
  EXPECT_EQ(instance.Duration(1, false, split), 520);
  EXPECT_EQ(instance.Duration(2, true, split), 1570);
  EXPECT_EQ(instance.Duration(3, true, split), 5080);
  EXPECT_EQ(instance.cost_profile_units, std::string("microseconds"));

  std::error_code ignored;
  std::filesystem::remove(profile_path, ignored);
}

TEST(CostProfile, AppliesRangePrefixAndRoleBiasDurations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 3;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const std::filesystem::path profile_path =
      std::filesystem::temp_directory_path() /
      "slackpipe_cost_profile_v2_unit.json";
  std::ofstream out(profile_path);
  out << "{\n"
      << "  \"schema_version\": \"slackpipe.cost_profile.v2\",\n"
      << "  \"model_manifest_hash\": \"manifest-sha\",\n"
      << "  \"cost_profile_hash\": \"profile-sha\",\n"
      << "  \"prefix_forward_us\": [0, 100, 300, 700, 800],\n"
      << "  \"prefix_backward_us\": [0, 200, 500, 1100, 1300],\n"
      << "  \"stage_role_bias_us\": {\n"
      << "    \"first\": {\"forward\": 10, \"backward\": 40},\n"
      << "    \"middle\": {\"forward\": 20, \"backward\": 50},\n"
      << "    \"last\": {\"forward\": 30, \"backward\": 60}\n"
      << "  }\n"
      << "}\n";
  out.close();
  ASSERT_TRUE(out.good());

  slackpipe::ApplyCostProfileFile(instance, profile_path.string());

  EXPECT_TRUE(instance.HasCostProfile());
  EXPECT_TRUE(instance.HasRangeCostProfile());
  EXPECT_FALSE(instance.HasAffineCostProfile());
  EXPECT_EQ(instance.cost_profile_hash, std::string("profile-sha"));
  EXPECT_EQ(instance.model_manifest_hash, std::string("manifest-sha"));

  const std::vector<slackpipe::Tick> split{1, 2, 1};
  EXPECT_EQ(instance.StageBeginLayer(1, split), 1);
  EXPECT_EQ(instance.StageEndLayer(1, split), 3);
  EXPECT_EQ(instance.Duration(0, false, split), 110);
  EXPECT_EQ(instance.Duration(1, false, split), 620);
  EXPECT_EQ(instance.Duration(2, true, split), 260);
  EXPECT_EQ(instance.ForwardDuration(1, 1, 3), 620);
  EXPECT_EQ(instance.BackwardDuration(0, 0, 1), 240);

  std::error_code ignored;
  std::filesystem::remove(profile_path, ignored);
}

TEST(MegatronPlanExport, CostProfileMetadataUsesMeasuredDurations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.profile_forward_slope_ticks = {100, 100, 100, 100};
  instance.profile_backward_slope_ticks = {200, 200, 200, 200};
  instance.profile_forward_bias_ticks = {1, 2, 3, 4};
  instance.profile_backward_bias_ticks = {5, 6, 7, 8};
  instance.cost_profile_path = "/tmp/profile.json";
  instance.cost_profile_schema_version = "slackpipe.cost_profile.v1";
  instance.cost_profile_units = "microseconds";

  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());

  const std::string json = slackpipe::ToMegatronSlackPipePlanJson(
      instance, evaluated.schedule, "FEASIBLE");

  EXPECT_NE(json.find("\"kind\": \"cost_profile\""), std::string::npos);
  EXPECT_NE(json.find("\"path\": \"/tmp/profile.json\""), std::string::npos);
  EXPECT_NE(json.find("\"tick_scale\": \"microseconds\""), std::string::npos);
  EXPECT_NE(json.find("\"forward_costs\": [201, 202, 203, 204]"),
            std::string::npos);
  EXPECT_NE(json.find("\"backward_costs\": [405, 406, 407, 408]"),
            std::string::npos);
  EXPECT_EQ(
      CountSubstring(json, "{\"kind\":"),
      static_cast<std::size_t>(2 * instance.microbatches * instance.stages));
}

TEST(MegatronPlanExport, RangeCostProfileEmitsPlanV2RangesAndHashes) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.profile_prefix_forward_ticks = {0,   10,  30,  60, 100,
                                           150, 210, 280, 360};
  instance.profile_prefix_backward_ticks = {0,   20,  60,  120, 200,
                                            300, 420, 560, 720};
  instance.profile_role_forward_bias_ticks = {1, 2, 3};
  instance.profile_role_backward_bias_ticks = {4, 5, 6};
  instance.cost_profile_path = "/tmp/profile-v2.json";
  instance.cost_profile_schema_version = "slackpipe.cost_profile.v2";
  instance.cost_profile_hash = "profile-sha";
  instance.model_manifest_hash = "manifest-sha";
  instance.cost_profile_units = "microseconds";

  const std::vector<slackpipe::Tick> split{1, 2, 1, 4};
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());

  const std::string json = slackpipe::ToMegatronSlackPipePlanJson(
      instance, evaluated.schedule, "FEASIBLE");

  EXPECT_NE(json.find("\"schema_version\": \"slackpipe.plan.v2\""),
            std::string::npos);
  EXPECT_NE(json.find("\"layer_split\": [1, 2, 1, 4]"), std::string::npos);
  EXPECT_NE(json.find("\"stage_layer_ranges\": [{\"begin\": 0, \"end\": 1}, "
                      "{\"begin\": 1, \"end\": 3}, {\"begin\": 3, \"end\": 4}, "
                      "{\"begin\": 4, \"end\": 8}]"),
            std::string::npos);
  EXPECT_NE(json.find("\"model_manifest_hash\": \"manifest-sha\""),
            std::string::npos);
  EXPECT_NE(json.find("\"cost_profile_hash\": \"profile-sha\""),
            std::string::npos);
  EXPECT_NE(
      json.find("\"cost_profile_version\": \"slackpipe.cost_profile.v2\""),
      std::string::npos);
  EXPECT_NE(json.find("\"forward_costs\": [11, 52, 42, 263]"),
            std::string::npos);
  EXPECT_NE(json.find("\"backward_costs\": [24, 105, 85, 526]"),
            std::string::npos);
}
