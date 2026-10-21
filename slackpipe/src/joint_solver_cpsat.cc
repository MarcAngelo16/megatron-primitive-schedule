#include "slackpipe/joint_solver.h"

#if SLACKPIPE_HAVE_ORTOOLS

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <cmath>
#include <utility>
#include <vector>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "slackpipe/activation_cpsat.h"
#include "slackpipe/deadline.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/result_validator.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;
using operations_research::Domain;
using operations_research::sat::BoolVar;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::IntervalVar;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::Model;
using operations_research::sat::NewFeasibleSolutionObserver;
using operations_research::sat::NewSatParameters;
using operations_research::sat::SatParameters;
using operations_research::sat::SolutionIntegerValue;
using operations_research::sat::SolveCpModel;

struct RawOperationTime { OperationId id; Tick start = 0, end = 0; };

double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

[[nodiscard]] Tick ConservativeJointHorizon(const Instance& instance) {
  if (instance.HasRangeCostProfile()) {
    const Tick forward_total = instance.profile_prefix_forward_ticks.back();
    const Tick backward_total = instance.profile_prefix_backward_ticks.back();
    Tick max_bias = 0;
    for (Tick value : instance.profile_role_forward_bias_ticks) {
      max_bias = std::max(max_bias, value);
    }
    for (Tick value : instance.profile_role_backward_bias_ticks) {
      max_bias = std::max(max_bias, value);
    }
    const Tick max_duration =
        CheckedAdd(std::max(forward_total, backward_total), max_bias,
                   "joint range profile horizon duration");
    const Tick compute = CheckedMul(instance.OperationCount(), max_duration,
                                    "joint range profile horizon compute");
    const Tick data_edges = CheckedMul(
        instance.microbatches,
        CheckedAdd(CheckedMul(2, instance.stages, "joint data edge count"), -1,
                   "joint data edge count"),
        "joint data edge count");
    const Tick communication =
        CheckedMul(data_edges, instance.communication_ticks,
                   "joint horizon communication");
    return CheckedAdd(compute, communication, "joint range profile horizon");
  }
  if (instance.HasAffineCostProfile()) {
    Tick max_slope = 0;
    Tick max_bias = 0;
    for (Index stage = 0; stage < instance.stages; ++stage) {
      const auto index = static_cast<std::size_t>(stage);
      max_slope =
          std::max(max_slope, instance.profile_forward_slope_ticks[index]);
      max_slope =
          std::max(max_slope, instance.profile_backward_slope_ticks[index]);
      max_bias = std::max(max_bias, instance.profile_forward_bias_ticks[index]);
      max_bias =
          std::max(max_bias, instance.profile_backward_bias_ticks[index]);
    }
    const Tick max_duration =
        CheckedAdd(CheckedMul(instance.total_layers, max_slope,
                              "joint profile horizon duration"),
                   max_bias, "joint profile horizon duration");
    const Tick compute = CheckedMul(instance.OperationCount(), max_duration,
                                    "joint profile horizon compute");
    const Tick data_edges = CheckedMul(
        instance.microbatches,
        CheckedAdd(CheckedMul(2, instance.stages, "joint data edge count"), -1,
                   "joint data edge count"),
        "joint data edge count");
    const Tick communication =
        CheckedMul(data_edges, instance.communication_ticks,
                   "joint horizon communication");
    return CheckedAdd(compute, communication, "joint profile horizon");
  }
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "joint horizon ratio sum");
  const Tick compute =
      CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                            "joint horizon compute"),
                 ratio_sum, "joint horizon compute");
  const Tick data_edges = CheckedMul(
      instance.microbatches,
      CheckedAdd(CheckedMul(2, instance.stages, "joint data edge count"), -1,
                 "joint data edge count"),
      "joint data edge count");
  const Tick communication = CheckedMul(
      data_edges, instance.communication_ticks, "joint horizon communication");
  return CheckedAdd(compute, communication, "joint horizon");
}

[[nodiscard]] std::string StatusName(CpSolverStatus status) {
  switch (status) {
    case operations_research::sat::OPTIMAL:
      return "OPTIMAL";
    case operations_research::sat::FEASIBLE:
      return "FEASIBLE";
    case operations_research::sat::INFEASIBLE:
      return "INFEASIBLE";
    case operations_research::sat::MODEL_INVALID:
      return "MODEL_INVALID";
    case operations_research::sat::UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNRECOGNIZED";
  }
}

[[nodiscard]] LinearExpr OperationDurationExpr(
    const Instance& instance, const std::vector<IntVar>& layers,
    Index chain_index) {
  const bool backward = chain_index >= instance.stages;
  const Index stage = StageForOperationPosition(instance, chain_index);
  const Tick coefficient =
      instance.HasAffineCostProfile()
          ? (backward
                 ? instance
                       .profile_backward_slope_ticks[static_cast<std::size_t>(
                           stage)]
                 : instance
                       .profile_forward_slope_ticks[static_cast<std::size_t>(
                           stage)])
          : (backward ? instance.backward_ratio_num
                      : instance.backward_ratio_den);
  const Tick bias =
      instance.HasAffineCostProfile()
          ? (backward
                 ? instance
                       .profile_backward_bias_ticks[static_cast<std::size_t>(
                           stage)]
                 : instance.profile_forward_bias_ticks[static_cast<std::size_t>(
                       stage)])
          : 0;
  return coefficient * layers[static_cast<std::size_t>(stage)] + bias;
}

[[nodiscard]] LinearExpr ChainPrefixExpr(const Instance& instance,
                                         const std::vector<IntVar>& layers,
                                         Index chain_index) {
  LinearExpr expr;
  for (Index n = 0; n < chain_index; ++n) {
    expr += OperationDurationExpr(instance, layers, n);
  }
  return expr;
}

[[nodiscard]] LinearExpr ChainSuffixExpr(const Instance& instance,
                                         const std::vector<IntVar>& layers,
                                         Index chain_index) {
  LinearExpr expr;
  for (Index n = chain_index + 1; n < 2 * instance.stages; ++n) {
    expr += OperationDurationExpr(instance, layers, n);
  }
  return expr;
}

[[nodiscard]] MachineOrders ExtractOrdersFromRawTimes(
    const Instance& instance, const std::vector<RawOperationTime>& raw_times) {
  MachineOrders orders(static_cast<std::size_t>(instance.workers));
  for (const RawOperationTime& raw : raw_times) {
    const OperationView view = DecodeOperation(instance, raw.id);
    orders[static_cast<std::size_t>(view.worker)].push_back(raw.id);
  }
  for (auto& order : orders) {
    std::sort(order.begin(), order.end(), [&](OperationId a, OperationId b) {
      const RawOperationTime& raw_a =
          raw_times[static_cast<std::size_t>(a.value)];
      const RawOperationTime& raw_b =
          raw_times[static_cast<std::size_t>(b.value)];
      const OperationView view_a = DecodeOperation(instance, a);
      const OperationView view_b = DecodeOperation(instance, b);
      return std::tuple<Tick, Tick, Index, Index>{raw_a.start, raw_a.end,
                                                  view_a.microbatch,
                                                  view_a.chain_index} <
             std::tuple<Tick, Tick, Index, Index>{
                 raw_b.start, raw_b.end, view_b.microbatch, view_b.chain_index};
    });
  }
  return orders;
}

void ValidateRawNoOverlap(const Instance& instance, const MachineOrders& orders,
                          const std::vector<RawOperationTime>& raw_times) {
  for (Index w = 0; w < instance.workers; ++w) {
    const auto& order = orders[static_cast<std::size_t>(w)];
    for (std::size_t i = 1; i < order.size(); ++i) {
      const RawOperationTime& prev =
          raw_times[static_cast<std::size_t>(order[i - 1].value)];
      const RawOperationTime& current =
          raw_times[static_cast<std::size_t>(order[i].value)];
      if (current.start < prev.end) {
        const OperationView view = DecodeOperation(instance, order[i]);
        std::ostringstream msg;
        msg << "WORKER_OVERLAP worker=" << w << " b=" << view.microbatch
            << " n=" << view.chain_index << " previous_end=" << prev.end
            << " current_start=" << current.start;
        throw Error(msg.str());
      }
    }
  }
}

[[nodiscard]] bool AddReplayCombinationNogood(
    CpModelBuilder& builder, const Instance& instance,
    const std::vector<IntVar>& layers, const std::vector<IntVar>& starts,
    const std::vector<IntVar>& ends, const std::vector<Tick>& split,
    const MachineOrders& orders, Index rejection_index) {
  std::vector<BoolVar> alternatives;
  for (Index s = 0; s < instance.stages; ++s) {
    const Tick rejected_layers = split[static_cast<std::size_t>(s)];
    if (rejected_layers > instance.min_layers) {
      BoolVar less = builder.NewBoolVar().WithName(
          "replay_cap_reject_split_less_" + std::to_string(rejection_index) +
          "_" + std::to_string(s));
      builder
          .AddLessOrEqual(layers[static_cast<std::size_t>(s)],
                          rejected_layers - 1)
          .OnlyEnforceIf(less);
      alternatives.push_back(less);
    }
    if (rejected_layers < instance.total_layers) {
      BoolVar greater = builder.NewBoolVar().WithName(
          "replay_cap_reject_split_greater_" + std::to_string(rejection_index) +
          "_" + std::to_string(s));
      builder
          .AddGreaterOrEqual(layers[static_cast<std::size_t>(s)],
                             rejected_layers + 1)
          .OnlyEnforceIf(greater);
      alternatives.push_back(greater);
    }
  }
  for (const std::vector<OperationId>& worker_order : orders) {
    for (std::size_t i = 1; i < worker_order.size(); ++i) {
      const OperationId prev = worker_order[i - 1];
      const OperationId next = worker_order[i];
      BoolVar inverted = builder.NewBoolVar().WithName(
          "replay_cap_reject_order_" + std::to_string(rejection_index) + "_" +
          std::to_string(prev.value) + "_" + std::to_string(next.value));
      builder
          .AddGreaterOrEqual(starts[static_cast<std::size_t>(prev.value)],
                             ends[static_cast<std::size_t>(next.value)])
          .OnlyEnforceIf(inverted);
      alternatives.push_back(inverted);
    }
  }
  if (alternatives.empty()) return false;
  builder.AddBoolOr(alternatives);
  return true;
}

}  // namespace

SearchResult OptimizeJointSplitAndScheduleCpSat(
    const Instance& instance, const JointOptimizerOptions& options) {
  const auto started = Clock::now();
  instance.Validate();
  if (!std::isfinite(options.time_limit_seconds) || options.time_limit_seconds < 0 ||
      options.num_workers < 1)
    throw Error("invalid CP-SAT budget or worker count");
  for (Index s = 0; s < instance.stages; ++s)
    if (instance.WorkerForStage(s) != s % instance.workers)
      throw Error("cpsat-no-hint requires cyclic placement");
  ValidateActivationOptions(instance, options.activation_options);
  Deadline deadline(options.time_limit_seconds);
  SearchResult result;
  result.method = "cpsat-no-hint";
  result.random_seed = options.random_seed;
  result.requested_seconds = options.time_limit_seconds;
  result.num_workers = options.num_workers;
  result.solver_status_raw = "NOT_RUN";
  const Tick horizon = ConservativeJointHorizon(instance);
  result.conservative_horizon = horizon;
  auto emit = [&](const std::string& phase) {
    if (!options.lifecycle) return;
    LifecycleEvent event;
    event.phase = phase;
    event.elapsed_seconds = Since(started);
    event.configured_solver_limit_seconds = options.time_limit_seconds;
    event.remaining_global_time_seconds = deadline.remaining_seconds_for_reporting();
    event.solver_status = result.solver_status_raw;
    options.lifecycle(event);
  };
  auto finish = [&]() {
    result.wall_time_seconds = result.timing.total_seconds = Since(started);
    if (!result.final_solution_available) {
      result.status = result.solver_status_raw == "UNAVAILABLE" ? "UNAVAILABLE" : "NO_VALID_SOLUTION";
      if (result.no_solution_reason.empty()) result.no_solution_reason = result.solver_status_raw;
      result.proven_optimal = false;
    }
    return result;
  };
  if (options.activation_options.enforce_activation_cap &&
      !ActivationCapSolverCanEnforce(options.activation_options, true)) {
    result.solver_status_raw = "UNAVAILABLE";
    result.no_solution_reason = ActivationCapSolverUnsupportedReason(options.activation_options, true);
    return finish();
  }
  if (deadline.expired()) {
    result.no_solution_reason = "deadline_before_model_build";
    return finish();
  }
  emit("MODEL_BUILD_START");
  const auto model_build_started = Clock::now();
  Index workload_constraint_count = 0;
  CpModelBuilder builder;
  std::vector<IntVar> layers;
  layers.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    layers.push_back(
        builder.NewIntVar(Domain(instance.min_layers, instance.total_layers))
            .WithName("layers_" + std::to_string(s)));
  }
  builder.AddEquality(LinearExpr::Sum(layers), instance.total_layers);
  std::vector<IntVar> cuts;
  cuts.reserve(static_cast<std::size_t>(instance.stages + 1));
  for (Index s = 0; s <= instance.stages; ++s) {
    const Domain domain =
        s == 0 ? Domain(0, 0)
               : (s == instance.stages
                      ? Domain(instance.total_layers, instance.total_layers)
                      : Domain(0, instance.total_layers));
    cuts.push_back(
        builder.NewIntVar(domain).WithName("cut_" + std::to_string(s)));
  }
  for (Index s = 0; s < instance.stages; ++s) {
    builder.AddEquality(cuts[static_cast<std::size_t>(s + 1)],
                        cuts[static_cast<std::size_t>(s)] +
                            layers[static_cast<std::size_t>(s)]);
  }
  const Index op_count = instance.OperationCount();
  std::vector<IntVar> starts;
  std::vector<IntVar> ends;
  std::vector<IntVar> durations;
  std::vector<IntervalVar> intervals;
  starts.reserve(static_cast<std::size_t>(op_count));
  ends.reserve(static_cast<std::size_t>(op_count));
  durations.reserve(static_cast<std::size_t>(op_count));
  intervals.reserve(static_cast<std::size_t>(op_count));

  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    starts.push_back(builder.NewIntVar(Domain(0, horizon))
                         .WithName("start_" + std::to_string(id)));
    ends.push_back(builder.NewIntVar(Domain(0, horizon))
                       .WithName("end_" + std::to_string(id)));
    if (instance.HasRangeCostProfile()) {
      const std::vector<Tick>& prefix =
          view.backward ? instance.profile_prefix_backward_ticks
                        : instance.profile_prefix_forward_ticks;
      const std::vector<Tick>& role_biases =
          view.backward ? instance.profile_role_backward_bias_ticks
                        : instance.profile_role_forward_bias_ticks;
      const Tick bias = role_biases[static_cast<std::size_t>(
          view.stage == 0 ? 0 : (view.stage == instance.stages - 1 ? 2 : 1))];
      const Tick max_duration =
          CheckedAdd(prefix.back(), bias, "joint range maximum duration");
      durations.push_back(builder.NewIntVar(Domain(0, max_duration))
                              .WithName("duration_" + std::to_string(id)));
      IntVar prefix_begin = builder.NewIntVar(Domain(0, prefix.back()))
                                .WithName("prefix_begin_" + std::to_string(id));
      IntVar prefix_end = builder.NewIntVar(Domain(0, prefix.back()))
                              .WithName("prefix_end_" + std::to_string(id));
      builder.AddElement(cuts[static_cast<std::size_t>(view.stage)], prefix,
                         prefix_begin);
      builder.AddElement(cuts[static_cast<std::size_t>(view.stage + 1)], prefix,
                         prefix_end);
      builder.AddEquality(durations[static_cast<std::size_t>(id)],
                          prefix_end - prefix_begin + bias);
    } else {
      const Tick coefficient =
          instance.HasAffineCostProfile()
              ? (view.backward ? instance.profile_backward_slope_ticks
                                     [static_cast<std::size_t>(view.stage)]
                               : instance.profile_forward_slope_ticks
                                     [static_cast<std::size_t>(view.stage)])
              : (view.backward ? instance.backward_ratio_num
                               : instance.backward_ratio_den);
      const Tick bias =
          instance.HasAffineCostProfile()
              ? (view.backward
                     ? instance.profile_backward_bias_ticks
                           [static_cast<std::size_t>(view.stage)]
                     : instance
                           .profile_forward_bias_ticks[static_cast<std::size_t>(
                               view.stage)])
              : 0;
      durations.push_back(
          builder
              .NewIntVar(Domain(
                  CheckedAdd(CheckedMul(instance.min_layers, coefficient,
                                        "joint minimum duration"),
                             bias, "joint minimum duration"),
                  CheckedAdd(CheckedMul(instance.total_layers, coefficient,
                                        "joint maximum duration"),
                             bias, "joint maximum duration")))
              .WithName("duration_" + std::to_string(id)));
      builder.AddEquality(
          durations[static_cast<std::size_t>(id)],
          coefficient * layers[static_cast<std::size_t>(view.stage)] + bias);
    }
    builder.AddEquality(ends[static_cast<std::size_t>(id)],
                        starts[static_cast<std::size_t>(id)] +
                            durations[static_cast<std::size_t>(id)]);
    intervals.push_back(
        builder.NewIntervalVar(starts[static_cast<std::size_t>(id)],
                               durations[static_cast<std::size_t>(id)],
                               ends[static_cast<std::size_t>(id)]));
  }

  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    if (view.chain_index + 1 < 2 * instance.stages) {
      const OperationId next =
          EncodeOperation(instance, view.microbatch, view.chain_index + 1);
      const OperationView next_view = DecodeOperation(instance, next);
      builder.AddGreaterOrEqual(
          starts[static_cast<std::size_t>(next.value)],
          ends[static_cast<std::size_t>(id)] +
              instance.EdgeDelay(view.worker, next_view.worker));
    }
    if (const std::optional<OperationId> fifo =
            FifoPredecessor(instance, OperationId{id})) {
      builder.AddGreaterOrEqual(starts[static_cast<std::size_t>(id)],
                                ends[static_cast<std::size_t>(fifo->value)]);
    }
    if (!instance.HasRangeCostProfile()) {
      builder.AddGreaterOrEqual(
          starts[static_cast<std::size_t>(id)],
          ChainPrefixExpr(instance, layers, view.chain_index));
    }
  }

  std::vector<std::vector<IntervalVar>> worker_intervals(
      static_cast<std::size_t>(instance.workers));
  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    worker_intervals[static_cast<std::size_t>(view.worker)].push_back(
        intervals[static_cast<std::size_t>(id)]);
  }
  for (auto& worker : worker_intervals) builder.AddNoOverlap(worker);

  // Retain the original redundant F_0 FIFO constraints for model parity.
  for (Index b = 1; b < instance.microbatches; ++b) {
    const OperationId prev = EncodeOperation(instance, b - 1, 0);
    const OperationId current = EncodeOperation(instance, b, 0);
    builder.AddGreaterOrEqual(starts[static_cast<std::size_t>(current.value)],
                              ends[static_cast<std::size_t>(prev.value)]);
  }

  IntVar makespan = builder.NewIntVar(Domain(0, horizon)).WithName("makespan");
  for (Index b = 0; b < instance.microbatches; ++b) {
    const OperationId final =
        EncodeOperation(instance, b, 2 * instance.stages - 1);
    builder.AddGreaterOrEqual(makespan,
                              ends[static_cast<std::size_t>(final.value)]);
  }
  builder.AddLessOrEqual(makespan, horizon);
  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    if (!instance.HasRangeCostProfile()) {
      builder.AddLessOrEqual(
          ends[static_cast<std::size_t>(id)] +
              ChainSuffixExpr(instance, layers, view.chain_index),
          makespan);
    }
  }

  if (!instance.HasCostProfile()) {
    const Tick ratio_sum =
        CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                   "joint critical-chain lower bound");
    builder.AddGreaterOrEqual(makespan,
                              CheckedMul(instance.total_layers, ratio_sum,
                                         "joint critical-chain lower bound"));
  }
  // Each duration variable already represents one F/B operation of one
  // microbatch. These redundant capacity bounds also cover variable profiles.
  std::vector<LinearExpr> worker_work(instance.workers);
  for (Index id = 0; id < op_count; ++id) {
    worker_work[DecodeOperation(instance, OperationId{id}).worker] += durations[id];
  }
  for (Index w = 0; w < instance.workers; ++w) {
    builder.AddGreaterOrEqual(makespan, worker_work[w])
        .WithName("worker_workload_bound_" + std::to_string(w));
    ++workload_constraint_count;
  }
  builder.AddGreaterOrEqual(instance.workers * LinearExpr(makespan), LinearExpr::Sum(durations))
      .WithName("global_workload_capacity_bound");
  ++workload_constraint_count;
  result.workload_constraint_count = workload_constraint_count;
  result.fifo_constraint_count = instance.microbatches > 1
      ? (instance.microbatches - 1) * (2 * instance.stages + 1) : 0;
  result.activation_cap_constraints = AddActivationCapacityConstraints(
      builder, instance, options.activation_options, starts, ends, &layers,
      nullptr, horizon, true);
  if (options.activation_options.enforce_activation_cap &&
      !result.activation_cap_constraints.constraints_added) {
    result.solver_status_raw = "UNAVAILABLE";
    result.no_solution_reason = result.activation_cap_constraints.unsupported_reason;
    return finish();
  }
  builder.Minimize(makespan);
  result.timing.model_build_seconds = Since(model_build_started);
  emit("MODEL_BUILD_END");
  while (true) {
    if (deadline.expired()) {
      result.no_solution_reason = "deadline_without_replay_valid_solution";
      return finish();
    }
    auto cp_model = builder.Build();
    result.external_hint_count = cp_model.solution_hint().vars_size();
    if (result.external_hint_count != 0) throw Error("cpsat-no-hint model contains external hints");
    Model model;
    SatParameters parameters;
    parameters.set_num_workers(options.num_workers);
    parameters.set_random_seed(options.random_seed);
    parameters.set_log_search_progress(options.log_search_progress);
    if (deadline.bounded()) parameters.set_max_time_in_seconds(deadline.remaining_seconds());
    model.Add(NewSatParameters(parameters));
    std::mutex observer_mutex;
    model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& response) {
      std::lock_guard<std::mutex> lock(observer_mutex);
      const Tick objective = static_cast<Tick>(response.objective_value());
      const double seconds = Since(started);
      if (result.incumbent_trace.empty()) {
        result.first_feasible_objective = objective;
        result.time_to_first_feasible_seconds = seconds;
      }
      if (result.incumbent_trace.empty() || objective < result.incumbent_trace.back().second) {
        result.incumbent_trace.emplace_back(seconds, objective);
        result.time_to_best_incumbent_seconds = seconds;
        ++result.incumbent_improvement_count;
      }
    }));
    emit("SOLVE_START");
    const auto solve_started = Clock::now();
    result.cp_sat_launched = true;
    const auto response = SolveCpModel(cp_model, &model);
    ++result.cp_sat_models_solved;
    result.timing.solver_seconds += Since(solve_started);
    result.timing.ortools_wall_time_seconds += response.wall_time();
    result.solver_status_raw = StatusName(response.status());
    result.solver_objective_ticks = response.objective_value();
    result.best_bound_ticks = response.best_objective_bound();
    result.solver_bound_available = response.status() != operations_research::sat::MODEL_INVALID;
    result.branches += response.num_branches();
    result.conflicts += response.num_conflicts();
    result.deterministic_time += response.deterministic_time();
    emit("SOLVE_END");
    if (response.status() != operations_research::sat::OPTIMAL &&
        response.status() != operations_research::sat::FEASIBLE) return finish();

    const auto extraction_started = Clock::now();
    std::vector<Tick> split;
    for (const auto& layer : layers) split.push_back(SolutionIntegerValue(response, layer));
    std::vector<RawOperationTime> raw_times(op_count);
    for (Index id = 0; id < op_count; ++id)
      raw_times[id] = {{id}, SolutionIntegerValue(response, starts[id]), SolutionIntegerValue(response, ends[id])};
    auto orders = ExtractOrdersFromRawTimes(instance, raw_times);
    ValidateRawNoOverlap(instance, orders, raw_times);
    result.timing.extraction_seconds += Since(extraction_started);
    const auto replay_started = Clock::now();
    auto checked = EvaluateSchedule(instance, split, orders);
    if (!checked.schedule.ok()) throw Error("CP-SAT extraction failed deterministic DAG replay");
    if (checked.schedule.makespan > static_cast<Tick>(response.objective_value()))
      throw Error("CP-SAT replay exceeds raw objective");
    const auto validation = ValidateScheduleSolutionIndependent(
        instance, checked.schedule, result.solver_status_raw, result.method);
    if (!validation.passed) throw Error("CP-SAT independent validation failed: " + validation.message);
    result.timing.canonicalization_seconds += Since(replay_started);
    // Earliest replay can increase live activations compared with raw solver
    // times. Keep the existing exact partition/order no-good and retry policy.
    if (options.activation_options.enforce_activation_cap &&
        !ActivationScheduleSatisfiesCap(instance, checked.schedule, options.activation_options)) {
      if (!AddReplayCombinationNogood(builder, instance, layers, starts, ends,
                                     split, orders, result.replay_cap_rejections)) {
        result.no_solution_reason = "replay_cap_rejected_no_nogood";
        return finish();
      }
      ++result.replay_cap_rejections;
      continue;
    }
    result.split = std::move(split);
    result.machine_orders = std::move(orders);
    result.schedule = std::move(checked.schedule);
    result.makespan_ticks = result.schedule.makespan;
    result.final_solution_available = true;
    result.status = result.solver_status_raw;
    result.solution_source = "cpsat";
    result.proven_optimal = response.status() == operations_research::sat::OPTIMAL;
    result.optimality_proof_source = result.proven_optimal ? "cpsat_joint_model" : "none";
    return finish();
  }
}

bool IsJointOptimizerAvailable() { return true; }
}  // namespace slackpipe
#endif
