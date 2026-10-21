#include "slackpipe/activation_analyzer.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/operation.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/instance.h"

#ifndef SLACKPIPE_HAVE_ORTOOLS
#define SLACKPIPE_HAVE_ORTOOLS 0
#endif

#ifndef SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED
#define SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED 0
#endif

#ifndef SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED
#define SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED 0
#endif

#ifndef SLACKPIPE_HAVE_CUMULATIVE
#define SLACKPIPE_HAVE_CUMULATIVE SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED
#endif

#ifndef SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND
#define SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND \
  SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED
#endif

namespace slackpipe {
namespace {

using Clock = std::chrono::steady_clock;

enum class EventType {
  kRelease,
  kAcquire,
};

struct Event {
  Tick timestamp = 0;
  EventType type = EventType::kAcquire;
  Index lifetime_index = 0;
};

[[nodiscard]] double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

[[nodiscard]] std::string NormalizeToken(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    if (ch == '-') return '_';
    return static_cast<char>(std::tolower(ch));
  });
  return text;
}

void AddError(ActivationAnalysisResult &result, const std::string &message) {
  result.errors.push_back(message);
  result.passed = false;
}

void AddWarning(ActivationAnalysisResult &result, const std::string &message) {
  result.warnings.push_back(message);
}

[[nodiscard]] std::uint64_t FnvUpdate(std::uint64_t hash,
                                      const std::string &text) {
  for (unsigned char ch : text) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  return hash;
}

[[nodiscard]] std::string StableHashHex(const std::string &text) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = FnvUpdate(hash, text);
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

void AppendTickVector(std::ostringstream &out,
                      const std::vector<Tick> &values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  out << ']';
}

[[nodiscard]] std::string BaselineDerivationSeed(
    const Instance &instance, const ActivationAnalysisOptions &options,
    const std::vector<Tick> &partition,
    const std::string &method_contract_hash) {
  std::ostringstream out;
  out << "activation-analysis-v" << kActivationAnalysisVersion << '|'
      << "B=" << instance.microbatches << '|' << "N=" << instance.stages << '|'
      << "W=" << instance.workers << '|' << "L=" << instance.total_layers << '|'
      << "min=" << instance.min_layers << '|'
      << "ratio=" << instance.backward_ratio_num << '/'
      << instance.backward_ratio_den << '|'
      << "comm=" << instance.communication_ticks << '|'
      << "model=" << ToString(options.model) << '|'
      << "units_per_layer=" << options.activation_units_per_layer << '|'
      << "stage_units=";
  AppendTickVector(out, options.explicit_stage_activation_units);
  out << "|bytes=";
  if (options.activation_bytes_per_unit) {
    out << *options.activation_bytes_per_unit;
  } else {
    out << "null";
  }
  out << "|partition=";
  AppendTickVector(out, partition);
  out << "|method_contract_hash=" << method_contract_hash;
  return out.str();
}

[[nodiscard]] OperationId ForwardOperationId(const Instance &instance,
                                             Index microbatch, Index stage) {
  return EncodeOperation(instance, microbatch, stage);
}

[[nodiscard]] OperationId BackwardOperationId(const Instance &instance,
                                              Index microbatch, Index stage) {
  return EncodeOperation(instance, microbatch, 2 * instance.stages - 1 - stage);
}

[[nodiscard]] Tick DemandForStage(const ActivationAnalysisOptions &options,
                                  const std::vector<Tick> &split, Index stage) {
  switch (options.model) {
    case ActivationModel::kCount:
      return 1;
    case ActivationModel::kLinearInStageLayers:
      return CheckedMul(options.activation_units_per_layer,
                        split[static_cast<std::size_t>(stage)],
                        "activation units");
    case ActivationModel::kExplicitStageUnits:
      return options
          .explicit_stage_activation_units[static_cast<std::size_t>(stage)];
  }
  throw Error("unsupported activation model");
}

[[nodiscard]] std::optional<double> UnitsToBytes(
    Tick units, const std::optional<double> &bytes_per_unit) {
  if (!bytes_per_unit) return std::nullopt;
  return static_cast<double>(units) * *bytes_per_unit;
}

[[nodiscard]] std::optional<double> RatioOrNull(
    Tick numerator, Tick denominator, std::vector<std::string> *warnings,
    const std::string &context) {
  if (denominator == 0) {
    if (warnings != nullptr)
      warnings->push_back(context + " denominator is zero");
    return std::nullopt;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

[[nodiscard]] bool EventLess(const std::vector<ActivationLifetime> &lifetimes,
                             const Event &a, const Event &b) {
  if (a.timestamp != b.timestamp) return a.timestamp < b.timestamp;
  if (a.type != b.type) return a.type == EventType::kRelease;
  const ActivationIdentity &ai =
      lifetimes[static_cast<std::size_t>(a.lifetime_index)].identity;
  const ActivationIdentity &bi =
      lifetimes[static_cast<std::size_t>(b.lifetime_index)].identity;
  return std::tuple<Index, Index, Index>{ai.worker, ai.stage, ai.microbatch} <
         std::tuple<Index, Index, Index>{bi.worker, bi.stage, bi.microbatch};
}

void RecordTrace(ActivationAnalysisResult &result,
                 const ActivationLifetime &lifetime, Tick timestamp,
                 EventType type, Index live_count_after,
                 Tick live_units_after) {
  ActivationEventTraceEntry entry;
  entry.timestamp = timestamp;
  entry.event_type = type == EventType::kRelease ? "release" : "acquire";
  entry.identity = lifetime.identity;
  entry.demand_units = lifetime.demand_units;
  entry.live_activation_count_after_event = live_count_after;
  entry.live_activation_units_after_event = live_units_after;
  result.event_trace.push_back(std::move(entry));
}

[[nodiscard]] ActivationCapViolation BuildViolation(
    const ActivationAnalysisResult &result, Index worker, Tick cap_units,
    Tick observed_units, Tick timestamp,
    const std::vector<std::set<Index>> &active_by_worker) {
  ActivationCapViolation violation;
  violation.worker = worker;
  violation.cap_units = cap_units;
  violation.observed_peak_units = observed_units;
  violation.excess_units = observed_units - cap_units;
  violation.first_violating_timestamp = timestamp;
  for (Index index : active_by_worker[static_cast<std::size_t>(worker)]) {
    violation.live_activations.push_back(
        result.lifetimes[static_cast<std::size_t>(index)].identity);
  }
  std::sort(
      violation.live_activations.begin(), violation.live_activations.end(),
      [](const ActivationIdentity &a, const ActivationIdentity &b) {
        return std::tuple<Index, Index, Index>{a.worker, a.stage,
                                               a.microbatch} <
               std::tuple<Index, Index, Index>{b.worker, b.stage, b.microbatch};
      });
  return violation;
}

[[nodiscard]] std::vector<Tick> ResolveCapOrError(
    const Instance &instance, const ActivationAnalysisOptions &options,
    const std::optional<ActivationUniformBaseline> &uniform_baseline,
    ActivationAnalysisResult &result) {
  if (options.cap_mode == ActivationCapMode::kNone) return {};
  if (options.cap_mode == ActivationCapMode::kExplicit) {
    try {
      result.activation_cap_source = "explicit";
      return ResolveExplicitActivationCap(instance,
                                          options.activation_cap_units);
    } catch (const Error &error) {
      AddError(result, error.what());
      return {};
    }
  }
  if (!uniform_baseline) {
    AddError(result,
             "uniform-baseline activation cap requires a derived baseline");
    return {};
  }
  result.activation_cap_source = "uniform_baseline";
  return uniform_baseline->cap_units_per_worker;
}

void ApplyUniformRatios(
    ActivationAnalysisResult &result,
    const std::optional<ActivationUniformBaseline> &baseline) {
  if (!baseline) return;
  result.activation_peak_ratio_to_uniform.per_worker_peak_units_ratio.clear();
  for (std::size_t w = 0; w < result.per_worker.size(); ++w) {
    const Tick denominator = w < baseline->cap_units_per_worker.size()
                                 ? baseline->cap_units_per_worker[w]
                                 : 0;
    result.activation_peak_ratio_to_uniform.per_worker_peak_units_ratio
        .push_back(
            RatioOrNull(result.per_worker[w].peak_activation_units, denominator,
                        &result.activation_peak_ratio_to_uniform.warnings,
                        "per-worker activation peak ratio"));
  }
  result.activation_peak_ratio_to_uniform.maximum_worker_peak_units_ratio =
      RatioOrNull(result.global.maximum_worker_peak_activation_units,
                  baseline->maximum_worker_peak_units,
                  &result.activation_peak_ratio_to_uniform.warnings,
                  "maximum-worker activation peak ratio");
  result.activation_peak_ratio_to_uniform.global_simultaneous_peak_units_ratio =
      RatioOrNull(
          result.global.peak_simultaneous_activation_units_across_workers,
          baseline->global_simultaneous_peak_units,
          &result.activation_peak_ratio_to_uniform.warnings,
          "global simultaneous activation peak ratio");
}

void PopulateConfigFields(ActivationAnalysisResult &result,
                          const ActivationAnalysisOptions &options) {
  result.activation_analysis_version = kActivationAnalysisVersion;
  result.activation_cap_formulation_version = kActivationCapFormulationVersion;
  result.model = options.model;
  result.activation_units_per_layer = options.activation_units_per_layer;
  result.explicit_stage_activation_units =
      options.explicit_stage_activation_units;
  result.activation_bytes_per_unit = options.activation_bytes_per_unit;
  result.cap_mode = options.cap_mode;
  result.activation_cap_enforced = options.enforce_activation_cap;
  result.activation_cap_enforcement_requested = options.enforce_activation_cap;
  result.activation_cap_solver_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_solver_supported =
      ActivationCapSolverSupport() != ActivationCapSolverSupportLevel::kNone;
  if (options.cap_mode == ActivationCapMode::kNone) {
    result.activation_cap_enforcement_mode = "none";
  } else if (options.enforce_activation_cap) {
    result.activation_cap_enforcement_mode = "posthoc_only";
  } else {
    result.activation_cap_enforcement_mode = "posthoc_only";
  }
}

}  // namespace

std::string ToString(ActivationModel model) {
  switch (model) {
    case ActivationModel::kCount:
      return "count";
    case ActivationModel::kLinearInStageLayers:
      return "linear_in_stage_layers";
    case ActivationModel::kExplicitStageUnits:
      return "explicit_stage_units";
  }
  return "unknown";
}

std::string ToCliString(ActivationModel model) {
  switch (model) {
    case ActivationModel::kCount:
      return "count";
    case ActivationModel::kLinearInStageLayers:
      return "linear-in-stage-layers";
    case ActivationModel::kExplicitStageUnits:
      return "explicit-stage-units";
  }
  return "unknown";
}

ActivationModel ParseActivationModel(const std::string &text) {
  const std::string normalized = NormalizeToken(text);
  if (normalized == "count") return ActivationModel::kCount;
  if (normalized == "linear_in_stage_layers")
    return ActivationModel::kLinearInStageLayers;
  if (normalized == "explicit_stage_units")
    return ActivationModel::kExplicitStageUnits;
  throw Error("unknown activation model: " + text);
}

std::string ToString(ActivationCapMode mode) {
  switch (mode) {
    case ActivationCapMode::kNone:
      return "none";
    case ActivationCapMode::kExplicit:
      return "explicit";
    case ActivationCapMode::kUniformBaseline:
      return "uniform_baseline";
  }
  return "unknown";
}

std::string ToCliString(ActivationCapMode mode) {
  switch (mode) {
    case ActivationCapMode::kNone:
      return "none";
    case ActivationCapMode::kExplicit:
      return "explicit";
    case ActivationCapMode::kUniformBaseline:
      return "uniform-baseline";
  }
  return "unknown";
}

ActivationCapMode ParseActivationCapMode(const std::string &text) {
  const std::string normalized = NormalizeToken(text);
  if (normalized == "none") return ActivationCapMode::kNone;
  if (normalized == "explicit") return ActivationCapMode::kExplicit;
  if (normalized == "uniform_baseline")
    return ActivationCapMode::kUniformBaseline;
  throw Error("unknown activation cap mode: " + text);
}

std::string ToString(ActivationCapSolverSupportLevel level) {
  switch (level) {
    case ActivationCapSolverSupportLevel::kNone:
      return "none";
    case ActivationCapSolverSupportLevel::kFixedDemandsOnly:
      return "fixed_demands_only";
    case ActivationCapSolverSupportLevel::kVariableDemands:
      return "variable_demands";
  }
  return "none";
}

bool ActivationCapCumulativeConstraintSupported() {
  return SLACKPIPE_HAVE_ORTOOLS != 0 && SLACKPIPE_HAVE_CUMULATIVE != 0;
}

bool ActivationCapVariableCumulativeDemandSupported() {
  return ActivationCapCumulativeConstraintSupported() &&
         SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND != 0;
}

ActivationCapSolverSupportLevel ActivationCapSolverSupport() {
  if (!ActivationCapCumulativeConstraintSupported()) {
    return ActivationCapSolverSupportLevel::kNone;
  }
  if (ActivationCapVariableCumulativeDemandSupported()) {
    return ActivationCapSolverSupportLevel::kVariableDemands;
  }
  return ActivationCapSolverSupportLevel::kFixedDemandsOnly;
}

bool ActivationCapSolverCanEnforce(const ActivationAnalysisOptions &options,
                                   bool partition_optimized) {
  if (!options.enforce_activation_cap ||
      options.cap_mode == ActivationCapMode::kNone) {
    return true;
  }
  const ActivationCapSolverSupportLevel support = ActivationCapSolverSupport();
  if (support == ActivationCapSolverSupportLevel::kNone) {
    return false;
  }
  if (options.model == ActivationModel::kLinearInStageLayers &&
      options.activation_units_per_layer <= 0) {
    return false;
  }
  if (options.model == ActivationModel::kExplicitStageUnits) {
    if (options.explicit_stage_activation_units.empty()) {
      return false;
    }
    for (Tick units : options.explicit_stage_activation_units) {
      if (units < 0) {
        return false;
      }
    }
  }
  return !(partition_optimized &&
           options.model == ActivationModel::kLinearInStageLayers &&
           support != ActivationCapSolverSupportLevel::kVariableDemands);
}

std::string ActivationCapSolverUnsupportedReason(
    const ActivationAnalysisOptions &options, bool partition_optimized) {
  if (!options.enforce_activation_cap ||
      options.cap_mode == ActivationCapMode::kNone) {
    return "";
  }
  const ActivationCapSolverSupportLevel support = ActivationCapSolverSupport();
  if (support == ActivationCapSolverSupportLevel::kNone) {
    return "activation cap solver enforcement requires OR-Tools cumulative "
           "constraint support";
  }
  if (options.model == ActivationModel::kLinearInStageLayers &&
      options.activation_units_per_layer <= 0) {
    return "linear activation cap solver enforcement requires a positive "
           "activation-units-per-layer";
  }
  if (options.model == ActivationModel::kExplicitStageUnits) {
    if (options.explicit_stage_activation_units.empty()) {
      return "explicit activation cap solver enforcement requires explicit "
             "stage activation units";
    }
    for (Tick units : options.explicit_stage_activation_units) {
      if (units < 0) {
        return "explicit activation cap solver enforcement requires "
               "non-negative stage activation units";
      }
    }
  }
  if (partition_optimized &&
      options.model == ActivationModel::kLinearInStageLayers &&
      support != ActivationCapSolverSupportLevel::kVariableDemands) {
    return "optimized linear activation demand requires variable cumulative "
           "demand support";
  }
  return "";
}

void ValidateActivationOptions(const Instance &instance,
                               const ActivationAnalysisOptions &options) {
  instance.Validate();
  if (options.activation_units_per_layer < 0) {
    throw Error("--activation-units-per-layer must be non-negative");
  }
  if (options.activation_bytes_per_unit &&
      *options.activation_bytes_per_unit <= 0.0) {
    throw Error("--activation-bytes-per-unit must be greater than zero");
  }
  if (options.model == ActivationModel::kExplicitStageUnits &&
      options.explicit_stage_activation_units.size() !=
          static_cast<std::size_t>(instance.stages)) {
    throw Error("--activation-stage-units requires exactly N values");
  }
  for (Tick units : options.explicit_stage_activation_units) {
    if (units < 0)
      throw Error("--activation-stage-units values must be non-negative");
  }
  for (Tick cap : options.activation_cap_units) {
    if (cap < 0)
      throw Error("--activation-cap-units values must be non-negative");
  }
  if (options.cap_mode == ActivationCapMode::kExplicit &&
      options.activation_cap_units.empty()) {
    throw Error(
        "--activation-cap-mode explicit requires --activation-cap-units");
  }
  if (options.cap_mode == ActivationCapMode::kNone &&
      !options.activation_cap_units.empty()) {
    throw Error("--activation-cap-units requires an activation cap mode");
  }
  if (options.enforce_activation_cap &&
      options.cap_mode == ActivationCapMode::kNone) {
    throw Error("--enforce-activation-cap requires an activation cap mode");
  }
  if (!options.activation_cap_units.empty()) {
    (void)ResolveExplicitActivationCap(instance, options.activation_cap_units);
  }
}

std::vector<Tick> ResolveExplicitActivationCap(
    const Instance &instance, const std::vector<Tick> &cap_units) {
  if (cap_units.empty()) return {};
  if (cap_units.size() == 1) {
    return std::vector<Tick>(static_cast<std::size_t>(instance.workers),
                             cap_units.front());
  }
  if (cap_units.size() != static_cast<std::size_t>(instance.workers)) {
    throw Error("--activation-cap-units must be a scalar or a W-length vector");
  }
  return cap_units;
}

std::vector<Tick> ResolveActivationCapUnits(
    const Instance &instance, const ActivationAnalysisOptions &options) {
  if (options.cap_mode == ActivationCapMode::kNone) {
    return {};
  }
  if (options.cap_mode == ActivationCapMode::kExplicit) {
    return ResolveExplicitActivationCap(instance, options.activation_cap_units);
  }
  if (options.uniform_baseline) {
    return options.uniform_baseline->cap_units_per_worker;
  }
  return DeriveUniformActivationBaseline(instance, options)
      .cap_units_per_worker;
}

bool ActivationScheduleSatisfiesCap(const Instance &instance,
                                    const ScheduleSolution &schedule,
                                    const ActivationAnalysisOptions &options) {
  if (options.cap_mode == ActivationCapMode::kNone) {
    return true;
  }
  ActivationAnalysisOptions check_options = options;
  check_options.enforce_activation_cap = false;
  check_options.emit_activation_trace = false;
  const ActivationAnalysisResult analysis =
      AnalyzeActivationMemoryWithUniformBaseline(instance, schedule,
                                                 check_options, false);
  return analysis.passed && analysis.activation_cap_satisfied.value_or(false);
}

void ApplyActivationCapConstraintMetadata(
    ActivationAnalysisResult &analysis,
    const ActivationCapConstraintMetadata &metadata,
    const std::string &enforcement_mode) {
  analysis.activation_cap_solver_supported = metadata.solver_supported;
  analysis.activation_cap_solver_support_level =
      metadata.model_support_level.empty()
          ? ToString(ActivationCapSolverSupport())
          : metadata.model_support_level;
  analysis.activation_cap_constraints_added = metadata.constraints_added;
  analysis.activation_retained_interval_count =
      metadata.retained_interval_count;
  analysis.activation_cumulative_constraint_count =
      metadata.cumulative_constraint_count;
  analysis.activation_variable_demand_count = metadata.variable_demand_count;
  analysis.activation_fixed_demand_count = metadata.fixed_demand_count;
  analysis.activation_workers_with_constraints =
      metadata.workers_with_constraints;
  analysis.activation_cap_unsupported_reason = metadata.unsupported_reason;
  analysis.activation_constraint_build_runtime_seconds =
      metadata.build_runtime_seconds;
  analysis.incumbent_rejected_for_activation_cap =
      metadata.incumbent_rejected_for_activation_cap;
  analysis.activation_cap_enforced_by_enumeration =
      metadata.enforced_by_enumeration;
  analysis.activation_cap_enforced_in_solver = metadata.constraints_added;
  if (!enforcement_mode.empty()) {
    analysis.activation_cap_enforcement_mode = enforcement_mode;
  } else if (analysis.cap_mode == ActivationCapMode::kNone) {
    analysis.activation_cap_enforcement_mode = "none";
  } else if (metadata.constraints_added) {
    analysis.activation_cap_enforcement_mode = "solver";
  } else if (metadata.enforced_by_enumeration) {
    analysis.activation_cap_enforcement_mode = "exact_enumeration";
  } else if (!metadata.unsupported_reason.empty()) {
    analysis.activation_cap_enforcement_mode = "unsupported";
  }
  if ((metadata.constraints_added || metadata.enforced_by_enumeration) &&
      analysis.activation_cap_satisfied) {
    analysis.activation_model_validation_agreement =
        analysis.activation_cap_satisfied.value_or(false);
    if (!analysis.activation_cap_satisfied.value_or(false)) {
      analysis.activation_model_disagreement_details =
          metadata.enforced_by_enumeration
              ? "enumeration-side activation cap filtering accepted a "
                "schedule, but the independent analyzer found a cap violation"
              : "solver-side activation cap constraints were added, but the "
                "independent analyzer found a cap violation";
    }
  }
}

ActivationAnalysisResult AnalyzeActivationMemory(
    const Instance &instance, const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &options,
    const std::optional<ActivationUniformBaseline> &uniform_baseline,
    bool activation_cap_enforced_in_solver) {
  ActivationAnalysisResult result;
  PopulateConfigFields(result, options);
  result.activation_cap_enforced_in_solver = activation_cap_enforced_in_solver;
  result.per_worker.resize(static_cast<std::size_t>(instance.workers));
  for (Index w = 0; w < instance.workers; ++w) {
    result.per_worker[static_cast<std::size_t>(w)].worker = w;
  }

  try {
    ValidateActivationOptions(instance, options);
    ValidateSplit(instance, schedule.split);
  } catch (const Error &error) {
    AddError(result, error.what());
    return result;
  }

  if (!schedule.ok()) {
    AddError(result, "activation analysis requires a valid schedule");
    for (const std::string &error : schedule.validation_errors)
      AddError(result, error);
    return result;
  }
  if (schedule.operations_by_id.size() !=
      static_cast<std::size_t>(instance.OperationCount())) {
    AddError(result, "activation analysis missing operation timing records");
    return result;
  }

  const std::optional<ActivationUniformBaseline> effective_baseline =
      uniform_baseline ? uniform_baseline : options.uniform_baseline;
  if (effective_baseline) {
    result.activation_baseline_run_id = effective_baseline->baseline_run_id;
    result.activation_cap_derivation_hash =
        effective_baseline->cap_derivation_hash;
    result.activation_baseline_partition = effective_baseline->partition;
    result.activation_baseline_method_contract_hash =
        effective_baseline->method_contract_hash;
    result.activation_cap_derivation_runtime_seconds =
        effective_baseline->derivation_runtime_seconds;
    for (const std::string &warning : effective_baseline->warnings)
      AddWarning(result, warning);
    for (const std::string &error : effective_baseline->errors)
      AddError(result, error);
  }

  result.activation_cap_units_per_worker =
      ResolveCapOrError(instance, options, effective_baseline, result);
  if (result.activation_cap_units_per_worker.empty()) {
    result.activation_cap_satisfied =
        options.cap_mode == ActivationCapMode::kNone
            ? std::optional<bool>{}
            : std::optional<bool>{false};
  } else {
    result.activation_cap_satisfied = true;
    for (std::size_t w = 0; w < result.per_worker.size(); ++w) {
      result.per_worker[w].cap_units =
          result.activation_cap_units_per_worker[w];
      result.per_worker[w].cap_satisfied = true;
    }
  }

  std::vector<Event> events;
  events.reserve(
      static_cast<std::size_t>(instance.microbatches * instance.stages * 2));

  for (Index b = 0; b < instance.microbatches; ++b) {
    for (Index s = 0; s < instance.stages; ++s) {
      const OperationId forward_id = ForwardOperationId(instance, b, s);
      const OperationId backward_id = BackwardOperationId(instance, b, s);
      if (forward_id.value < 0 || backward_id.value < 0 ||
          forward_id.value >= instance.OperationCount() ||
          backward_id.value >= instance.OperationCount()) {
        AddError(result, "activation analysis produced invalid operation id");
        continue;
      }
      const ScheduledOperation &forward =
          schedule.operations_by_id[static_cast<std::size_t>(forward_id.value)];
      const ScheduledOperation &backward =
          schedule
              .operations_by_id[static_cast<std::size_t>(backward_id.value)];
      if (forward.id != forward_id || backward.id != backward_id) {
        AddError(result, "activation analysis operation record id mismatch");
        continue;
      }
      const OperationView forward_view = DecodeOperation(instance, forward_id);
      const OperationView backward_view =
          DecodeOperation(instance, backward_id);
      const Index expected_worker = instance.WorkerForStage(s);
      if (forward.worker != backward.worker ||
          forward_view.worker != backward_view.worker ||
          forward.worker != expected_worker) {
        std::ostringstream msg;
        msg << "activation paired operations map to different workers for b="
            << b << " stage=" << s;
        AddError(result, msg.str());
        continue;
      }
      const Tick demand = DemandForStage(options, schedule.split, s);
      if (demand < 0) {
        std::ostringstream msg;
        msg << "activation demand is negative for b=" << b << " stage=" << s;
        AddError(result, msg.str());
        continue;
      }
      ActivationLifetime lifetime;
      lifetime.identity.microbatch = b;
      lifetime.identity.stage = s;
      lifetime.identity.worker = expected_worker;
      lifetime.identity.forward_operation_id = forward_id;
      lifetime.identity.backward_operation_id = backward_id;
      lifetime.start = forward.end;
      lifetime.end = backward.start;
      lifetime.demand_units = demand;
      if (lifetime.end < lifetime.start) {
        std::ostringstream msg;
        msg << "activation backward starts before forward completes for b=" << b
            << " stage=" << s;
        AddError(result, msg.str());
      }
      lifetime.zero_length = lifetime.end == lifetime.start;
      const Index lifetime_index = static_cast<Index>(result.lifetimes.size());
      result.lifetimes.push_back(lifetime);
      ++result.total_activation_lifetimes;
      if (lifetime.zero_length) {
        ++result.zero_length_activation_lifetimes;
      } else if (lifetime.end > lifetime.start) {
        events.push_back(
            Event{lifetime.start, EventType::kAcquire, lifetime_index});
        events.push_back(
            Event{lifetime.end, EventType::kRelease, lifetime_index});
      }
    }
  }

  if (!result.passed) return result;

  std::sort(events.begin(), events.end(), [&](const Event &a, const Event &b) {
    return EventLess(result.lifetimes, a, b);
  });

  std::vector<Index> live_count(static_cast<std::size_t>(instance.workers), 0);
  std::vector<Tick> live_units(static_cast<std::size_t>(instance.workers), 0);
  std::vector<std::set<Index>> active_by_worker(
      static_cast<std::size_t>(instance.workers));
  Tick previous_time = events.empty() ? 0 : events.front().timestamp;

  for (std::size_t i = 0; i < events.size();) {
    const Tick timestamp = events[i].timestamp;
    const Tick elapsed = timestamp - previous_time;
    if (elapsed < 0) {
      AddError(result, "activation events are not sorted by time");
      return result;
    }
    if (elapsed > 0) {
      for (Index w = 0; w < instance.workers; ++w) {
        ActivationWorkerMetrics &metrics =
            result.per_worker[static_cast<std::size_t>(w)];
        metrics.activation_count_time_area = CheckedAdd(
            metrics.activation_count_time_area,
            CheckedMul(
                static_cast<Tick>(live_count[static_cast<std::size_t>(w)]),
                elapsed, "activation count time area"),
            "activation count time area");
        metrics.activation_unit_time_area =
            CheckedAdd(metrics.activation_unit_time_area,
                       CheckedMul(live_units[static_cast<std::size_t>(w)],
                                  elapsed, "activation unit time area"),
                       "activation unit time area");
      }
    }
    previous_time = timestamp;

    while (i < events.size() && events[i].timestamp == timestamp) {
      const Event &event = events[i];
      const ActivationLifetime &lifetime =
          result.lifetimes[static_cast<std::size_t>(event.lifetime_index)];
      const Index worker = lifetime.identity.worker;
      if (event.type == EventType::kRelease) {
        if (live_count[static_cast<std::size_t>(worker)] <= 0 ||
            live_units[static_cast<std::size_t>(worker)] <
                lifetime.demand_units) {
          AddError(
              result,
              "activation release encountered without matching live state");
          return result;
        }
        --live_count[static_cast<std::size_t>(worker)];
        live_units[static_cast<std::size_t>(worker)] =
            CheckedAdd(live_units[static_cast<std::size_t>(worker)],
                       -lifetime.demand_units, "activation live units");
        active_by_worker[static_cast<std::size_t>(worker)].erase(
            event.lifetime_index);
      } else {
        ++live_count[static_cast<std::size_t>(worker)];
        live_units[static_cast<std::size_t>(worker)] =
            CheckedAdd(live_units[static_cast<std::size_t>(worker)],
                       lifetime.demand_units, "activation live units");
        active_by_worker[static_cast<std::size_t>(worker)].insert(
            event.lifetime_index);
      }

      if (options.emit_activation_trace) {
        RecordTrace(result, lifetime, timestamp, event.type,
                    live_count[static_cast<std::size_t>(worker)],
                    live_units[static_cast<std::size_t>(worker)]);
      }
      ++i;
    }

    Index simultaneous_count = 0;
    Tick simultaneous_units = 0;
    for (Index w = 0; w < instance.workers; ++w) {
      ActivationWorkerMetrics &metrics =
          result.per_worker[static_cast<std::size_t>(w)];
      const Index count = live_count[static_cast<std::size_t>(w)];
      const Tick units = live_units[static_cast<std::size_t>(w)];
      if (count > metrics.peak_live_activation_count) {
        metrics.peak_live_activation_count = count;
      }
      if (units > metrics.peak_activation_units) {
        metrics.peak_activation_units = units;
        metrics.time_of_peak = timestamp;
      }
      if (metrics.cap_units && units > *metrics.cap_units) {
        metrics.cap_satisfied = false;
        result.activation_cap_satisfied = false;
        if (!result.activation_cap_violation) {
          result.activation_cap_violation =
              BuildViolation(result, w, *metrics.cap_units, units, timestamp,
                             active_by_worker);
        }
      }
      simultaneous_count += count;
      simultaneous_units =
          CheckedAdd(simultaneous_units, units, "global activation units");
    }
    if (simultaneous_count >
        result.global.peak_simultaneous_activation_count_across_workers) {
      result.global.peak_simultaneous_activation_count_across_workers =
          simultaneous_count;
    }
    if (simultaneous_units >
        result.global.peak_simultaneous_activation_units_across_workers) {
      result.global.peak_simultaneous_activation_units_across_workers =
          simultaneous_units;
      result.global.time_of_global_peak = timestamp;
    }
  }

  for (ActivationWorkerMetrics &metrics : result.per_worker) {
    metrics.peak_activation_bytes = UnitsToBytes(
        metrics.peak_activation_units, options.activation_bytes_per_unit);
    result.global.maximum_worker_peak_activation_count =
        std::max(result.global.maximum_worker_peak_activation_count,
                 metrics.peak_live_activation_count);
    result.global.maximum_worker_peak_activation_units =
        std::max(result.global.maximum_worker_peak_activation_units,
                 metrics.peak_activation_units);
    result.global.total_activation_unit_time_area = CheckedAdd(
        result.global.total_activation_unit_time_area,
        metrics.activation_unit_time_area, "total activation unit time area");
    if (schedule.makespan > 0) {
      metrics.average_live_activation_count =
          static_cast<double>(metrics.activation_count_time_area) /
          static_cast<double>(schedule.makespan);
      metrics.average_activation_units =
          static_cast<double>(metrics.activation_unit_time_area) /
          static_cast<double>(schedule.makespan);
    }
  }
  result.global.maximum_worker_peak_activation_bytes =
      UnitsToBytes(result.global.maximum_worker_peak_activation_units,
                   options.activation_bytes_per_unit);
  result.global.peak_simultaneous_activation_bytes_across_workers =
      UnitsToBytes(
          result.global.peak_simultaneous_activation_units_across_workers,
          options.activation_bytes_per_unit);

  ApplyUniformRatios(result, uniform_baseline);
  return result;
}

ActivationAnalysisResult MakeActivationAnalysisMetadata(
    const Instance &instance, const ActivationAnalysisOptions &options) {
  ActivationAnalysisResult result;
  PopulateConfigFields(result, options);
  try {
    ValidateActivationOptions(instance, options);
    result.activation_cap_units_per_worker =
        ResolveActivationCapUnits(instance, options);
    if (options.cap_mode == ActivationCapMode::kNone) {
      result.activation_cap_satisfied = std::nullopt;
    } else {
      result.activation_cap_satisfied = std::nullopt;
      result.activation_cap_source =
          options.cap_mode == ActivationCapMode::kExplicit ? "explicit"
                                                           : "uniform_baseline";
    }
    if (options.uniform_baseline) {
      result.activation_baseline_run_id =
          options.uniform_baseline->baseline_run_id;
      result.activation_cap_derivation_hash =
          options.uniform_baseline->cap_derivation_hash;
      result.activation_baseline_partition =
          options.uniform_baseline->partition;
      result.activation_baseline_method_contract_hash =
          options.uniform_baseline->method_contract_hash;
      result.activation_cap_derivation_runtime_seconds =
          options.uniform_baseline->derivation_runtime_seconds;
    }
  } catch (const Error &error) {
    AddError(result, error.what());
  }
  return result;
}

ActivationUniformBaseline DeriveUniformActivationBaseline(
    const Instance &instance, const ActivationAnalysisOptions &options) {
  const auto started = Clock::now();
  ActivationUniformBaseline baseline;
  try {
    ValidateActivationOptions(instance, options);
    baseline.partition = UniformSplit(instance);
    const MachineOrders orders = BreadthFirstOrders(instance);
    const EvaluationResult evaluated =
        EvaluateSchedule(instance, baseline.partition, orders);
    const ResultValidationResult validation =
        ValidateScheduleSolutionIndependent(instance, evaluated.schedule,
                                            "FEASIBLE",
                                            kUniformBreadthFirstMethod);
    if (!evaluated.schedule.ok()) {
      baseline.errors.push_back("uniform baseline schedule is invalid: " +
                                evaluated.schedule.validation_errors.front());
    }
    if (!validation.passed) {
      baseline.errors.push_back(
          "uniform baseline independent validation failed: " +
          validation.error_code);
    }
    ActivationAnalysisOptions baseline_options = options;
    baseline_options.cap_mode = ActivationCapMode::kNone;
    baseline_options.activation_cap_units.clear();
    baseline_options.enforce_activation_cap = false;
    baseline_options.emit_activation_trace = false;
    baseline_options.uniform_baseline = std::nullopt;
    const ActivationAnalysisResult analyzed = AnalyzeActivationMemory(
        instance, evaluated.schedule, baseline_options, std::nullopt, false);
    for (const ActivationWorkerMetrics &worker : analyzed.per_worker) {
      baseline.cap_units_per_worker.push_back(worker.peak_activation_units);
    }
    baseline.maximum_worker_peak_units =
        analyzed.global.maximum_worker_peak_activation_units;
    baseline.global_simultaneous_peak_units =
        analyzed.global.peak_simultaneous_activation_units_across_workers;
    for (const std::string &warning : analyzed.warnings)
      baseline.warnings.push_back(warning);
    for (const std::string &error : analyzed.errors)
      baseline.errors.push_back(error);
    baseline.method_contract_hash = StableHashHex(
        std::string("internal_activation_baseline.v1|") + kUniformPartitionRule +
        "|" + kFixedScheduleRuleBreadthFirst);
    const std::string seed = BaselineDerivationSeed(
        instance, options, baseline.partition, baseline.method_contract_hash);
    baseline.cap_derivation_hash = StableHashHex(seed);
    baseline.baseline_run_id =
        "activation-uniform-" + baseline.cap_derivation_hash;
  } catch (const std::exception &error) {
    baseline.errors.push_back(error.what());
  }
  baseline.derivation_runtime_seconds = Since(started);
  return baseline;
}

ActivationAnalysisResult AnalyzeActivationMemoryWithUniformBaseline(
    const Instance &instance, const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &options,
    bool activation_cap_enforced_in_solver) {
  const ActivationUniformBaseline baseline =
      options.uniform_baseline
          ? *options.uniform_baseline
          : DeriveUniformActivationBaseline(instance, options);
  return AnalyzeActivationMemory(instance, schedule, options, baseline,
                                 activation_cap_enforced_in_solver);
}

}  // namespace slackpipe
