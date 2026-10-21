#include "slackpipe/result_schema.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>

#include "slackpipe/operation.h"

#ifndef SLACKPIPE_BUILD_GIT_COMMIT
#define SLACKPIPE_BUILD_GIT_COMMIT ""
#endif

#ifndef SLACKPIPE_BUILD_GIT_DIRTY
#define SLACKPIPE_BUILD_GIT_DIRTY ""
#endif

#ifndef SLACKPIPE_HAVE_ORTOOLS
#define SLACKPIPE_HAVE_ORTOOLS 0
#endif

#ifndef SLACKPIPE_ORTOOLS_VERSION
#define SLACKPIPE_ORTOOLS_VERSION ""
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
std::optional<std::string> NonEmpty(std::string value) {
  if (value.empty()) return std::nullopt;
  return value;
}

std::optional<std::string> EnvString(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string(value);
}

std::optional<bool> ParseBoolText(const std::string &value) {
  if (value == "true" || value == "1" || value == "dirty") return true;
  if (value == "false" || value == "0" || value == "clean") return false;
  return std::nullopt;
}

std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '"':
      case '\\':
        out << '\\' << ch;
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void WriteJsonString(std::ostringstream &out, const std::string &value) {
  out << "\"" << JsonEscape(value) << "\"";
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

void WriteOptionalBool(std::ostringstream &out,
                       const std::optional<bool> &value) {
  if (value) {
    out << (*value ? "true" : "false");
  } else {
    out << "null";
  }
}

void WriteOptionalDouble(std::ostringstream &out,
                         const std::optional<double> &value) {
  if (value) {
    out << *value;
  } else {
    out << "null";
  }
}

void WriteStringArray(std::ostringstream &out,
                      const std::vector<std::string> &values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    WriteJsonString(out, values[i]);
  }
  out << "]";
}

void WriteActivationIdentity(std::ostringstream &out,
                             const ActivationIdentity &identity) {
  out << "{\"microbatch\": " << identity.microbatch
      << ", \"stage\": " << identity.stage
      << ", \"worker\": " << identity.worker
      << ", \"forward_operation_id\": " << identity.forward_operation_id.value
      << ", \"backward_operation_id\": " << identity.backward_operation_id.value
      << "}";
}

void WriteActivationLifetime(std::ostringstream &out,
                             const ActivationLifetime &lifetime,
                             const std::string &indent) {
  out << "{\n";
  out << indent << "  \"identity\": ";
  WriteActivationIdentity(out, lifetime.identity);
  out << ",\n";
  out << indent << "  \"start\": " << lifetime.start << ",\n";
  out << indent << "  \"end\": " << lifetime.end << ",\n";
  out << indent << "  \"demand_units\": " << lifetime.demand_units << ",\n";
  out << indent
      << "  \"zero_length\": " << (lifetime.zero_length ? "true" : "false")
      << "\n";
  out << indent << "}";
}

void WriteActivationMetrics(std::ostringstream &out,
                            const ActivationAnalysisResult &analysis,
                            const std::string &indent) {
  const std::string nested = indent + "  ";
  out << "{\n";
  out << nested << "\"total_activation_lifetimes\": "
      << analysis.total_activation_lifetimes << ",\n";
  out << nested << "\"zero_length_activation_lifetimes\": "
      << analysis.zero_length_activation_lifetimes << ",\n";
  out << nested << "\"per_worker\": [";
  for (std::size_t i = 0; i < analysis.per_worker.size(); ++i) {
    if (i != 0) out << ",";
    const ActivationWorkerMetrics &worker = analysis.per_worker[i];
    out << "\n" << nested << "  {\n";
    out << nested << "    \"worker\": " << worker.worker << ",\n";
    out << nested << "    \"peak_live_activation_count\": "
        << worker.peak_live_activation_count << ",\n";
    out << nested
        << "    \"peak_activation_units\": " << worker.peak_activation_units
        << ",\n";
    out << nested << "    \"peak_activation_bytes\": ";
    WriteOptionalDouble(out, worker.peak_activation_bytes);
    out << ",\n";
    out << nested << "    \"time_of_peak\": " << worker.time_of_peak << ",\n";
    out << nested << "    \"average_live_activation_count\": "
        << worker.average_live_activation_count << ",\n";
    out << nested << "    \"average_activation_units\": "
        << worker.average_activation_units << ",\n";
    out << nested << "    \"activation_count_time_area\": "
        << worker.activation_count_time_area << ",\n";
    out << nested << "    \"activation_unit_time_area\": "
        << worker.activation_unit_time_area << ",\n";
    out << nested << "    \"cap_units\": ";
    WriteOptionalNumber(out, worker.cap_units);
    out << ",\n";
    out << nested << "    \"cap_satisfied\": ";
    WriteOptionalBool(out, worker.cap_satisfied);
    out << "\n" << nested << "  }";
  }
  if (!analysis.per_worker.empty()) out << "\n" << nested;
  out << "],\n";
  out << nested << "\"global\": {\n";
  out << nested << "  \"maximum_worker_peak_activation_count\": "
      << analysis.global.maximum_worker_peak_activation_count << ",\n";
  out << nested << "  \"maximum_worker_peak_activation_units\": "
      << analysis.global.maximum_worker_peak_activation_units << ",\n";
  out << nested << "  \"maximum_worker_peak_activation_bytes\": ";
  WriteOptionalDouble(out,
                      analysis.global.maximum_worker_peak_activation_bytes);
  out << ",\n";
  out << nested << "  \"peak_simultaneous_activation_count_across_workers\": "
      << analysis.global.peak_simultaneous_activation_count_across_workers
      << ",\n";
  out << nested << "  \"peak_simultaneous_activation_units_across_workers\": "
      << analysis.global.peak_simultaneous_activation_units_across_workers
      << ",\n";
  out << nested << "  \"peak_simultaneous_activation_bytes_across_workers\": ";
  WriteOptionalDouble(
      out, analysis.global.peak_simultaneous_activation_bytes_across_workers);
  out << ",\n";
  out << nested
      << "  \"time_of_global_peak\": " << analysis.global.time_of_global_peak
      << ",\n";
  out << nested << "  \"total_activation_unit_time_area\": "
      << analysis.global.total_activation_unit_time_area << "\n";
  out << nested << "},\n";
  out << nested << "\"activation_lifetimes\": [";
  for (std::size_t i = 0; i < analysis.lifetimes.size(); ++i) {
    if (i != 0) out << ",";
    out << "\n" << nested << "  ";
    WriteActivationLifetime(out, analysis.lifetimes[i], nested + "  ");
  }
  if (!analysis.lifetimes.empty()) out << "\n" << nested;
  out << "],\n";
  out << nested << "\"event_trace\": [";
  for (std::size_t i = 0; i < analysis.event_trace.size(); ++i) {
    if (i != 0) out << ",";
    const ActivationEventTraceEntry &entry = analysis.event_trace[i];
    out << "\n" << nested << "  {\n";
    out << nested << "    \"timestamp\": " << entry.timestamp << ",\n";
    out << nested << "    \"event_type\": ";
    WriteJsonString(out, entry.event_type);
    out << ",\n";
    out << nested << "    \"activation_id\": ";
    WriteActivationIdentity(out, entry.identity);
    out << ",\n";
    out << nested << "    \"demand_units\": " << entry.demand_units << ",\n";
    out << nested << "    \"live_activation_count_after_event\": "
        << entry.live_activation_count_after_event << ",\n";
    out << nested << "    \"live_activation_units_after_event\": "
        << entry.live_activation_units_after_event << "\n";
    out << nested << "  }";
  }
  if (!analysis.event_trace.empty()) out << "\n" << nested;
  out << "],\n";
  out << nested << "\"warnings\": ";
  WriteStringArray(out, analysis.warnings);
  out << ",\n";
  out << nested << "\"errors\": ";
  WriteStringArray(out, analysis.errors);
  out << "\n" << indent << "}";
}

std::optional<std::string> GitCommitValue() {
  if (const std::optional<std::string> env =
          EnvString("SLACKPIPE_GIT_COMMIT")) {
    return env;
  }
  return NonEmpty(SLACKPIPE_BUILD_GIT_COMMIT);
}

std::optional<bool> GitDirtyValue(std::optional<std::string> *scope) {
  if (const std::optional<std::string> env = EnvString("SLACKPIPE_GIT_DIRTY")) {
    if (const std::optional<bool> parsed = ParseBoolText(*env)) {
      if (scope != nullptr) *scope = std::string("runtime_environment");
      return parsed;
    }
  }
  const std::optional<std::string> built = NonEmpty(SLACKPIPE_BUILD_GIT_DIRTY);
  if (!built) return std::nullopt;
  if (const std::optional<bool> parsed = ParseBoolText(*built)) {
    if (scope != nullptr) *scope = std::string("build_time");
    return parsed;
  }
  return std::nullopt;
}

std::optional<std::string> BuildTypeValue() {
  if (const std::optional<std::string> env =
          EnvString("SLACKPIPE_BUILD_TYPE")) {
    return env;
  }
#ifdef NDEBUG
  return std::string("Release");
#else
  return std::string("Debug");
#endif
}
}
std::string CurrentTimestampUtc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::optional<std::string> CurrentGitCommit() { return GitCommitValue(); }

std::optional<bool> CurrentGitDirty(std::optional<std::string> *scope) {
  return GitDirtyValue(scope);
}

std::optional<std::string> CurrentBuildType() { return BuildTypeValue(); }

bool OrToolsCompiledIn() { return SLACKPIPE_HAVE_ORTOOLS != 0; }

std::optional<std::string> OrToolsVersion() {
  return NonEmpty(SLACKPIPE_ORTOOLS_VERSION);
}

bool CumulativeConstraintCompiledIn() {
  return OrToolsCompiledIn() && SLACKPIPE_HAVE_CUMULATIVE != 0;
}

bool VariableCumulativeDemandCompiledIn() {
  return CumulativeConstraintCompiledIn() &&
         SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND != 0;
}

std::string ActivationCapSolverSupportLevelForBuild() {
  if (!CumulativeConstraintCompiledIn()) {
    return "none";
  }
  if (VariableCumulativeDemandCompiledIn()) {
    return "variable_demands";
  }
  return "fixed_demands_only";
}

std::string CommandLineFromArgv(int argc, char **argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) out << ' ';
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    const bool needs_quotes =
        arg.find_first_of(" \t\n\"'\\") != std::string::npos;
    if (!needs_quotes) {
      out << arg;
      continue;
    }
    out << '\'';
    for (char ch : arg) {
      if (ch == '\'') {
        out << "'\\''";
      } else {
        out << ch;
      }
    }
    out << '\'';
  }
  return out.str();
}

std::string ExecutableNameFromArgv0(const std::string &argv0) {
  const std::size_t slash = argv0.find_last_of("/\\");
  return slash == std::string::npos ? argv0 : argv0.substr(slash + 1);
}

std::string MakeRunId(const std::string &seed) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : seed) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << "run-" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

bool IsTerminalFeasibleStatus(const std::string &status) {
  return status == "OPTIMAL" || status == "FEASIBLE";
}

std::optional<double> RelativeGap(Tick makespan, double best_bound) {
  if (makespan <= 0 || best_bound <= 0.0) return std::nullopt;
  return std::max(0.0, (static_cast<double>(makespan) - best_bound) /
                           static_cast<double>(makespan));
}
std::string ActivationAnalysisJson(const ActivationAnalysisResult& analysis) {
  std::ostringstream out;
  out << "{\"model\":\"" << ToString(analysis.model) << "\","
      << "\"cap_mode\":\"" << ToString(analysis.cap_mode) << "\","
      << "\"enforcement_requested\":" << (analysis.activation_cap_enforcement_requested ? "true" : "false") << ","
      << "\"enforced_in_solver\":" << (analysis.activation_cap_enforced_in_solver ? "true" : "false") << ","
      << "\"enforcement_mode\":\"" << analysis.activation_cap_enforcement_mode << "\","
      << "\"cap_satisfied\":";
  WriteOptionalBool(out, analysis.activation_cap_satisfied);
  out << ",\"metrics\":";
  WriteActivationMetrics(out, analysis, "  ");
  out << "}";
  return out.str();
}
}  // namespace slackpipe
