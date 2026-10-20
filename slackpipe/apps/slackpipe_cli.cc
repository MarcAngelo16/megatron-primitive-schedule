#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/alternating_solver.h"
#include "slackpipe/bfs_solver.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/cost_profile.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/fixed_order_partition_solver.h"
#include "slackpipe/interleaving_stats.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/one_f_one_b.h"
#include "slackpipe/operation.h"
#include "slackpipe/octopipe_tuner.h"
#include "slackpipe/plan_export.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/slackpipe_solver.h"

namespace {

using Clock = std::chrono::steady_clock;

void SignalHandler(int signal_number) {
  const char prefix[] = "SLACKPIPE_LIFECYCLE phase=CLI_EXIT detail=signal\n";
  (void)signal_number;
  const ssize_t written = write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
  (void)written;
}

std::string UtcTimestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

long PeakRssKb() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  return usage.ru_maxrss;
}

std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') out << '\\';
    if (ch == '\n') {
      out << "\\n";
    } else {
      out << ch;
    }
  }
  return out.str();
}

void LogLifecycle(Clock::time_point started,
                  const slackpipe::Instance &instance, const std::string &phase,
                  const std::string &algorithm,
                  double configured_solver_limit_seconds, int solver_threads,
                  const std::string &solver_status = "",
                  double effective_solver_limit_seconds = 0.0,
                  const std::string &detail = "",
                  double requested_solver_limit_seconds = 0.0,
                  double remaining_global_time_seconds = 0.0,
                  double phase_specific_cap_seconds = 0.0) {
  std::cerr
      << "SLACKPIPE_LIFECYCLE" << " ts=" << UtcTimestamp() << " phase=" << phase
      << " elapsed_seconds=" << Since(started) << " B=" << instance.microbatches
      << " N=" << instance.stages << " W=" << instance.workers
      << " L=" << instance.total_layers << " algorithm=" << algorithm
      << " configured_solver_limit_seconds=" << configured_solver_limit_seconds
      << " requested_solver_limit_seconds=" << requested_solver_limit_seconds
      << " effective_solver_limit_seconds=" << effective_solver_limit_seconds
      << " remaining_global_time_seconds=" << remaining_global_time_seconds
      << " phase_specific_cap_seconds=" << phase_specific_cap_seconds
      << " solver_threads=" << solver_threads
      << " solver_status=" << solver_status << " peak_rss_kb=" << PeakRssKb();
  if (!detail.empty()) std::cerr << " detail=" << detail;
  std::cerr << "\n" << std::flush;
}

std::string ValueAfter(int &i, int argc, char **argv) {
  if (i + 1 >= argc) throw slackpipe::Error("missing value for flag");
  ++i;
  return argv[i];
}

std::optional<std::string> InlineValue(const std::string &flag,
                                       const std::string &name) {
  const std::string prefix = name + "=";
  if (flag.rfind(prefix, 0) != 0) return std::nullopt;
  return flag.substr(prefix.size());
}

std::vector<slackpipe::Tick> ParseSplit(const std::string &text) {
  std::vector<slackpipe::Tick> split;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    split.push_back(std::stoll(token));
  }
  return split;
}

struct FixedPartitionResolution {
  std::vector<slackpipe::Tick> split;
  std::string reference_source;
  double phase_limit_seconds = 0.0;
  double runtime_seconds = 0.0;
  slackpipe::Index cp_sat_models_solved = 0;
  std::string status;
};

bool ParseBool(const std::string &text) {
  if (text == "true" || text == "1") return true;
  if (text == "false" || text == "0") return false;
  throw slackpipe::Error("expected boolean value true or false");
}

bool ParseOnOff(const std::string &text) {
  if (text == "on") return true;
  if (text == "off") return false;
  throw slackpipe::Error("expected value on or off");
}

bool ParseOptionalBoolFlag(int &i, int argc, char **argv) {
  if (i + 1 >= argc) return true;
  const std::string next = argv[i + 1] == nullptr ? "" : argv[i + 1];
  if (next.rfind("--", 0) == 0) return true;
  ++i;
  return ParseBool(next);
}

bool IsDeterministicUniformFixedOrderMethod(const std::string &algorithm) {
  return algorithm == slackpipe::kUniformBreadthFirstMethod ||
         algorithm == slackpipe::kUniformInterleavedOneFOneBMethod;
}

void PrintHelp() {
  std::cout
      << "Usage: slackpipe_cli [options]\n\n"
      << "Problem size:\n"
      << "  --B INT                         microbatch count\n"
      << "  --N INT                         stage count\n"
      << "  --J INT                         worker/GPU count\n"
      << "  --L INT                         total layer count\n"
      << "  --min-layers INT                minimum layers per stage\n"
      << "  --ratio-num INT                 backward duration numerator\n"
      << "  --ratio-den INT                 forward duration denominator\n"
      << "  --cost-profile PATH             slackpipe.cost_profile.v1/v2 "
         "measured "
         "cost model\n"
      << "  --communication INT             inter-worker edge delay\n\n"
      << "Split and algorithm:\n"
      << "  --split CSV                     comma-separated fixed stage split\n"
      << "  --fixed-partition-source NAME   uniform, load-balanced, "
         "partition-only\n"
      << "  --split-mode MODE               fixed, local, global, "
         "worker-fixed, worker-local\n"
      << "  --algorithm NAME                eval-bfs, optimize-bfs, "
         "partition-only,\n"
      << "                                  uniform-interleaved-1f1b, "
         "eval-1f1b,\n"
      << "                                  optimize-joint, joint, "
         "schedule-only-uniform,\n"
      << "                                  "
         "sequential-partition-then-schedule,\n"
      << "                                  alternating-partition-schedule, "
         "slackpipe, heuristic, octopipe-algorithm1-fixed-stage\n"
      << "  --octopipe-max-iterations INT    deterministic tuning limit (0: unlimited)\n"
      << "  --octopipe-candidates-per-iteration INT  F/B edit limit (default 64)\n"
      << "  --octopipe-fixed-placement BOOL disable whole-stage swaps (default false)\n"
      << "  --stage-to-worker CSV           OctoPipe initial placement (default cyclic)\n"
      << "  --method NAME                   optimize-bfs method: auto, "
         "enumerate, cpsat\n"
      << "  --fixed-order-partition-backend NAME\n"
      << "                                  fixed-order partition backend: "
         "auto, enumerate, cpsat\n"
      << "  --bfs-method NAME               incumbent method: auto, "
         "heuristic, hybrid-slack,\n"
      << "                                  uniform, enumerate, cpsat\n"
      << "  --incumbent-method NAME         external incumbent: heuristic, slack, "
         "canonical, none\n"
      << "  --incumbent-bound on|off        use incumbent as CP-SAT horizon "
         "bound\n"
      << "  --incumbent-hints on|off        add incumbent solution hints to "
         "CP-SAT\n"
      << "  --use-bfs-hints BOOL            use BFS hints in joint solver\n\n"
      << "  --alternating-max-rounds INT    max alternating rounds, default "
         "4\n\n"
      << "Solver controls:\n"
      << "  --time-limit-seconds FLOAT      CP-SAT time limit per solve\n"
      << "  --num-workers INT               CP-SAT parallel search workers\n"
      << "  --random-seed INT               CP-SAT random seed\n"
      << "  --heuristic-time-limit-seconds FLOAT  default 60; shared with global budget\n"
      << "  --heuristic-evaluation-limit INT  full search replay limit, default 100000000\n"
      << "  --heuristic-refinement NAME    coupled (only supported mode)\n"
      << "  --heuristic-fixed-evaluations BOOL  deterministic standalone test mode, ignores wall deadlines\n"
      << "  --heuristic-verify-replay BOOL  debug cross-check against general evaluator\n"
      << "  --heuristic-trace PATH         standalone diagnostics and acceptance trace JSON\n"
      << "  Early hints: --incumbent-method heuristic (joint) / --bfs-method heuristic (SlackPipe)\n"
      << "  --require-optimal BOOL          reject non-optimal solver result\n"
      << "  --log-search-progress BOOL      enable CP-SAT search logging\n"
      << "  --enumeration-threshold INT     split enumeration threshold\n"
      << "  --fifo-ordering on|off          enforce FIFO micro-batch order at "
         "each operation position\n"
      << "  --symmetry-break-f0-fifo BOOL   enable F0 FIFO symmetry break\n"
      << "  --worker-balance-tolerance-percent FLOAT\n"
      << "                                  hard worker aggregate layer "
         "balance tolerance percent\n"
      << "  --worker-balance-tolerance-layers INT\n"
      << "                                  hard worker aggregate layer "
         "balance tolerance in layers\n"
      << "  --worker-balance-pruning on|off require/apply worker-balance "
         "bounds when on\n"
      << "  --enable-pressure-pruning       enable pressure-guided partition "
         "pruning\n"
      << "  --pressure-partition-top-k INT  keep lowest-pressure top-K "
         "partitions when enabled\n"
      << "  --pressure-partition-epsilon FLOAT\n"
      << "                                  also keep partitions within (1 + "
         "epsilon) of best pressure\n"
      << "  --pressure-lambda FLOAT         pressure co-location term weight, "
         "default 1.0\n"
      << "  --pressure-gamma FLOAT          pressure distance exponent, "
         "default 2.0\n"
      << "  --pressure-alpha FLOAT          max-vs-average pressure blend, "
         "default 0.7\n"
      << "  --pressure-beam-width INT       pressure split beam width, default "
         "512\n"
      << "  --pressure-branch-width INT     pressure split branch width, "
         "default 16\n"
      << "  --pressure-generated-partitions INT\n"
      << "                                  max generated pressure split "
         "candidates, default max(512, 8 * top-k)\n"
      << "  --pressure-pred-top-k INT       experimental predecessor candidate "
         "cap; not applied to joint CP-SAT NoOverlap\n\n"
      << "Activation memory:\n"
      << "  --activation-model NAME         count, linear-in-stage-layers, "
         "explicit-stage-units\n"
      << "  --activation-units-per-layer INT\n"
      << "                                  unit scale for "
         "linear-in-stage-layers, default 1\n"
      << "  --activation-stage-units CSV    N explicit stage activation units\n"
      << "  --activation-bytes-per-unit FLOAT\n"
      << "                                  optional byte conversion per "
         "controlled unit\n"
      << "  --activation-cap-mode NAME      none, explicit, uniform-baseline\n"
      << "  --activation-cap-units CSV      scalar or W-length cap vector for "
         "explicit mode, or a pre-materialized uniform-baseline vector\n"
      << "  --activation-cap-derivation-hash HEX\n"
      << "                                  provenance hash for a materialized "
         "uniform-baseline cap\n"
      << "  --enforce-activation-cap [BOOL] request activation-cap "
         "enforcement\n"
      << "  --activation-cap-enforcement NAME\n"
      << "                                  posthoc-only, solver, or "
         "exact-enumeration; enforced modes also enable "
         "--enforce-activation-cap\n"
      << "  --emit-activation-trace [BOOL] serialize activation event trace\n\n"
      << "SlackPipe split controls:\n"
      << "  Reference split is optimized BFS/hybrid incumbent; fixed mode "
         "fixes only partition, not order\n"
      << "  worker-fixed fixes worker aggregate layer totals, not the complete "
         "stage split\n"
      << "  --move-budget INT                local mode: max moved layers, "
         "defined as half stage L1\n"
      << "  --per-stage-delta INT            local mode: max per-stage "
         "deviation from reference\n"
      << "  --worker-move-budget INT         worker-local mode: max moved "
         "worker layers, defined as half worker-load L1\n"
      << "  --per-worker-delta INT           worker-local mode: max per-worker "
         "aggregate deviation\n\n"
      << "Output:\n"
      << "  validate-result --input PATH     validate one SlackPipe result "
         "JSON\n"
      << "  build-info                       print build/provenance JSON\n"
      << "  describe-method NAME            print canonical method contract "
         "JSON\n"
      << "  --output-prefix PATH            output prefix for "
         "JSON/CSV/orders/SVG\n"
      << "  --emit-plan PATH                write Megatron slackpipe.plan.v1 "
         "JSON\n"
      << "  --emit-search-stats BOOL        print concise search statistics\n"
      << "  --search-stats-json PATH        write full search statistics JSON\n"
      << "  --dump-interleaving-csv PATH    write raw F/B adjacency "
         "interleaving events\n"
      << "  --dump-interleaving-summary-csv PATH\n"
      << "                                  write aggregated F/B interleaving "
         "counts\n"
      << "  -h, --help                      print this help and exit\n";
}

std::string ReadTextFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw slackpipe::Error("failed to open input file: " + path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

void WriteOptionalStringJson(std::ostringstream &out,
                             const std::optional<std::string> &value) {
  if (value) {
    out << "\"" << JsonEscape(*value) << "\"";
  } else {
    out << "null";
  }
}

void WriteOptionalBoolJson(std::ostringstream &out,
                           const std::optional<bool> &value) {
  if (value) {
    out << (*value ? "true" : "false");
  } else {
    out << "null";
  }
}

std::string BuildInfoJson() {
  std::optional<std::string> dirty_scope;
  std::ostringstream out;
  out << "{\n";
  out << "  \"schema_version\": " << slackpipe::kBuildInfoSchemaVersion
      << ",\n";
  out << "  \"evaluation_result_schema_version\": "
      << slackpipe::kEvaluationResultSchemaVersion << ",\n";
  out << "  \"evaluation_method_version\": "
      << slackpipe::kEvaluationMethodVersion << ",\n";
  out << "  \"budget_policy_version\": "
      << slackpipe::kEvaluationBudgetPolicyVersion << ",\n";
  out << "  \"validation_version\": " << slackpipe::kResultValidationVersion
      << ",\n";
  out << "  \"activation_analysis_version\": "
      << slackpipe::kActivationAnalysisVersion << ",\n";
  out << "  \"git_commit\": ";
  WriteOptionalStringJson(out, slackpipe::CurrentGitCommit());
  out << ",\n";
  out << "  \"git_dirty\": ";
  WriteOptionalBoolJson(out, slackpipe::CurrentGitDirty(&dirty_scope));
  out << ",\n";
  out << "  \"git_dirty_scope\": ";
  WriteOptionalStringJson(out, dirty_scope);
  out << ",\n";
  out << "  \"build_type\": ";
  WriteOptionalStringJson(out, slackpipe::CurrentBuildType());
  out << ",\n";
  out << "  \"ortools_compiled\": "
      << (slackpipe::OrToolsCompiledIn() ? "true" : "false") << ",\n";
  out << "  \"ortools_enabled\": "
      << (slackpipe::OrToolsCompiledIn() ? "true" : "false") << ",\n";
  out << "  \"ortools_version\": ";
  WriteOptionalStringJson(out, slackpipe::OrToolsVersion());
  out << ",\n";
  out << "  \"cumulative_constraint_supported\": "
      << (slackpipe::CumulativeConstraintCompiledIn() ? "true" : "false")
      << ",\n";
  out << "  \"variable_cumulative_demand_supported\": "
      << (slackpipe::VariableCumulativeDemandCompiledIn() ? "true" : "false")
      << ",\n";
  out << "  \"activation_cap_solver_support\": \""
      << JsonEscape(slackpipe::ActivationCapSolverSupportLevelForBuild())
      << "\"\n";
  out << "}\n";
  return out.str();
}

}  // namespace

int main(int argc, char **argv) {
  const auto cli_started = Clock::now();
  const std::string requested_command =
      slackpipe::CommandLineFromArgv(argc, argv);
  const std::string invocation_timestamp = slackpipe::CurrentTimestampUtc();
  const std::string executable_name =
      argc > 0 && argv[0] != nullptr
          ? slackpipe::ExecutableNameFromArgv0(argv[0])
          : std::string("slackpipe_cli");
  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGINT, SignalHandler);
  if (argc >= 2 && std::string(argv[1]) == "build-info") {
    std::cout << BuildInfoJson();
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "validate-result") {
    try {
      std::string input_path;
      bool activation_summary = false;
      slackpipe::ActivationAnalysisOptions offline_activation_options;
      for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--input") {
          input_path = ValueAfter(i, argc, argv);
        } else if (flag == "--activation-summary") {
          activation_summary = ParseOptionalBoolFlag(i, argc, argv);
        } else if (flag == "--activation-model") {
          offline_activation_options.model =
              slackpipe::ParseActivationModel(ValueAfter(i, argc, argv));
        } else if (flag == "--activation-units-per-layer") {
          offline_activation_options.activation_units_per_layer =
              std::stoll(ValueAfter(i, argc, argv));
        } else if (flag == "--activation-stage-units") {
          offline_activation_options.explicit_stage_activation_units =
              ParseSplit(ValueAfter(i, argc, argv));
        } else if (flag == "--activation-bytes-per-unit") {
          offline_activation_options.activation_bytes_per_unit =
              std::stod(ValueAfter(i, argc, argv));
        } else if (flag == "--activation-cap-mode") {
          offline_activation_options.cap_mode =
              slackpipe::ParseActivationCapMode(ValueAfter(i, argc, argv));
        } else if (flag == "--activation-cap-units") {
          offline_activation_options.activation_cap_units =
              ParseSplit(ValueAfter(i, argc, argv));
        } else if (flag == "--enforce-activation-cap") {
          offline_activation_options.enforce_activation_cap =
              ParseOptionalBoolFlag(i, argc, argv);
        } else if (flag == "--activation-cap-enforcement") {
          const std::string enforcement = ValueAfter(i, argc, argv);
          if (enforcement == "solver") {
            offline_activation_options.enforce_activation_cap = true;
          } else if (enforcement == "posthoc-only" ||
                     enforcement == "posthoc_only" ||
                     enforcement == "posthoc") {
            offline_activation_options.enforce_activation_cap = false;
          } else {
            throw slackpipe::Error("unknown activation cap enforcement mode: " +
                                   enforcement);
          }
        } else if (flag == "--emit-activation-trace") {
          offline_activation_options.emit_activation_trace =
              ParseOptionalBoolFlag(i, argc, argv);
        } else if (flag == "--help" || flag == "-h") {
          std::cout << "Usage: slackpipe_cli validate-result --input PATH "
                       "[--activation-summary]\n";
          return 0;
        } else {
          throw slackpipe::Error("unknown validate-result flag: " + flag);
        }
      }
      if (input_path.empty()) {
        throw slackpipe::Error("validate-result requires --input PATH");
      }
      const std::string json_text = ReadTextFile(input_path);
      const slackpipe::ResultValidationResult validation =
          slackpipe::ValidateResultJsonText(json_text);
      if (activation_summary) {
        std::ostringstream out;
        out << "{\n";
        out << "  \"validation\": "
            << slackpipe::ResultValidationToJson(validation, "  ") << ",\n";
        if (!validation.passed) {
          out << "  \"canonical_result\": null,\n";
          out << "  \"activation_reanalysis_error\": \"result validation "
                 "failed before activation analysis\"\n";
          out << "}\n";
          std::cout << out.str();
          return 2;
        }
        slackpipe::ResultValidationInput input =
            slackpipe::ValidationInputFromResultJsonText(json_text);
        slackpipe::ValidateActivationOptions(input.instance,
                                             offline_activation_options);
        if (offline_activation_options.cap_mode ==
                slackpipe::ActivationCapMode::kUniformBaseline &&
            !offline_activation_options.uniform_baseline) {
          offline_activation_options.uniform_baseline =
              slackpipe::DeriveUniformActivationBaseline(
                  input.instance, offline_activation_options);
        }
        slackpipe::ScheduleSolution schedule =
            slackpipe::ScheduleSolutionFromValidationInput(input);
        slackpipe::CanonicalSemantics semantics;
        semantics.canonical_method =
            input.method_name.value_or("offline-result-reanalysis");
        semantics.actual_solver_path = "offline-result-reanalysis";
        slackpipe::CanonicalRequestContext context;
        context.timestamp_utc = invocation_timestamp;
        context.requested_command = requested_command;
        context.requested_method = semantics.canonical_method;
        context.executable_name = executable_name;
        slackpipe::CanonicalOutcome outcome = slackpipe::OutcomeFromSchedule(
            schedule, input.reported_status.value_or("FEASIBLE"));
        slackpipe::CanonicalResultMetadata canonical =
            slackpipe::BuildCanonicalResultMetadata(
                input.instance, context, semantics, outcome,
                input.selected_partition, std::nullopt);
        slackpipe::ApplyResultValidation(canonical.outcome, validation);
        const slackpipe::ActivationAnalysisResult activation =
            slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
                input.instance, schedule, offline_activation_options, false);
        slackpipe::ApplyActivationAnalysis(canonical, activation);
        out << "  \"canonical_result\": "
            << slackpipe::CanonicalResultToJson(canonical, "  ") << "\n";
        out << "}\n";
        std::cout << out.str();
        return canonical.outcome.reported_status == "INVALID_RESULT" ? 2 : 0;
      }
      std::cout << slackpipe::ResultValidationToJson(validation, "") << "\n";
      return validation.passed ? 0 : 2;
    } catch (const std::exception &error) {
      slackpipe::ResultValidationResult validation;
      validation.passed = false;
      validation.error_code = "result_json_unreadable";
      validation.error_category = "serialization";
      validation.message = error.what();
      std::cout << slackpipe::ResultValidationToJson(validation, "") << "\n";
      return 2;
    }
  }
  if (argc >= 2 && std::string(argv[1]) == "describe-method") {
    try {
      if (argc == 3) {
        std::cout << slackpipe::DescribeEvaluationMethodJson(argv[2]);
        return 0;
      }
      if (argc == 4 && std::string(argv[2]) == "--method") {
        std::cout << slackpipe::DescribeEvaluationMethodJson(argv[3]);
        return 0;
      }
      std::cout << "Usage: slackpipe_cli describe-method METHOD\n";
      return 2;
    } catch (const std::exception &error) {
      std::cerr << error.what() << "\n";
      return 2;
    }
  }
  slackpipe::Instance instance;
  std::vector<slackpipe::Tick> split;
  std::string fixed_partition_source;
  std::string output_prefix;
  std::string emit_plan_path;
  std::string heuristic_trace_path;
  std::string cost_profile_path;
  std::string algorithm = "eval-bfs";
  std::string method = "auto";
  std::string fixed_order_partition_backend = "auto";
  bool fixed_order_partition_backend_provided = false;
  bool emit_search_stats = false;
  bool use_bfs_hints_provided = false;
  bool bfs_method_provided = false;
  bool incumbent_method_provided = false;
  bool incumbent_bound_provided = false;
  bool incumbent_hints_provided = false;
  std::string search_stats_json_path;
  std::string interleaving_csv_path;
  std::string interleaving_summary_csv_path;
  slackpipe::SearchStats search_stats;
  slackpipe::BfsSplitOptimizerOptions options;
  slackpipe::JointOptimizerOptions joint_options;
  slackpipe::SlackPipeOptions slackpipe_options;
  slackpipe::OctoPipeOptions octopipe_options;
  bool octopipe_flags_provided = false;
  slackpipe::ActivationAnalysisOptions activation_options;
  std::string activation_cap_derivation_hash;
  std::string activation_cap_enforcement_request;
  int alternating_max_rounds = slackpipe::kDefaultAlternatingMaxRounds;
  try {
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      PrintHelp();
      return 0;
    }

    LogLifecycle(cli_started, instance, "CLI_START", algorithm,
                 joint_options.time_limit_seconds, joint_options.num_workers);

    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--B") {
        instance.microbatches = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--N") {
        instance.stages = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--J") {
        instance.workers = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--L") {
        instance.total_layers = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--min-layers") {
        instance.min_layers = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--ratio-num") {
        instance.backward_ratio_num = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--ratio-den") {
        instance.backward_ratio_den = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--cost-profile") {
        cost_profile_path = ValueAfter(i, argc, argv);
      } else if (flag == "--communication") {
        instance.communication_ticks = std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--split") {
        split = ParseSplit(ValueAfter(i, argc, argv));
      } else if (flag == "--stage-to-worker") {
        instance.stage_to_worker = ParseSplit(ValueAfter(i, argc, argv));
        octopipe_flags_provided = true;
      } else if (flag == "--octopipe-max-iterations") {
        octopipe_options.max_iterations = std::stoll(ValueAfter(i, argc, argv));
        octopipe_flags_provided = true;
      } else if (flag == "--octopipe-candidates-per-iteration") {
        octopipe_options.candidates_per_iteration = std::stoll(ValueAfter(i, argc, argv));
        octopipe_flags_provided = true;
      } else if (flag == "--octopipe-fixed-placement") {
        octopipe_options.tune_placement = !ParseBool(ValueAfter(i, argc, argv));
        octopipe_flags_provided = true;
      } else if (flag == "--fixed-partition-source") {
        fixed_partition_source = ValueAfter(i, argc, argv);
      } else if (flag == "--output-prefix") {
        output_prefix = ValueAfter(i, argc, argv);
      } else if (flag == "--emit-plan") {
        emit_plan_path = ValueAfter(i, argc, argv);
      } else if (flag == "--emit-search-stats") {
        emit_search_stats = ParseBool(ValueAfter(i, argc, argv));
      } else if (flag == "--search-stats-json") {
        search_stats_json_path = ValueAfter(i, argc, argv);
      } else if (flag == "--dump-interleaving-csv") {
        interleaving_csv_path = ValueAfter(i, argc, argv);
      } else if (flag == "--dump-interleaving-summary-csv") {
        interleaving_summary_csv_path = ValueAfter(i, argc, argv);
      } else if (flag == "--algorithm") {
        algorithm = ValueAfter(i, argc, argv);
      } else if (flag == "--method") {
        method = ValueAfter(i, argc, argv);
      } else if (flag == "--fixed-order-partition-backend") {
        fixed_order_partition_backend = ValueAfter(i, argc, argv);
        fixed_order_partition_backend_provided = true;
      } else if (flag == "--split-mode") {
        slackpipe_options.split_mode =
            slackpipe::ParseSlackPipeSplitMode(ValueAfter(i, argc, argv));
      } else if (flag == "--move-budget") {
        slackpipe_options.move_budget = std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.move_budget_provided = true;
      } else if (flag == "--per-stage-delta") {
        slackpipe_options.per_stage_delta =
            std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--worker-move-budget") {
        slackpipe_options.worker_move_budget =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.worker_move_budget_provided = true;
      } else if (flag == "--per-worker-delta") {
        slackpipe_options.per_worker_delta =
            std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--enumeration-threshold") {
        options.enumeration_threshold =
            static_cast<std::uint64_t>(std::stoull(ValueAfter(i, argc, argv)));
        joint_options.enumeration_threshold = options.enumeration_threshold;
        slackpipe_options.enumeration_threshold = options.enumeration_threshold;
      } else if (flag == "--time-limit-seconds") {
        options.time_limit_seconds = std::stod(ValueAfter(i, argc, argv));
        joint_options.time_limit_seconds = options.time_limit_seconds;
        slackpipe_options.time_limit_seconds = options.time_limit_seconds;
      } else if (flag == "--num-workers") {
        options.num_workers = std::stoi(ValueAfter(i, argc, argv));
        joint_options.num_workers = options.num_workers;
        slackpipe_options.num_workers = options.num_workers;
      } else if (flag == "--heuristic-time-limit-seconds") {
        joint_options.heuristic_options.time_limit_seconds = std::stod(ValueAfter(i, argc, argv));
        slackpipe_options.heuristic_options = joint_options.heuristic_options;
      } else if (flag == "--heuristic-evaluation-limit") {
        joint_options.heuristic_options.evaluation_limit = std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.heuristic_options = joint_options.heuristic_options;
      } else if (flag == "--heuristic-refinement") {
        joint_options.heuristic_options.refinement_mode = ValueAfter(i, argc, argv);
        slackpipe_options.heuristic_options = joint_options.heuristic_options;
      } else if (flag == "--heuristic-fixed-evaluations") {
        joint_options.heuristic_options.fixed_evaluation_test = ParseBool(ValueAfter(i, argc, argv));
        slackpipe_options.heuristic_options = joint_options.heuristic_options;
      } else if (flag == "--heuristic-verify-replay") {
        joint_options.heuristic_options.verify_cached_replay = ParseBool(ValueAfter(i, argc, argv));
        slackpipe_options.heuristic_options = joint_options.heuristic_options;
      } else if (flag == "--heuristic-trace") {
        heuristic_trace_path = ValueAfter(i, argc, argv);
        joint_options.heuristic_options.record_trace = true;
        slackpipe_options.heuristic_options = joint_options.heuristic_options;
      } else if (flag == "--random-seed") {
        options.random_seed = std::stoi(ValueAfter(i, argc, argv));
        joint_options.random_seed = options.random_seed;
        slackpipe_options.random_seed = options.random_seed;
      } else if (flag == "--require-optimal") {
        options.require_optimal = ParseBool(ValueAfter(i, argc, argv));
        joint_options.require_optimal = options.require_optimal;
        slackpipe_options.require_optimal = options.require_optimal;
      } else if (flag == "--log-search-progress") {
        options.log_search_progress = ParseBool(ValueAfter(i, argc, argv));
        joint_options.log_search_progress = options.log_search_progress;
        slackpipe_options.log_search_progress = options.log_search_progress;
      } else if (flag == "--symmetry-break-f0-fifo") {
        joint_options.symmetry_break_f0_fifo =
            ParseBool(ValueAfter(i, argc, argv));
        slackpipe_options.symmetry_break_f0_fifo =
            joint_options.symmetry_break_f0_fifo;
      } else if (flag == "--fifo-ordering" ||
                 InlineValue(flag, "--fifo-ordering")) {
        const std::optional<std::string> inline_value =
            InlineValue(flag, "--fifo-ordering");
        const std::string value =
            inline_value ? *inline_value : ValueAfter(i, argc, argv);
        joint_options.fifo_ordering = ParseOnOff(value);
        slackpipe_options.fifo_ordering = joint_options.fifo_ordering;
      } else if (flag == "--worker-balance-tolerance-percent") {
        joint_options.worker_balance_tolerance_percent =
            std::stod(ValueAfter(i, argc, argv));
        slackpipe_options.worker_balance_tolerance_percent =
            joint_options.worker_balance_tolerance_percent;
      } else if (flag == "--worker-balance-tolerance-layers") {
        joint_options.worker_balance_tolerance_layers =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.worker_balance_tolerance_layers =
            *joint_options.worker_balance_tolerance_layers;
      } else if (flag == "--worker-balance-pruning") {
        joint_options.worker_balance_pruning =
            ParseOnOff(ValueAfter(i, argc, argv));
      } else if (flag == "--enable-pressure-pruning") {
        joint_options.pressure_pruning.enabled = true;
        slackpipe_options.pressure_pruning.enabled = true;
      } else if (flag == "--pressure-partition-top-k") {
        joint_options.pressure_pruning.partition_top_k =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.partition_top_k =
            joint_options.pressure_pruning.partition_top_k;
      } else if (flag == "--pressure-partition-epsilon") {
        joint_options.pressure_pruning.partition_epsilon =
            std::stod(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.partition_epsilon =
            joint_options.pressure_pruning.partition_epsilon;
      } else if (flag == "--pressure-lambda") {
        joint_options.pressure_pruning.lambda =
            std::stod(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.lambda =
            joint_options.pressure_pruning.lambda;
      } else if (flag == "--pressure-gamma") {
        joint_options.pressure_pruning.gamma =
            std::stod(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.gamma =
            joint_options.pressure_pruning.gamma;
      } else if (flag == "--pressure-alpha") {
        joint_options.pressure_pruning.alpha =
            std::stod(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.alpha =
            joint_options.pressure_pruning.alpha;
      } else if (flag == "--pressure-beam-width") {
        joint_options.pressure_pruning.beam_width =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.beam_width =
            joint_options.pressure_pruning.beam_width;
      } else if (flag == "--pressure-branch-width") {
        joint_options.pressure_pruning.branch_width =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.branch_width =
            joint_options.pressure_pruning.branch_width;
      } else if (flag == "--pressure-generated-partitions") {
        joint_options.pressure_pruning.generated_partitions =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.generated_partitions =
            joint_options.pressure_pruning.generated_partitions;
      } else if (flag == "--pressure-pred-top-k") {
        joint_options.pressure_pruning.predecessor_top_k =
            std::stoll(ValueAfter(i, argc, argv));
        slackpipe_options.pressure_pruning.predecessor_top_k =
            joint_options.pressure_pruning.predecessor_top_k;
      } else if (flag == "--activation-model") {
        activation_options.model =
            slackpipe::ParseActivationModel(ValueAfter(i, argc, argv));
      } else if (flag == "--activation-units-per-layer") {
        activation_options.activation_units_per_layer =
            std::stoll(ValueAfter(i, argc, argv));
      } else if (flag == "--activation-stage-units") {
        activation_options.explicit_stage_activation_units =
            ParseSplit(ValueAfter(i, argc, argv));
      } else if (flag == "--activation-bytes-per-unit") {
        activation_options.activation_bytes_per_unit =
            std::stod(ValueAfter(i, argc, argv));
      } else if (flag == "--activation-cap-mode") {
        activation_options.cap_mode =
            slackpipe::ParseActivationCapMode(ValueAfter(i, argc, argv));
      } else if (flag == "--activation-cap-units") {
        activation_options.activation_cap_units =
            ParseSplit(ValueAfter(i, argc, argv));
      } else if (flag == "--activation-cap-derivation-hash") {
        activation_cap_derivation_hash = ValueAfter(i, argc, argv);
      } else if (flag == "--enforce-activation-cap") {
        activation_options.enforce_activation_cap =
            ParseOptionalBoolFlag(i, argc, argv);
      } else if (flag == "--activation-cap-enforcement") {
        const std::string enforcement = ValueAfter(i, argc, argv);
        if (enforcement == "solver") {
          activation_options.enforce_activation_cap = true;
          activation_cap_enforcement_request = "solver";
        } else if (enforcement == "exact-enumeration" ||
                   enforcement == "exact_enumeration") {
          activation_options.enforce_activation_cap = true;
          activation_cap_enforcement_request = "exact_enumeration";
        } else if (enforcement == "posthoc-only" ||
                   enforcement == "posthoc_only" || enforcement == "posthoc") {
          activation_options.enforce_activation_cap = false;
          activation_cap_enforcement_request = "posthoc_only";
        } else {
          throw slackpipe::Error("unknown activation cap enforcement mode: " +
                                 enforcement);
        }
      } else if (flag == "--emit-activation-trace") {
        activation_options.emit_activation_trace =
            ParseOptionalBoolFlag(i, argc, argv);
      } else if (flag == "--bfs-method") {
        joint_options.bfs_method = ValueAfter(i, argc, argv);
        slackpipe_options.bfs_method = joint_options.bfs_method;
        bfs_method_provided = true;
      } else if (flag == "--incumbent-method") {
        const std::string incumbent_method = ValueAfter(i, argc, argv);
        if (incumbent_method != "slack" && incumbent_method != "canonical" &&
            incumbent_method != "none" &&
            incumbent_method != "heuristic") {
          throw slackpipe::Error(
              "--incumbent-method must be heuristic, slack, canonical, or none");
        }
        joint_options.incumbent_method = incumbent_method;
        incumbent_method_provided = true;
      } else if (flag == "--incumbent-bound") {
        joint_options.incumbent_bound = ParseOnOff(ValueAfter(i, argc, argv));
        incumbent_bound_provided = true;
      } else if (flag == "--incumbent-hints") {
        const bool enabled = ParseOnOff(ValueAfter(i, argc, argv));
        joint_options.incumbent_hints = enabled;
        incumbent_hints_provided = true;
      } else if (flag == "--use-bfs-hints") {
        const bool use_bfs_hints = ParseBool(ValueAfter(i, argc, argv));
        joint_options.use_bfs_hints = use_bfs_hints;
        slackpipe_options.use_bfs_hints = use_bfs_hints;
        use_bfs_hints_provided = true;
      } else if (flag == "--alternating-max-rounds") {
        alternating_max_rounds = std::stoi(ValueAfter(i, argc, argv));
      } else {
        throw slackpipe::Error("unknown flag: " + flag);
      }
    }

    if (output_prefix.empty()) output_prefix = "slackpipe";
    const std::string requested_algorithm = algorithm;
    algorithm = slackpipe::CanonicalizeEvaluationMethodName(algorithm);
    if (octopipe_flags_provided && algorithm != "octopipe-algorithm1-fixed-stage") {
      throw slackpipe::Error("OctoPipe flags require --algorithm octopipe-algorithm1-fixed-stage");
    }
    if (!fixed_order_partition_backend_provided) {
      fixed_order_partition_backend = method;
    }
    if (bfs_method_provided && incumbent_method_provided) {
      throw slackpipe::Error(
          "--bfs-method and --incumbent-method cannot both be supplied");
    }
    if (use_bfs_hints_provided && incumbent_hints_provided) {
      const bool requested_hints =
          slackpipe::IncumbentHintsRequested(joint_options);
      if (requested_hints != joint_options.use_bfs_hints) {
        throw slackpipe::Error(
            "--use-bfs-hints and --incumbent-hints disagree");
      }
    }
    if (joint_options.incumbent_method &&
        *joint_options.incumbent_method == "none") {
      if (!incumbent_bound_provided) {
        joint_options.incumbent_bound = false;
      } else if (joint_options.incumbent_bound) {
        throw slackpipe::Error(
            "--incumbent-method=none cannot be combined with "
            "--incumbent-bound=on");
      }
      if (!incumbent_hints_provided) {
        joint_options.incumbent_hints = false;
      } else if (slackpipe::IncumbentHintsRequested(joint_options)) {
        throw slackpipe::Error(
            "--incumbent-method=none cannot be combined with "
            "--incumbent-hints=on");
      }
    }
    slackpipe::ValidateJointOptimizerMechanismOptions(joint_options);
    if (activation_cap_enforcement_request == "exact_enumeration" &&
        (algorithm != "partition-only-fixed-order" ||
         fixed_order_partition_backend != "enumerate")) {
      throw slackpipe::Error(
          "--activation-cap-enforcement exact-enumeration requires "
          "partition-only-fixed-order with --fixed-order-partition-backend "
          "enumerate");
    }
    if (activation_cap_enforcement_request == "solver" &&
        algorithm == "partition-only-fixed-order" &&
        fixed_order_partition_backend == "enumerate") {
      throw slackpipe::Error(
          "--activation-cap-enforcement solver is incompatible with the "
          "fixed-order enumerate backend; use exact-enumeration for oracle "
          "enumeration");
    }
    options.fixed_order_partition_backend = fixed_order_partition_backend;
    if (!cost_profile_path.empty()) {
      slackpipe::ApplyCostProfileFile(instance, cost_profile_path);
    }
    LogLifecycle(cli_started, instance, "ARGS_PARSED", algorithm,
                 joint_options.time_limit_seconds, joint_options.num_workers);
    instance.Validate();
    slackpipe::ValidateActivationOptions(instance, activation_options);
    if (activation_options.cap_mode ==
            slackpipe::ActivationCapMode::kUniformBaseline &&
        !activation_options.activation_cap_units.empty() &&
        !activation_options.uniform_baseline) {
      slackpipe::ActivationUniformBaseline baseline;
      baseline.partition = slackpipe::UniformSplit(instance);
      baseline.cap_units_per_worker = slackpipe::ResolveExplicitActivationCap(
          instance, activation_options.activation_cap_units);
      baseline.cap_derivation_hash = activation_cap_derivation_hash;
      if (baseline.cap_derivation_hash.empty()) {
        throw slackpipe::Error(
            "--activation-cap-derivation-hash is required when "
            "--activation-cap-units materializes uniform-baseline caps");
      }
      baseline.baseline_run_id =
          "activation-uniform-" + baseline.cap_derivation_hash;
      if (const slackpipe::EvaluationMethodDefinition *definition =
              slackpipe::FindEvaluationMethod(
                  slackpipe::kUniformBreadthFirstMethod)) {
        baseline.method_contract_hash =
            slackpipe::EvaluationMethodContractHash(*definition);
      }
      activation_options.uniform_baseline = baseline;
    }
    if (activation_options.cap_mode ==
            slackpipe::ActivationCapMode::kUniformBaseline &&
        !activation_options.uniform_baseline) {
      activation_options.uniform_baseline =
          slackpipe::DeriveUniformActivationBaseline(instance,
                                                     activation_options);
    }
    options.activation_options = activation_options;
    joint_options.activation_options = activation_options;
    slackpipe_options.activation_options = activation_options;
    LogLifecycle(cli_started, instance, "INSTANCE_CREATED", algorithm,
                 joint_options.time_limit_seconds, joint_options.num_workers);
    const bool algorithm_accepts_joint_hints =
        algorithm == "joint-unrestricted-no-overlap" ||
        algorithm == "schedule-only-uniform" ||
        algorithm == "sequential-partition-then-schedule" ||
        algorithm == "alternating-partition-schedule" ||
        algorithm == "slackpipe";
    const bool algorithm_accepts_joint_mechanism_controls =
        algorithm == "joint-unrestricted-no-overlap" ||
        algorithm == "schedule-only-uniform";
    const bool joint_mechanism_flag_provided =
        incumbent_method_provided || incumbent_bound_provided ||
        incumbent_hints_provided ||
        joint_options.worker_balance_pruning.has_value();
    if (joint_mechanism_flag_provided &&
        !algorithm_accepts_joint_mechanism_controls) {
      throw slackpipe::Error(
          "joint mechanism controls apply only to joint or schedule-only");
    }
    if (use_bfs_hints_provided && !algorithm_accepts_joint_hints) {
      throw slackpipe::Error(
          "--use-bfs-hints applies only to joint, optimize-joint, "
          "schedule-only, sequential, alternating, or slackpipe");
    }
    const bool predecessor_restriction_requested =
        joint_options.pressure_pruning.predecessor_top_k > 0 ||
        slackpipe_options.pressure_pruning.predecessor_top_k > 0;
    if (predecessor_restriction_requested && algorithm_accepts_joint_hints) {
      std::cerr
          << "warning: --pressure-pred-top-k was requested, but production "
          << "joint CP-SAT uses unrestricted NoOverlap; predecessor candidate "
          << "restriction is recorded as inactive.\n";
    }

    auto lifecycle = [&](const slackpipe::LifecycleEvent &event) {
      LogLifecycle(cli_started, instance, event.phase, algorithm,
                   event.configured_solver_limit_seconds, event.solver_threads,
                   event.solver_status, event.effective_solver_limit_seconds,
                   event.detail, event.requested_solver_limit_seconds,
                   event.remaining_global_time_seconds,
                   event.phase_specific_cap_seconds);
    };
    options.lifecycle = lifecycle;
    joint_options.lifecycle = lifecycle;
    slackpipe_options.lifecycle = lifecycle;
    auto request_context = [&](const std::string &requested_method,
                               double effective_time_limit_seconds) {
      slackpipe::CanonicalRequestContext context;
      context.timestamp_utc = invocation_timestamp;
      context.requested_command = requested_command;
      context.requested_method = requested_method;
      context.executable_name = executable_name;
      context.requested_time_limit_seconds = options.time_limit_seconds;
      context.effective_time_limit_seconds = effective_time_limit_seconds;
      context.random_seed = options.random_seed;
      context.solver_threads = options.num_workers;
      return context;
    };
    auto finish = [&](int code) {
      LogLifecycle(cli_started, instance, "CLI_EXIT", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   "", 0.0, "code=" + std::to_string(code));
      return code;
    };

    const bool search_stats_enabled =
        emit_search_stats || !search_stats_json_path.empty();
    if (search_stats_enabled) {
      search_stats.algorithm = algorithm;
      options.search_stats = &search_stats;
      joint_options.search_stats = &search_stats;
      slackpipe_options.search_stats = &search_stats;
    }
    auto finish_search_stats = [&]() {
      if (!search_stats_json_path.empty()) {
        slackpipe::WriteTextFile(search_stats_json_path,
                                 slackpipe::ToJson(instance, search_stats));
      }
      if (emit_search_stats) {
        std::cout << slackpipe::SearchStatsSummary(search_stats) << "\n";
      }
    };

    auto validate_canonical_with_reference =
        [&](const slackpipe::ScheduleSolution &schedule,
            slackpipe::CanonicalResultMetadata &canonical,
            const std::optional<std::vector<slackpipe::Tick>>
                &fixed_partition_reference,
            const slackpipe::ActivationCapConstraintMetadata
                *activation_constraints = nullptr,
            const std::string &activation_enforcement_mode = "") {
          auto apply_activation = [&](bool has_valid_schedule) {
            slackpipe::ActivationAnalysisResult activation =
                has_valid_schedule
                    ? slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
                          instance, schedule, activation_options,
                          activation_constraints != nullptr &&
                              activation_constraints->constraints_added)
                    : slackpipe::MakeActivationAnalysisMetadata(
                          instance, activation_options);
            if (activation_constraints != nullptr) {
              slackpipe::ApplyActivationCapConstraintMetadata(
                  activation, *activation_constraints,
                  activation_enforcement_mode);
            } else if (!activation_enforcement_mode.empty()) {
              slackpipe::ActivationCapConstraintMetadata metadata;
              metadata.model_support_level =
                  slackpipe::ToString(slackpipe::ActivationCapSolverSupport());
              metadata.solver_supported =
                  slackpipe::ActivationCapSolverCanEnforce(activation_options,
                                                           false);
              slackpipe::ApplyActivationCapConstraintMetadata(
                  activation, metadata, activation_enforcement_mode);
            }
            slackpipe::ApplyActivationAnalysis(canonical, activation);
            return activation;
          };
          if (!canonical.outcome.feasible.value_or(false) || !schedule.ok()) {
            (void)apply_activation(false);
            slackpipe::ResultValidationResult skipped;
            skipped.passed = true;
            skipped.error_code = "";
            skipped.message = "non-feasible result has no schedule to validate";
            return skipped;
          }
          slackpipe::ResultValidationInput input =
              slackpipe::ValidationInputFromCanonical(instance, schedule,
                                                      canonical);
          input.fixed_partition_reference = fixed_partition_reference;
          slackpipe::ResultValidationResult validation =
              slackpipe::ValidateResult(input);
          slackpipe::ApplyResultValidation(canonical.outcome, validation);
          if (validation.passed && schedule.ok()) {
            const slackpipe::ActivationAnalysisResult activation =
                apply_activation(true);
            const bool cap_violation =
                activation.activation_cap_satisfied &&
                !activation.activation_cap_satisfied.value_or(true);
            const bool activation_invalid =
                !activation.passed ||
                (activation.activation_cap_enforced && cap_violation) ||
                (activation.activation_cap_enforced_in_solver && cap_violation);
            if (activation_invalid) {
              validation.passed = false;
              validation.error_code = cap_violation
                                          ? "activation_cap_violation"
                                          : "activation_analysis_failed";
              validation.error_category = "activation_memory";
              validation.message =
                  canonical.outcome.result_validation
                      ? canonical.outcome.result_validation->message
                      : canonical.outcome.result_validation_error.value_or(
                            validation.error_code);
            }
          }
          return validation;
        };
    auto validate_canonical =
        [&](const slackpipe::ScheduleSolution &schedule,
            slackpipe::CanonicalResultMetadata &canonical) {
          const std::string mode =
              IsDeterministicUniformFixedOrderMethod(algorithm) &&
                      activation_options.enforce_activation_cap
                  ? std::string("deterministic_postconstruction_check")
                  : std::string();
          return validate_canonical_with_reference(schedule, canonical,
                                                   std::nullopt, nullptr, mode);
        };

    auto invalid_result_exit =
        [&](const slackpipe::ResultValidationResult &validation) {
          std::cerr << validation.error_code;
          if (!validation.message.empty()) {
            std::cerr << ": " << validation.message;
          }
          std::cerr << "\n";
          finish_search_stats();
          return finish(2);
        };

    auto emit_plan_after_validation =
        [&](const slackpipe::ScheduleSolution &schedule,
            const std::string &solver_status) {
          if (emit_plan_path.empty()) return;
          slackpipe::WriteMegatronSlackPipePlanFile(emit_plan_path, instance,
                                                    schedule, solver_status);
        };

    auto emit_plan_after_predecessor_validation =
        [&](const slackpipe::ScheduleSolution &schedule,
            const slackpipe::MachinePredecessors &predecessors,
            const std::string &solver_status, bool fifo_ordering) {
          if (emit_plan_path.empty()) return;
          const slackpipe::EvaluationResult predecessor_validation =
              slackpipe::EvaluateScheduleWithPredecessors(
                  instance, schedule.split, predecessors, fifo_ordering);
          if (!predecessor_validation.schedule.ok()) {
            std::ostringstream msg;
            msg << "cannot emit SlackPipe plan: predecessor validation failed";
            if (!predecessor_validation.schedule.validation_errors.empty()) {
              msg << ": "
                  << predecessor_validation.schedule.validation_errors.front();
            }
            throw slackpipe::Error(msg.str());
          }
          if (predecessor_validation.schedule.makespan != schedule.makespan) {
            std::ostringstream msg;
            msg << "cannot emit SlackPipe plan: predecessor makespan "
                << predecessor_validation.schedule.makespan
                << " does not match schedule makespan " << schedule.makespan;
            throw slackpipe::Error(msg.str());
          }
          slackpipe::WriteMegatronSlackPipePlanFile(emit_plan_path, instance,
                                                    schedule, solver_status);
        };

    auto dump_interleavings =
        [&](const slackpipe::ScheduleSolution &schedule,
            const slackpipe::InterleavingRunMetadata &metadata) {
          if (interleaving_csv_path.empty() &&
              interleaving_summary_csv_path.empty()) {
            return;
          }
          if (!schedule.ok()) {
            throw slackpipe::Error(
                "cannot dump interleaving CSV for invalid schedule");
          }
          const slackpipe::InterleavingStats stats =
              slackpipe::CollectInterleavingStats(instance, schedule);
          if (!interleaving_csv_path.empty()) {
            slackpipe::WriteTextFile(
                interleaving_csv_path,
                slackpipe::ToInterleavingEventsCsv(instance, metadata, stats));
          }
          if (!interleaving_summary_csv_path.empty()) {
            slackpipe::WriteTextFile(
                interleaving_summary_csv_path,
                slackpipe::ToInterleavingSummaryCsv(instance, metadata, stats));
          }
        };

    slackpipe::Deadline schedule_only_deadline(options.time_limit_seconds);
    auto fixed_split_for_schedule_only = [&]() {
      const auto started = Clock::now();
      FixedPartitionResolution resolution;
      if (!split.empty()) {
        slackpipe::ValidateSplit(instance, split);
        if (fixed_partition_source.empty()) fixed_partition_source = "manual";
        resolution.split = split;
        resolution.reference_source = "supplied input";
        resolution.status = "SUPPLIED";
        resolution.runtime_seconds = Since(started);
        return resolution;
      }
      if (fixed_partition_source.empty() ||
          fixed_partition_source == "uniform") {
        fixed_partition_source = "uniform";
        resolution.split = slackpipe::UniformSplit(instance);
        resolution.reference_source = "uniform_deterministic";
        resolution.status = "DETERMINISTIC";
        resolution.runtime_seconds = Since(started);
        return resolution;
      }
      if (fixed_partition_source == "load-balanced") {
        resolution.split = slackpipe::LoadBalancedSplit(instance);
        resolution.reference_source = "load-balanced construction";
        resolution.status = "DETERMINISTIC";
        resolution.runtime_seconds = Since(started);
        return resolution;
      }
      if (fixed_partition_source == "partition-only") {
        slackpipe::BfsSplitOptimizerOptions partition_options = options;
        partition_options.enumeration_threshold =
            options.time_limit_seconds > 0.0
                ? 0
                : partition_options.enumeration_threshold;
        partition_options.time_limit_seconds =
            options.time_limit_seconds > 0.0
                ? schedule_only_deadline.clamp_solver_limit(
                      options.time_limit_seconds *
                      slackpipe::kScheduleOnlyPartitionBudgetFraction)
                : 0.0;
        resolution.phase_limit_seconds = partition_options.time_limit_seconds;
        if (options.time_limit_seconds > 0.0 &&
            resolution.phase_limit_seconds <= 0.0) {
          fixed_partition_source = "partition_only_within_global_budget";
          resolution.split = slackpipe::UniformSplit(instance);
          resolution.reference_source = fixed_partition_source;
          resolution.status = "NOT_RUN";
          resolution.runtime_seconds = Since(started);
          return resolution;
        }
        slackpipe::BfsSplitOptimizationResult partition_only =
            slackpipe::OptimizeBfsSplitAuto(instance, partition_options);
        if (!(partition_only.status == "OPTIMAL" ||
              partition_only.status == "FEASIBLE")) {
          throw slackpipe::Error(
              "partition-only fixed partition source did not solve");
        }
        if (partition_only.machine_orders !=
            slackpipe::BreadthFirstOrders(instance)) {
          throw slackpipe::Error(
              "PARTITION_ONLY_ORDER_MUTATION canonical order changed");
        }
        fixed_partition_source = "partition_only_within_global_budget";
        resolution.split = partition_only.split;
        resolution.reference_source = fixed_partition_source;
        resolution.status = partition_only.status;
        resolution.cp_sat_models_solved =
            partition_only.method.find("cpsat") == std::string::npos ? 0 : 1;
        resolution.runtime_seconds = Since(started);
        return resolution;
      }
      throw slackpipe::Error("unknown fixed partition source: " +
                             fixed_partition_source);
    };

    if (!heuristic_trace_path.empty() && algorithm != "heuristic")
      throw slackpipe::Error("--heuristic-trace is standalone-only; embedded hints retain summary metadata");
    if (algorithm == "heuristic") {
      if (!split.empty() || activation_options.enforce_activation_cap ||
          !joint_options.fifo_ordering || slackpipe_options.pressure_pruning.enabled ||
          options.num_workers != 1 || !fixed_partition_source.empty())
        throw slackpipe::Error("standalone heuristic requires uniform start, FIFO, one search thread, no activation cap/pressure pruning");
      auto heuristic = joint_options.heuristic_options;
      heuristic.partition_restriction = slackpipe::MakePartitionRestriction(
          instance, slackpipe::UniformSplit(instance), slackpipe_options);
      heuristic.random_seed = options.random_seed;
      if (options.time_limit_seconds > 0)
        heuristic.time_limit_seconds = std::min(heuristic.time_limit_seconds, options.time_limit_seconds);
      const auto balance = slackpipe::ComputeJointWorkerBalanceConstraint(instance, joint_options);
      if (balance.enabled) heuristic.allowed_partition = [&, balance](const auto& p) {
        for (auto load : slackpipe::WorkerLayerTotals(instance, p))
          if (load < balance.lower_bound || load > balance.upper_bound) return false;
        return true;
      };
      auto result = slackpipe::BuildHeuristicIncumbent(instance, heuristic);
      const auto validation = slackpipe::ValidateScheduleSolutionIndependent(
          instance, result.schedule, result.status, "heuristic");
      if (!validation.passed) return invalid_result_exit(validation);
      slackpipe::WriteTextFile(output_prefix + ".json", slackpipe::ToJson(instance, result));
      slackpipe::WriteTextFile(output_prefix + ".csv", slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(output_prefix + ".orders.txt", slackpipe::ToOrdersText(instance, result.machine_orders));
      slackpipe::WriteTextFile(output_prefix + ".svg", slackpipe::ToSvg(instance, result.schedule));
      if (!heuristic_trace_path.empty())
        slackpipe::WriteTextFile(heuristic_trace_path, result.heuristic_metadata_json);
      emit_plan_after_validation(result.schedule, result.status);
      dump_interleavings(result.schedule, {result.status, result.makespan_ticks, std::nullopt});
      std::cout << "method=heuristic status=" << result.status << " makespan=" << result.makespan_ticks
                << " search_seconds=" << result.wall_time_seconds
                << " evaluations=" << result.hint_candidates_simulated
                << " stop=" << result.hint_termination_reason << "\n";
      finish_search_stats();
      return finish(0);
    }
    if (algorithm == "octopipe-algorithm1-fixed-stage") {
      if (!joint_options.fifo_ordering || activation_options.enforce_activation_cap ||
          slackpipe_options.pressure_pruning.enabled ||
          slackpipe_options.worker_balance_tolerance_percent >= 0 ||
          slackpipe_options.worker_balance_tolerance_layers ||
          !fixed_partition_source.empty() ||
          (bfs_method_provided && slackpipe_options.bfs_method != "uniform") ||
          incumbent_method_provided) {
        throw slackpipe::Error("OctoPipe requires FIFO and does not support activation caps, "
                               "pressure/worker-balance pruning, or --fixed-partition-source; "
                               "use --split for an initial partition and --bfs-method uniform");
      }
      if (split.empty()) split = slackpipe::UniformSplit(instance);
      octopipe_options.time_limit_seconds = options.time_limit_seconds;
      if (options.log_search_progress || search_stats_enabled) {
        search_stats.worker_partitions_enabled = octopipe_options.tune_placement;
        search_stats.worker_partitions_reason = "OctoPipe whole-stage swaps; disabled by --octopipe-fixed-placement";
        octopipe_options.progress = [&](const slackpipe::OctoPipeIteration& log,
                                       const slackpipe::OctoPipeState&) {
          search_stats.candidate_schedules_extracted += log.candidates;
          search_stats.candidate_schedules_deterministically_evaluated += log.candidates;
          search_stats.candidate_schedules_accepted += log.valid_candidates;
          search_stats.candidate_schedules_rejected += log.candidates - log.valid_candidates;
          if (!options.log_search_progress) return;
          std::cerr << "OCTOPIPE iter=" << log.iteration << " elapsed=" << log.elapsed_seconds
                    << " makespan=" << log.input_makespan << " delta_b=" << log.metrics.delta_b
                    << " boundary_bubble=" << log.metrics.boundary_sum
                    << " residual_bubble=" << log.metrics.residual_sum
                    << " t_layer=" << log.t_layer << " phase=" << slackpipe::ToString(log.phase)
                    << " candidates=" << log.candidates << " valid=" << log.valid_candidates
                    << " accepted=" << log.accepted << " best=" << log.best_makespan << "\n";
        };
      }
      const auto result = slackpipe::TuneOctoPipeAlgorithm1(
          instance, split, slackpipe::BreadthFirstOrders(instance), octopipe_options);
      instance = result.best.instance;
      const auto& schedule = result.best.schedule;
      slackpipe::CanonicalSemantics semantics;
      semantics.canonical_method = algorithm;
      semantics.actual_solver_path = "octopipe_algorithm1_fixed_stage_common_evaluator";
      semantics.partition_decision = "local_contiguous_boundary_moves";
      semantics.schedule_decision = "local_fb_only";
      semantics.partition_optimized = true;
      semantics.schedule_optimized = true;
      semantics.cp_sat_launched = false;
      semantics.cp_sat_models_solved = 0;
      auto canonical = slackpipe::BuildCanonicalResultMetadata(
          instance, request_context(requested_algorithm, options.time_limit_seconds), semantics,
          slackpipe::OutcomeFromSchedule(schedule, "FEASIBLE", Since(cli_started)),
          schedule.split, schedule.orders);
      const auto validation = validate_canonical(schedule, canonical);
      if (!validation.passed) return invalid_result_exit(validation);
      slackpipe::WriteTextFile(output_prefix + ".json", slackpipe::ToJson(
          instance, schedule, canonical, search_stats_enabled ? &search_stats : nullptr));
      slackpipe::WriteTextFile(output_prefix + ".csv", slackpipe::ToCsv(instance, schedule));
      slackpipe::WriteTextFile(output_prefix + ".orders.txt", slackpipe::ToOrdersText(instance, schedule.orders));
      slackpipe::WriteTextFile(output_prefix + ".svg", slackpipe::ToSvg(instance, schedule));
      emit_plan_after_predecessor_validation(schedule,
          slackpipe::ExtractMachinePredecessors(instance, schedule.orders), "FEASIBLE", true);
      dump_interleavings(schedule, {"FEASIBLE", schedule.makespan, std::nullopt});
      std::cout << "algorithm=" << algorithm << " initial_N=" << instance.stages
                << " final_N=" << schedule.split.size() << " initial_makespan=" << result.initial_makespan
                << " makespan=" << schedule.makespan << " iterations=" << result.iterations
                << " tuning_seconds=" << result.tuning_seconds << "\n";
      finish_search_stats();
      return finish(0);
    }

    if (IsDeterministicUniformFixedOrderMethod(algorithm)) {
      if (!split.empty() && split != slackpipe::UniformSplit(instance)) {
        throw slackpipe::Error(algorithm +
                               " does not accept a non-uniform --split");
      }
      split = slackpipe::UniformSplit(instance);
      const slackpipe::MachineOrders orders =
          algorithm == slackpipe::kUniformBreadthFirstMethod
              ? slackpipe::BreadthFirstOrders(instance)
              : slackpipe::InterleavedOneFOneBOrders(instance);
      if (search_stats_enabled) {
        ++search_stats.candidate_schedules_extracted;
        ++search_stats.candidate_schedules_deterministically_evaluated;
      }
      slackpipe::EvaluationResult result =
          slackpipe::EvaluateSchedule(instance, split, orders);
      if (search_stats_enabled) {
        if (result.schedule.ok()) {
          ++search_stats.candidate_schedules_accepted;
        } else {
          ++search_stats.candidate_schedules_rejected;
        }
      }

      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.schedule.ok() ? "OK" : "INVALID", 0.0, output_prefix);
      slackpipe::CanonicalResultMetadata canonical =
          slackpipe::BuildCanonicalResultMetadata(
              instance,
              request_context(requested_algorithm, options.time_limit_seconds),
              algorithm == slackpipe::kUniformBreadthFirstMethod
                  ? slackpipe::SemanticsForUniformFixedOrderBaseline()
                  : slackpipe::SemanticsForUniformInterleavedOneFOneB(),
              slackpipe::OutcomeFromSchedule(
                  result.schedule,
                  result.schedule.ok() ? "FEASIBLE" : "INVALID"),
              split, orders);
      const slackpipe::ResultValidationResult validation =
          validate_canonical(result.schedule, canonical);
      slackpipe::WriteTextFile(
          output_prefix + ".json",
          slackpipe::ToJson(instance, result.schedule, canonical,
                            search_stats_enabled ? &search_stats : nullptr));
      slackpipe::WriteTextFile(output_prefix + ".csv",
                               slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(output_prefix + ".orders.txt",
                               slackpipe::ToOrdersText(instance, orders));
      slackpipe::WriteTextFile(output_prefix + ".svg",
                               slackpipe::ToSvg(instance, result.schedule));
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.schedule.ok() ? "OK" : "INVALID", 0.0, output_prefix);

      if (!result.schedule.ok()) {
        for (const std::string &error : result.schedule.validation_errors) {
          std::cerr << error << "\n";
        }
        finish_search_stats();
        return finish(2);
      }
      if (!validation.passed) return invalid_result_exit(validation);
      emit_plan_after_validation(result.schedule, "FEASIBLE");
      dump_interleavings(result.schedule,
                         slackpipe::InterleavingRunMetadata{
                             "OK", result.schedule.makespan, std::nullopt});
      std::cout << "makespan=" << result.schedule.makespan << "\n";
      finish_search_stats();
      return finish(0);
    }

    if (algorithm == "partition-only-fixed-order") {
      if (!split.empty()) {
        throw slackpipe::Error(
            "--split must not be provided for partition-only-fixed-order");
      }
      const slackpipe::MachineOrders fixed_orders =
          slackpipe::BreadthFirstOrders(instance);
      slackpipe::BfsSplitOptimizationResult result =
          slackpipe::OptimizePartitionForFixedOrder(instance, fixed_orders,
                                                    options);

      result.canonical = slackpipe::BuildCanonicalResultMetadata(
          instance,
          request_context(requested_algorithm, options.time_limit_seconds),
          slackpipe::SemanticsForPartitionOnlyFixedOrder(result),
          slackpipe::OutcomeFromBfsResult(result), result.split,
          result.machine_orders);
      const slackpipe::ResultValidationResult validation =
          validate_canonical_with_reference(
              result.schedule, *result.canonical, std::nullopt,
              &result.activation_cap_constraints,
              result.activation_cap_constraints.enforced_by_enumeration
                  ? (result.enumeration_proved_optimal
                         ? std::string("exact_enumeration")
                         : std::string("partial_enumeration"))
                  : std::string());
      if (!validation.passed &&
          slackpipe::IsTerminalFeasibleStatus(result.status)) {
        result.status = "INVALID_RESULT";
      }
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.status, 0.0, output_prefix);
      slackpipe::WriteTextFile(output_prefix + ".json",
                               slackpipe::ToJson(instance, result));
      slackpipe::WriteTextFile(output_prefix + ".csv",
                               slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(
          output_prefix + ".orders.txt",
          slackpipe::ToOrdersText(instance, result.machine_orders));
      slackpipe::WriteTextFile(output_prefix + ".svg",
                               slackpipe::ToSvg(instance, result.schedule));
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.status, 0.0, output_prefix);

      if (result.machine_orders != slackpipe::BreadthFirstOrders(instance)) {
        throw slackpipe::Error(
            "PARTITION_ONLY_ORDER_MUTATION canonical order changed");
      }

      if (result.status == "UNAVAILABLE") {
        std::cerr << (result.diagnostic.empty()
                          ? "fixed-order partition backend unavailable"
                          : result.diagnostic)
                  << "\n";
        finish_search_stats();
        return finish(3);
      }
      if (!validation.passed) return invalid_result_exit(validation);
      if (options.require_optimal && !result.proven_optimal) {
        std::cerr << "optimizer did not prove optimality; status="
                  << result.status << "\n";
        finish_search_stats();
        return finish(4);
      }
      if (result.schedule.ok()) {
        emit_plan_after_validation(result.schedule, result.status);
        dump_interleavings(result.schedule,
                           slackpipe::InterleavingRunMetadata{
                               result.status, result.makespan_ticks,
                               static_cast<double>(result.best_bound_ticks)});
      }
      std::cout << "status=" << result.status << " proven_optimal="
                << (result.proven_optimal ? "true" : "false")
                << " makespan=" << result.makespan_ticks << "\n";
      finish_search_stats();
      return finish(0);
    }

    if (algorithm == "joint-unrestricted-no-overlap") {
      if (!split.empty()) {
        throw slackpipe::Error(
            "--split must not be provided for joint-unrestricted-no-overlap");
      }
      slackpipe::JointOptimizationResult result =
          slackpipe::OptimizeJointSplitAndScheduleCpSat(instance,
                                                        joint_options);
      result.canonical = slackpipe::BuildCanonicalResultMetadata(
          instance,
          request_context(requested_algorithm, result.joint_budget_seconds),
          slackpipe::SemanticsForJointUnrestrictedNoOverlap(
              result, predecessor_restriction_requested),
          slackpipe::OutcomeFromJointResult(result), result.split,
          result.machine_orders);
      const slackpipe::ResultValidationResult validation =
          validate_canonical_with_reference(result.schedule, *result.canonical,
                                            std::nullopt,
                                            &result.activation_cap_constraints);
      if (!validation.passed &&
          slackpipe::IsTerminalFeasibleStatus(result.status)) {
        result.status = "INVALID_RESULT";
      }
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.status, 0.0, output_prefix);
      slackpipe::WriteTextFile(output_prefix + ".json",
                               slackpipe::ToJson(instance, result));
      slackpipe::WriteTextFile(output_prefix + ".csv",
                               slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(
          output_prefix + ".orders.txt",
          slackpipe::ToOrdersText(instance, result.machine_orders));
      slackpipe::WriteTextFile(output_prefix + ".svg",
                               slackpipe::ToSvg(instance, result.schedule));
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.status, 0.0, output_prefix);

      if (result.status == "UNAVAILABLE") {
        std::cerr << (result.diagnostic.empty() ? "joint solver unavailable"
                                                : result.diagnostic)
                  << "\n";
        finish_search_stats();
        return finish(3);
      }
      if (joint_options.require_optimal && !result.proven_optimal) {
        std::cerr << "joint optimizer did not prove optimality; status="
                  << result.status << "\n";
        finish_search_stats();
        return finish(4);
      }
      if (!validation.passed) return invalid_result_exit(validation);
      emit_plan_after_validation(result.schedule, result.status);
      if (result.schedule.ok()) {
        dump_interleavings(
            result.schedule,
            slackpipe::InterleavingRunMetadata{
                result.status, result.makespan_ticks, result.best_bound_ticks});
      }
      std::cout << "status=" << result.status << " proven_optimal="
                << (result.proven_optimal ? "true" : "false")
                << " solution_source=" << result.solution_source
                << " hints_effective="
                << (result.hints_effective ? "true" : "false")
                << " fallback_used="
                << (result.fallback_used ? "true" : "false")
                << " makespan=" << result.makespan_ticks << "\n";
      finish_search_stats();
      return finish(0);
    }

    if (algorithm == "sequential-partition-then-schedule" ||
        algorithm == "alternating-partition-schedule") {
      if (!split.empty()) {
        throw slackpipe::Error("--split must not be provided for " + algorithm);
      }
      slackpipe::AlternatingOptimizerOptions alternating_options;
      alternating_options.time_limit_seconds = options.time_limit_seconds;
      alternating_options.num_workers = options.num_workers;
      alternating_options.random_seed = options.random_seed;
      alternating_options.require_optimal = options.require_optimal;
      alternating_options.log_search_progress = options.log_search_progress;
      alternating_options.enumeration_threshold = options.enumeration_threshold;
      alternating_options.use_bfs_hints = joint_options.use_bfs_hints;
      alternating_options.fifo_ordering = joint_options.fifo_ordering;
      alternating_options.symmetry_break_f0_fifo =
          joint_options.symmetry_break_f0_fifo;
      alternating_options.fixed_order_partition_backend =
          fixed_order_partition_backend_provided ? fixed_order_partition_backend
                                                 : std::string("cpsat");
      alternating_options.max_rounds = alternating_max_rounds;
      alternating_options.activation_options = activation_options;

      slackpipe::AlternatingOptimizationResult result =
          algorithm == "sequential-partition-then-schedule"
              ? slackpipe::OptimizeSequentialPartitionThenSchedule(
                    instance, alternating_options)
              : slackpipe::OptimizeAlternatingPartitionSchedule(
                    instance, alternating_options);
      slackpipe::CanonicalSemantics semantics =
          algorithm == "sequential-partition-then-schedule"
              ? slackpipe::SemanticsForSequentialPartitionThenSchedule(
                    result.cp_sat_models_solved)
              : slackpipe::SemanticsForAlternatingPartitionSchedule(
                    result.cp_sat_models_solved);
      result.canonical = slackpipe::BuildCanonicalResultMetadata(
          instance,
          request_context(requested_algorithm, options.time_limit_seconds),
          semantics, slackpipe::OutcomeFromAlternatingResult(result),
          result.split, result.machine_orders);
      slackpipe::ApplyAlternatingCanonicalFields(result, *result.canonical);
      const slackpipe::ResultValidationResult validation =
          validate_canonical_with_reference(result.schedule, *result.canonical,
                                            std::nullopt,
                                            &result.activation_cap_constraints);
      if (!validation.passed &&
          slackpipe::IsTerminalFeasibleStatus(result.status)) {
        result.status = "INVALID_RESULT";
      }
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                   options.time_limit_seconds, options.num_workers,
                   result.status, 0.0, output_prefix);
      slackpipe::WriteTextFile(output_prefix + ".json",
                               slackpipe::ToJson(instance, result));
      slackpipe::WriteTextFile(output_prefix + ".csv",
                               slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(
          output_prefix + ".orders.txt",
          slackpipe::ToOrdersText(instance, result.machine_orders));
      slackpipe::WriteTextFile(output_prefix + ".svg",
                               slackpipe::ToSvg(instance, result.schedule));
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                   options.time_limit_seconds, options.num_workers,
                   result.status, 0.0, output_prefix);

      if (result.status == "UNAVAILABLE") {
        std::cerr << (result.diagnostic.empty() ? algorithm + " unavailable"
                                                : result.diagnostic)
                  << "\n";
        finish_search_stats();
        return finish(3);
      }
      if (alternating_options.require_optimal && !result.proven_optimal) {
        std::cerr << algorithm
                  << " did not prove optimality; status=" << result.status
                  << "\n";
        finish_search_stats();
        return finish(4);
      }
      if (!validation.passed) return invalid_result_exit(validation);
      emit_plan_after_validation(result.schedule, result.status);
      if (result.schedule.ok()) {
        dump_interleavings(
            result.schedule,
            slackpipe::InterleavingRunMetadata{
                result.status, result.makespan_ticks, result.best_bound_ticks});
      }
      std::cout << "status=" << result.status
                << " makespan=" << result.makespan_ticks
                << " completed_rounds=" << result.alternating_completed_rounds
                << " convergence=" << result.alternating_convergence_reason
                << "\n";
      finish_search_stats();
      return finish(0);
    }

    if (algorithm == "schedule-only-uniform") {
      if (!fixed_partition_source.empty() &&
          fixed_partition_source != "uniform") {
        throw slackpipe::Error(
            "schedule-only-uniform requires --fixed-partition-source uniform");
      }
      if (!split.empty() && split != slackpipe::UniformSplit(instance)) {
        throw slackpipe::Error(
            "schedule-only-uniform does not accept a non-uniform --split");
      }
      const FixedPartitionResolution fixed_partition =
          fixed_split_for_schedule_only();
      const std::vector<slackpipe::Tick> &fixed_split = fixed_partition.split;
      slackpipe::JointOptimizerOptions schedule_options = joint_options;
      schedule_options.time_limit_seconds =
          options.time_limit_seconds > 0.0
              ? schedule_only_deadline.clamp_solver_limit(0.0)
              : 0.0;
      slackpipe::JointOptimizationResult result =
          options.time_limit_seconds > 0.0 && schedule_only_deadline.expired()
              ? slackpipe::BuildScheduleOnlyFixedSplitDeadlineFallback(
                    instance, fixed_split, schedule_options,
                    schedule_only_deadline.elapsed_seconds(),
                    "global_deadline_expired_before_cp_sat")
              : slackpipe::OptimizeScheduleForFixedSplitCpSat(
                    instance, fixed_split, schedule_options);
      result.cp_sat_models_solved += fixed_partition.cp_sat_models_solved;
      slackpipe::CanonicalOutcome outcome =
          slackpipe::OutcomeFromJointResult(result);
      if (fixed_partition.runtime_seconds > 0.0) {
        outcome.reference_runtime_seconds = fixed_partition.runtime_seconds;
      }
      if (fixed_partition.phase_limit_seconds > 0.0) {
        outcome.phase_budget.reference_phase_limit_seconds =
            fixed_partition.phase_limit_seconds;
      }
      slackpipe::CanonicalPhaseBudget partition_phase;
      partition_phase.phase = "fixed_partition_reference";
      if (fixed_partition.phase_limit_seconds > 0.0) {
        partition_phase.effective_limit_seconds =
            fixed_partition.phase_limit_seconds;
      }
      if (fixed_partition.runtime_seconds > 0.0) {
        partition_phase.runtime_seconds = fixed_partition.runtime_seconds;
      }
      if (!fixed_partition.status.empty()) {
        partition_phase.status = fixed_partition.status;
      }
      outcome.phase_budget.phases.insert(outcome.phase_budget.phases.begin(),
                                         partition_phase);
      result.canonical = slackpipe::BuildCanonicalResultMetadata(
          instance,
          request_context(requested_algorithm, result.joint_budget_seconds),
          slackpipe::SemanticsForScheduleOnlyFixedSplit(
              fixed_partition.reference_source, result,
              predecessor_restriction_requested),
          outcome, result.split, result.machine_orders);
      if ((result.status == "OPTIMAL" || result.status == "FEASIBLE") &&
          result.split != fixed_split) {
        throw slackpipe::Error(
            "SCHEDULE_ONLY_PARTITION_MUTATION fixed partition changed");
      }
      const slackpipe::ResultValidationResult validation =
          validate_canonical_with_reference(result.schedule, *result.canonical,
                                            fixed_split,
                                            &result.activation_cap_constraints);
      if (!validation.passed &&
          slackpipe::IsTerminalFeasibleStatus(result.status)) {
        result.status = "INVALID_RESULT";
      }
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                   schedule_options.time_limit_seconds,
                   schedule_options.num_workers, result.status, 0.0,
                   output_prefix);
      slackpipe::WriteTextFile(output_prefix + ".json",
                               slackpipe::ToJson(instance, result));
      slackpipe::WriteTextFile(output_prefix + ".csv",
                               slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(
          output_prefix + ".orders.txt",
          slackpipe::ToOrdersText(instance, result.machine_orders));
      slackpipe::WriteTextFile(output_prefix + ".svg",
                               slackpipe::ToSvg(instance, result.schedule));
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                   schedule_options.time_limit_seconds,
                   schedule_options.num_workers, result.status, 0.0,
                   output_prefix);

      if (result.status == "UNAVAILABLE") {
        std::cerr << (result.diagnostic.empty()
                          ? "schedule-only solver unavailable"
                          : result.diagnostic)
                  << "\n";
        finish_search_stats();
        return finish(3);
      }
      if (schedule_options.require_optimal && !result.proven_optimal) {
        std::cerr << "schedule-only optimizer did not prove optimality; status="
                  << result.status << "\n";
        finish_search_stats();
        return finish(4);
      }
      if (!validation.passed) return invalid_result_exit(validation);
      emit_plan_after_validation(result.schedule, result.status);
      if (result.schedule.ok()) {
        dump_interleavings(
            result.schedule,
            slackpipe::InterleavingRunMetadata{
                result.status, result.makespan_ticks, result.best_bound_ticks});
      }
      std::cout
          << "status=" << result.status
          << " proven_optimal=" << (result.proven_optimal ? "true" : "false")
          << " fixed_partition_source=" << fixed_partition.reference_source
          << " solution_source=" << result.solution_source
          << " hints_effective=" << (result.hints_effective ? "true" : "false")
          << " fallback_used=" << (result.fallback_used ? "true" : "false")
          << " makespan=" << result.makespan_ticks << "\n";
      finish_search_stats();
      return finish(0);
    }

    if (algorithm == "slackpipe") {
      if (!split.empty()) {
        throw slackpipe::Error("--split must not be provided for slackpipe");
      }
      slackpipe::SlackPipeResult result =
          slackpipe::SolveCanonicalSlackPipe(instance, slackpipe_options);
      result.canonical = slackpipe::BuildCanonicalResultMetadata(
          instance,
          request_context(requested_algorithm,
                          result.joint_remaining_budget_seconds),
          slackpipe::SemanticsForSlackPipe(slackpipe_options.split_mode, result,
                                           predecessor_restriction_requested),
          slackpipe::OutcomeFromSlackPipeResult(result), result.split,
          result.machine_orders);
      const slackpipe::ResultValidationResult validation =
          validate_canonical_with_reference(
              result.schedule, *result.canonical,
              slackpipe::SlackPipeModeFixesFullPartition(
                  slackpipe_options.split_mode)
                  ? std::optional<std::vector<slackpipe::Tick>>(
                        result.bfs.split)
                  : std::nullopt,
              &result.activation_cap_constraints);
      if (!validation.passed &&
          slackpipe::IsTerminalFeasibleStatus(result.status)) {
        result.status = "INVALID_RESULT";
      }
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.status, 0.0, output_prefix);
      slackpipe::WriteTextFile(output_prefix + ".json",
                               slackpipe::ToJson(instance, result));
      slackpipe::WriteTextFile(output_prefix + ".csv",
                               slackpipe::ToCsv(instance, result.schedule));
      slackpipe::WriteTextFile(
          output_prefix + ".orders.txt",
          slackpipe::ToOrdersText(instance, result.machine_orders));
      slackpipe::WriteTextFile(output_prefix + ".svg",
                               slackpipe::ToSvg(instance, result.schedule));
      LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                   joint_options.time_limit_seconds, joint_options.num_workers,
                   result.status, 0.0, output_prefix);

      if (result.status == "UNAVAILABLE") {
        std::cerr << (result.diagnostic.empty() ? "slackpipe unavailable"
                                                : result.diagnostic)
                  << "\n";
        finish_search_stats();
        return finish(3);
      }
      if (slackpipe_options.require_optimal && !result.proven_optimal) {
        std::cerr << "SlackPipe did not prove optimality; status="
                  << result.status << "\n";
        finish_search_stats();
        return finish(4);
      }
      if (!validation.passed) return invalid_result_exit(validation);
      emit_plan_after_predecessor_validation(
          result.schedule, result.machine_predecessors, result.status,
          slackpipe_options.fifo_ordering);
      if (result.schedule.ok()) {
        dump_interleavings(
            result.schedule,
            slackpipe::InterleavingRunMetadata{
                result.status, result.makespan_ticks, result.best_bound_ticks});
      }
      std::cout << "status=" << result.status
                << " makespan=" << result.makespan_ticks
                << " cp_sat_models_solved=" << result.cp_sat_models_solved
                << " joint_solution_source=" << result.joint_solution_source
                << " joint_hints_effective="
                << (result.joint_hints_effective ? "true" : "false")
                << " joint_fallback_used="
                << (result.joint_fallback_used ? "true" : "false")
                << " proven_global_optimal="
                << (result.proven_global_optimal ? "true" : "false") << "\n";
      finish_search_stats();
      return finish(0);
    }

    throw slackpipe::Error("unknown algorithm: " + algorithm);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    LogLifecycle(cli_started, instance, "CLI_EXIT", algorithm,
                 joint_options.time_limit_seconds, joint_options.num_workers,
                 "ERROR", 0.0, "exception");
    if (!output_prefix.empty()) {
      const bool diagnostic_hints_requested =
          algorithm == "slackpipe" ? slackpipe_options.use_bfs_hints
                                   : joint_options.use_bfs_hints;
      const std::string diagnostic_bfs_method =
          algorithm == "slackpipe" ? slackpipe_options.bfs_method
                                   : joint_options.bfs_method;
      std::ostringstream diagnostic;
      diagnostic << "{\n"
                 << "  \"status\": \"ERROR\",\n"
                 << "  \"algorithm\": \"" << JsonEscape(algorithm) << "\",\n"
                 << "  \"message\": \"" << JsonEscape(error.what()) << "\",\n"
                 << "  \"elapsed_seconds\": " << Since(cli_started) << ",\n"
                 << "  \"microbatches\": " << instance.microbatches << ",\n"
                 << "  \"stages\": " << instance.stages << ",\n"
                 << "  \"workers\": " << instance.workers << ",\n"
                 << "  \"total_layers\": " << instance.total_layers << ",\n"
                 << "  \"time_limit_seconds\": "
                 << joint_options.time_limit_seconds << ",\n"
                 << "  \"num_workers\": " << joint_options.num_workers << ",\n"
                 << "  \"incumbent_method_requested\": \""
                 << JsonEscape(diagnostic_bfs_method) << "\",\n"
                 << "  \"incumbent_method_effective\": \"none\",\n"
                 << "  \"bfs_incumbent_method_requested\": \""
                 << JsonEscape(diagnostic_bfs_method) << "\",\n"
                 << "  \"bfs_incumbent_method_effective\": \"none\",\n"
                 << "  \"incumbent_source\": \"none\",\n"
                 << "  \"incumbent_feasible\": false,\n"
                 << "  \"incumbent_primary_objective\": 0,\n"
                 << "  \"incumbent_hybrid_min_slack\": 0,\n"
                 << "  \"incumbent_baseline_primary_objective\": 0,\n"
                 << "  \"incumbent_baseline_hybrid_min_slack\": 0,\n"
                 << "  \"incumbent_improved_over_baseline\": false,\n"
                 << "  \"incumbent_hybrid_stage_scores\": [],\n"
                 << "  \"incumbent_hybrid_bottleneck_stages\": [],\n"
                 << "  \"horizon_source\": \"none\",\n"
                 << "  \"hint_budget_seconds\": 0,\n"
                 << "  \"hint_elapsed_seconds\": 0,\n"
                 << "  \"hint_iterations\": 0,\n"
                 << "  \"hint_candidates_generated\": 0,\n"
                 << "  \"hint_candidates_simulated\": 0,\n"
                 << "  \"hint_partition_moves_accepted\": 0,\n"
                 << "  \"hint_interleaving_moves_accepted\": 0,\n"
                 << "  \"hint_deadline_reached\": false,\n"
                 << "  \"hint_termination_reason\": \"error\",\n"
                 << "  \"hints_requested\": "
                 << (diagnostic_hints_requested ? "true" : "false") << ",\n"
                 << "  \"hints_effective\": false,\n"
                 << "  \"hint_source\": \"none\",\n"
                 << "  \"hint_scope\": \"none\",\n"
                 << "  \"hint_complete_for_basic_model\": false,\n"
                 << "  \"hint_complete_for_full_model\": false,\n"
                 << "  \"hinted_layer_variable_count\": 0,\n"
                 << "  \"hinted_operation_variable_count\": 0,\n"
                 << "  \"hinted_scalar_variable_count\": 0,\n"
                 << "  \"hinted_auxiliary_variable_count\": 0,\n"
                 << "  \"hinted_total_variable_count\": 0,\n"
                 << "  \"fallback_available\": false,\n"
                 << "  \"fallback_used\": false,\n"
                 << "  \"fallback_source\": \"none\",\n"
                 << "  \"solution_source\": \"none\"\n"
                 << "}\n";
      try {
        LogLifecycle(cli_started, instance, "OUTPUT_WRITE_START", algorithm,
                     joint_options.time_limit_seconds,
                     joint_options.num_workers, "ERROR", 0.0,
                     output_prefix + ".diagnostic.json");
        slackpipe::WriteTextFile(output_prefix + ".diagnostic.json",
                                 diagnostic.str());
        LogLifecycle(cli_started, instance, "OUTPUT_WRITE_END", algorithm,
                     joint_options.time_limit_seconds,
                     joint_options.num_workers, "ERROR", 0.0,
                     output_prefix + ".diagnostic.json");
      } catch (const std::exception &write_error) {
        std::cerr << "failed to write diagnostic JSON: " << write_error.what()
                  << "\n";
      }
    }
    return 1;
  }
}
