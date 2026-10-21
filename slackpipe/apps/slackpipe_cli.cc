#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "slackpipe/breadth_first.h"
#include "slackpipe/cost_profile.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/octopipe_tuner.h"
#include "slackpipe/plan_export.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/slack_heuristic.h"

namespace {
using namespace slackpipe;
std::string Read(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw Error("cannot read " + path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::vector<Tick> Ticks(const std::string& value) {
  std::vector<Tick> result;
  std::istringstream in(value);
  std::string part;
  while (std::getline(in, part, ',')) result.push_back(std::stoll(part));
  return result;
}
void Help() {
  std::cout <<
      "slackpipe_cli --algorithm {slack-heuristic,octopipe-tuning,cpsat-no-hint}\n"
      "  --B B --N N --J workers --L layers --ratio-num 2 --ratio-den 1\n"
      "  --communication ticks --cost-profile PATH --min-layers 1\n"
      "  --time-limit 300 --num-workers 1 --seed 1 --output-prefix PATH --emit-plan PATH\n"
      "  --heuristic-evaluations COUNT --heuristic-fixed-evaluations true --heuristic-trace true\n"
      "  --octopipe-max-iterations COUNT --octopipe-candidates 64\n"
      "  --enforce-activation-cap true --activation-cap-units scalar[,worker...] (CP-SAT only)\n"
      "  --activation-model {count,linear-in-stage-layers,explicit-stage-units}\n"
      "  --activation-stage-units LIST --activation-units-per-layer 1 --activation-trace true\n"
      "  --build-info | --describe-method METHOD | --validate-result PATH\n"
      "Native 1f1b/interleaved-1f1b run through tools/run_slackpipe_eval.py, not this CLI.\n";
}
}

int main(int argc, char** argv) {
  using namespace slackpipe;
  try {
    std::map<std::string, std::string> args;
    const std::vector<std::string> allowed{
      "--algorithm", "--B", "--N", "--J", "--L", "--min-layers", "--ratio-num", "--ratio-den",
      "--communication", "--cost-profile", "--time-limit", "--num-workers", "--seed",
      "--output-prefix", "--emit-plan", "--heuristic-evaluations", "--heuristic-fixed-evaluations",
      "--heuristic-trace", "--octopipe-max-iterations", "--octopipe-candidates",
      "--enforce-activation-cap", "--activation-cap-units", "--activation-model",
      "--activation-stage-units", "--activation-units-per-layer", "--activation-trace",
      "--describe-method", "--validate-result", "--validation-output"};
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--help" || key == "-h") { Help(); return 0; }
      if (key == "--build-info") {
        std::cout << "{\"schema_version\":2,\"ortools_compiled_in\":"
          << (OrToolsCompiledIn() ? "true" : "false") << ",\"ortools_enabled\":"
          << (OrToolsCompiledIn() ? "true" : "false") << ",\"ortools_version\":\""
          << OrToolsVersion().value_or("") << "\",\"git_commit\":\""
          << CurrentGitCommit().value_or("") << "\",\"evaluation_method_version\":2}\n";
        return 0;
      }
      if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
        throw Error("unknown/retired option " + key + "; see --help");
      if (++i == argc || args.contains(key)) throw Error("missing or duplicate " + key);
      args[key] = argv[i];
    }
    auto get = [&](const std::string& key, const std::string& fallback) {
      const auto it = args.find(key);
      return it == args.end() ? fallback : it->second;
    };
    auto flag = [&](const std::string& key) {
      auto value = get(key, "false");
      if (value != "true" && value != "false") throw Error(key + " expects true/false");
      return value == "true";
    };
    if (args.contains("--describe-method")) {
      std::cout << DescribeEvaluationMethodJson(args.at("--describe-method")); return 0;
    }
    if (args.contains("--validate-result")) {
      auto validation = ValidateResultJsonText(Read(args.at("--validate-result")));
      const auto json = ResultValidationToJson(validation, "");
      std::cout << json << "\n";
      if (args.contains("--validation-output")) WriteTextFile(args.at("--validation-output"), json);
      return validation.passed ? 0 : 2;
    }
    const std::string method = get("--algorithm", "slack-heuristic");
    const auto* definition = FindEvaluationMethod(method);
    if (!definition) throw Error("unknown/retired method " + method);
    if (definition->executor != "slackpipe")
      throw Error("native methods must run through tools/run_slackpipe_eval.py");
    const int threads = std::stoi(get("--num-workers", "1"));
    if (threads < 1 || (!definition->supports_solver_threads && threads != 1))
      throw Error(method + " requires one search thread");
    const double budget = std::stod(get("--time-limit", "300"));
    if (!std::isfinite(budget) || budget < 0) throw Error("invalid time budget");
    const bool fixed_evaluations = flag("--heuristic-fixed-evaluations");
    const Index iterations = std::stoll(get("--octopipe-max-iterations", "0"));
    if (budget == 0 && !(method == "octopipe-tuning" && iterations > 0))
      throw Error("a positive wall-clock budget is required (OctoPipe iteration-only diagnostics excepted)");
    for (const auto& [key, value] : args) {
      (void)value;
      if (key.starts_with("--heuristic-") && method != "slack-heuristic")
        throw Error("heuristic option supplied to " + method);
      if (key.starts_with("--octopipe-") && method != "octopipe-tuning")
        throw Error("OctoPipe option supplied to " + method);
    }
    Instance instance;
    instance.microbatches = std::stoll(get("--B", "4"));
    instance.stages = std::stoll(get("--N", "4"));
    instance.workers = std::stoll(get("--J", "2"));
    instance.total_layers = std::stoll(get("--L", "8"));
    instance.min_layers = std::stoll(get("--min-layers", "1"));
    instance.backward_ratio_num = std::stoll(get("--ratio-num", "2"));
    instance.backward_ratio_den = std::stoll(get("--ratio-den", "1"));
    instance.communication_ticks = std::stoll(get("--communication", "0"));
    if (args.contains("--cost-profile")) ApplyCostProfileFile(instance, args.at("--cost-profile"));
    instance.Validate();
    ActivationAnalysisOptions activation;
    activation.enforce_activation_cap = flag("--enforce-activation-cap");
    activation.emit_activation_trace = flag("--activation-trace");
    activation.model = ParseActivationModel(get("--activation-model", "linear-in-stage-layers"));
    activation.activation_units_per_layer = std::stoll(get("--activation-units-per-layer", "1"));
    if (args.contains("--activation-stage-units"))
      activation.explicit_stage_activation_units = Ticks(args.at("--activation-stage-units"));
    if (args.contains("--activation-cap-units")) {
      activation.cap_mode = ActivationCapMode::kExplicit;
      activation.activation_cap_units = Ticks(args.at("--activation-cap-units"));
    }
    if (activation.enforce_activation_cap && !definition->supports_activation_cap)
      throw Error(method + " does not support solver-enforced activation caps");
    ValidateActivationOptions(instance, activation);
    // Never leave an old plan at a requested output path after a failed run.
    if (args.contains("--emit-plan") && std::filesystem::exists(args.at("--emit-plan")))
      throw Error("refusing to overwrite existing plan; use a fresh output path");
    SearchResult result;
    if (method == "cpsat-no-hint") {
      JointOptimizerOptions options;
      options.time_limit_seconds = budget;
      options.num_workers = threads;
      options.random_seed = std::stoi(get("--seed", "1"));
      options.activation_options = activation;
      result = OptimizeJointSplitAndScheduleCpSat(instance, options);
    } else if (method == "slack-heuristic") {
      SlackHeuristicOptions options;
      options.time_limit_seconds = budget;
      options.evaluation_limit = std::stoll(get("--heuristic-evaluations", "100000000"));
      options.fixed_evaluation_test = fixed_evaluations;
      options.random_seed = std::stoi(get("--seed", "1"));
      options.record_trace = flag("--heuristic-trace");
      result = SolveSlackHeuristic(instance, options);
    } else {
      OctoPipeOptions options;
      options.time_limit_seconds = budget;
      options.max_iterations = iterations;
      options.candidates_per_iteration = std::stoll(get("--octopipe-candidates", "64"));
      options.tune_placement = false;
      std::ostringstream trace;
      trace << "[";
      bool comma = false;
      options.progress = [&](const OctoPipeIteration& it, const OctoPipeState&) {
        if (comma) trace << ",";
        comma = true;
        trace << "{\"iteration\":" << it.iteration << ",\"phase\":\"" << ToString(it.phase)
              << "\",\"input_makespan\":" << it.input_makespan
              << ",\"best_makespan\":" << it.best_makespan
              << ",\"candidates\":" << it.candidates << ",\"valid_candidates\":" << it.valid_candidates
              << ",\"accepted\":" << (it.accepted ? "true" : "false") << "}";
      };
      auto octo = TuneOctoPipeAlgorithm1(instance, UniformSplit(instance), BreadthFirstOrders(instance), options);
      trace << "]";
      result.schedule = std::move(octo.best.schedule);
      result.split = result.schedule.split;
      result.machine_orders = result.schedule.orders;
      result.makespan_ticks = result.schedule.makespan;
      result.status = "FEASIBLE";
      result.final_solution_available = true;
      result.solution_source = "octopipe-tuning";
      result.wall_time_seconds = octo.tuning_seconds;
      result.iterations = octo.iterations;
      result.baseline_primary_objective = octo.initial_makespan;
      result.method_diagnostics_json = trace.str();
    }
    result.method = method;
    result.random_seed = std::stoi(get("--seed", "1"));
    result.requested_command = CommandLineFromArgv(argc, argv);
    result.requested_seconds = budget;
    result.num_workers = threads;
    if (result.final_solution_available) {
      result.activation_analysis = AnalyzeActivationMemory(instance, result.schedule, activation);
      ApplyActivationCapConstraintMetadata(*result.activation_analysis, result.activation_cap_constraints);
    }
    const std::string json = ToJson(instance, result);
    const std::string prefix = get("--output-prefix", "");
    if (!prefix.empty()) {
      WriteTextFile(prefix + ".json", json);
      if (result.final_solution_available) {
        WriteTextFile(prefix + ".csv", ToCsv(instance, result.schedule));
        WriteTextFile(prefix + ".orders.txt", ToOrdersText(instance, result.schedule.orders));
        WriteTextFile(prefix + ".svg", ToSvg(instance, result.schedule));
      }
    }
    if (result.final_solution_available && args.contains("--emit-plan"))
      WriteMegatronSlackPipePlanFile(args.at("--emit-plan"), instance, result.schedule, result.status);
    std::cout << json;
    // A completed no-solution search is a recorded outcome, not an invented plan.
    return result.status == "UNAVAILABLE" ? 3 : 0;
  } catch (const std::exception& error) {
    std::cerr << "slackpipe_cli: " << error.what() << "\n"; return 2;
  }
}
