#pragma once
#include <optional>
#include <string>
#include "slackpipe/activation_analyzer.h"
namespace slackpipe {
inline constexpr int kEvaluationResultSchemaVersion = 2;
inline constexpr int kBuildInfoSchemaVersion = 2;
[[nodiscard]] std::string CurrentTimestampUtc();
[[nodiscard]] std::optional<std::string> CurrentGitCommit();
[[nodiscard]] std::optional<bool> CurrentGitDirty(
    std::optional<std::string> *scope = nullptr);
[[nodiscard]] std::optional<std::string> CurrentBuildType();
[[nodiscard]] bool OrToolsCompiledIn();
[[nodiscard]] std::optional<std::string> OrToolsVersion();
[[nodiscard]] bool CumulativeConstraintCompiledIn();
[[nodiscard]] bool VariableCumulativeDemandCompiledIn();
[[nodiscard]] std::string ActivationCapSolverSupportLevelForBuild();
[[nodiscard]] std::string CommandLineFromArgv(int argc, char **argv);
[[nodiscard]] std::string ExecutableNameFromArgv0(const std::string &argv0);
[[nodiscard]] std::string MakeRunId(const std::string &seed);
[[nodiscard]] bool IsTerminalFeasibleStatus(const std::string &status);
[[nodiscard]] std::optional<double> RelativeGap(Tick makespan,
                                                double best_bound);

std::string ActivationAnalysisJson(const ActivationAnalysisResult& analysis);
}
