#pragma once
#include <string>
#include <vector>

namespace slackpipe {
inline constexpr int kEvaluationMethodVersion = 2;
// Internal diagnostic orders, deliberately NOT public evaluation methods.
inline constexpr const char* kUniformBreadthFirstMethod = "uniform-breadth-first";
inline constexpr const char* kUniformInterleavedOneFOneBMethod = "uniform-interleaved-1f1b";
inline constexpr const char* kFixedScheduleRuleBreadthFirst = "breadth_first_wavefront_microbatch_plus_position";
inline constexpr const char* kUniformPartitionRule = "as_even_as_possible_remainder_to_lowest_stage";

struct EvaluationMethodDefinition {
  std::string canonical_name;
  std::string display_name;
  std::string executor;
  bool requires_ortools;
  bool supports_solver_threads;
  bool supports_activation_cap;
  std::string semantics;
};
const std::vector<EvaluationMethodDefinition>& EvaluationMethodRegistry();
const EvaluationMethodDefinition* FindEvaluationMethod(const std::string& name);
std::string CanonicalizeEvaluationMethodName(const std::string& name);
std::string EvaluationMethodContractHash(const EvaluationMethodDefinition& definition);
std::string DescribeEvaluationMethodJson(const std::string& name);
}  // namespace slackpipe
