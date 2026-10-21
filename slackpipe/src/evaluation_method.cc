#include "slackpipe/evaluation_method.h"
#include <cstdint>
#include <iomanip>
#include <sstream>
#include "slackpipe/types.h"

namespace slackpipe {
const std::vector<EvaluationMethodDefinition>& EvaluationMethodRegistry() {
  static const std::vector<EvaluationMethodDefinition> methods{
    {"1f1b", "1F1B", "megatron-native", false, false, false, "native noninterleaved; N=W"},
    {"interleaved-1f1b", "Interleaved 1F1B", "megatron-native", false, false, false, "native interleaved; explicit N"},
    {"octopipe-tuning", "OctoPipe-Tuning", "slackpipe", false, false, false,
     "restricted fixed-stage complete F/B Algorithm 1; fixed cyclic placement; unchanged phase selection"},
    {"cpsat-no-hint", "CP-SAT without hints", "slackpipe", true, true, true,
     "joint unrestricted partition/order; conservative horizon; no external hints, incumbent or fallback; FIFO; NoOverlap"},
    {"slack-heuristic", "Slack-based Heuristic", "slackpipe", false, false, false,
     "standalone total-float heuristic; structured initialization; coupled refinement; valid lower-bound certificate"}
  };
  return methods;
}
const EvaluationMethodDefinition* FindEvaluationMethod(const std::string& name) {
  for (const auto& method : EvaluationMethodRegistry())
    if (method.canonical_name == name) return &method;
  return nullptr;
}
std::string CanonicalizeEvaluationMethodName(const std::string& name) { return name; }
std::string EvaluationMethodContractHash(const EvaluationMethodDefinition& method) {
  const auto contract = "v2|" + method.canonical_name + "|" + method.executor + "|" + method.semantics;
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : contract) { hash ^= ch; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
std::string DescribeEvaluationMethodJson(const std::string& name) {
  const auto* method = FindEvaluationMethod(name);
  if (!method) throw Error("unknown method (retired names are not aliases): " + name);
  std::ostringstream out;
  out << "{\"evaluation_method_version\":2,\"method_id\":\"" << name
      << "\",\"display_name\":\"" << method->display_name
      << "\",\"executor\":\"" << method->executor
      << "\",\"method_contract_hash\":\"" << EvaluationMethodContractHash(*method)
      << "\",\"requires_ortools\":" << (method->requires_ortools ? "true" : "false")
      << ",\"supports_solver_threads\":" << (method->supports_solver_threads ? "true" : "false")
      << ",\"supports_activation_cap\":" << (method->supports_activation_cap ? "true" : "false")
      << ",\"semantics\":\"" << method->semantics << "\"}\n";
  return out.str();
}
}  // namespace slackpipe
