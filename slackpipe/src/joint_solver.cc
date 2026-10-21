#include "slackpipe/joint_solver.h"

namespace slackpipe {
#if !SLACKPIPE_HAVE_ORTOOLS
SearchResult OptimizeJointSplitAndScheduleCpSat(
    const Instance& instance, const JointOptimizerOptions& options) {
  instance.Validate();
  SearchResult result;
  result.method = "cpsat-no-hint";
  result.status = result.solver_status_raw = "UNAVAILABLE";
  result.no_solution_reason = "OR-Tools support is not compiled in";
  result.requested_seconds = options.time_limit_seconds;
  result.num_workers = options.num_workers;
  return result;
}
bool IsJointOptimizerAvailable() { return false; }
#endif
}  // namespace slackpipe
