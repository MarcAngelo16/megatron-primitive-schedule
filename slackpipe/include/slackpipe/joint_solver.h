#pragma once

#include "slackpipe/deadline.h"
#include "slackpipe/lifecycle.h"
#include "slackpipe/search_result.h"

namespace slackpipe {

struct JointOptimizerOptions {
  double time_limit_seconds = 0;
  int num_workers = 1;
  int random_seed = 1;
  bool log_search_progress = false;
  ActivationAnalysisOptions activation_options;
  LifecycleCallback lifecycle;
};

// Unrestricted contiguous partition + worker-local order, fixed cyclic
// placement, mandatory FIFO. No external incumbent, hints or fallback API.
[[nodiscard]] SearchResult OptimizeJointSplitAndScheduleCpSat(
    const Instance& instance, const JointOptimizerOptions& options = {});
[[nodiscard]] bool IsJointOptimizerAvailable();

}  // namespace slackpipe
