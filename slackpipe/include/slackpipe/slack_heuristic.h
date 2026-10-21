#pragma once

#include <functional>
#include <optional>
#include <string>

#include "slackpipe/search_result.h"
#include "slackpipe/deadline.h"

namespace slackpipe {

struct SlackHeuristicOptions {
  double time_limit_seconds = 60;
  Index evaluation_limit = 100000000;
  int random_seed = 1;
  std::string refinement_mode = "coupled";
  bool fixed_evaluation_test = false;
  bool verify_cached_replay = false;
  bool record_trace = false;
  const Deadline* parent_deadline = nullptr;
  std::function<bool(const std::vector<Tick>&)> allowed_partition;
};

[[nodiscard]] SearchResult SolveSlackHeuristic(
    const Instance& instance, const SlackHeuristicOptions& options = {});

}  // namespace slackpipe
