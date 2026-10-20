#pragma once

#include <functional>
#include <optional>
#include <string>

#include "slackpipe/bfs_solver.h"
#include "slackpipe/deadline.h"
#include "slackpipe/partition_restriction.h"

namespace slackpipe {

struct HeuristicHintOptions {
  double time_limit_seconds = 60;
  Index evaluation_limit = 100000000;
  int random_seed = 1;
  std::string refinement_mode = "coupled";
  bool fixed_evaluation_test = false;
  bool verify_cached_replay = false;
  bool record_trace = false;
  const Deadline* parent_deadline = nullptr;
  std::optional<PartitionRestriction> partition_restriction;
  std::function<bool(const std::vector<Tick>&)> allowed_partition;
};

[[nodiscard]] BfsSplitOptimizationResult BuildHeuristicIncumbent(
    const Instance& instance, const HeuristicHintOptions& options = {});

}  // namespace slackpipe
