#pragma once

#include <vector>
#include <string>

#include "slackpipe/types.h"

namespace slackpipe {

struct Instance {
  Index microbatches = 0;
  Index stages = 0;
  Index workers = 0;
  Index total_layers = 0;
  Index min_layers = 1;
  Tick backward_ratio_num = 1;
  Tick backward_ratio_den = 1;
  Tick communication_ticks = 0;
  std::vector<Tick> profile_forward_slope_ticks;
  std::vector<Tick> profile_backward_slope_ticks;
  std::vector<Tick> profile_forward_bias_ticks;
  std::vector<Tick> profile_backward_bias_ticks;
  std::vector<Tick> profile_prefix_forward_ticks;
  std::vector<Tick> profile_prefix_backward_ticks;
  std::vector<Tick> profile_role_forward_bias_ticks;
  std::vector<Tick> profile_role_backward_bias_ticks;
  std::string cost_profile_path;
  std::string cost_profile_schema_version;
  std::string cost_profile_hash;
  std::string model_manifest_hash;
  std::string cost_profile_units = "microseconds";
  // Empty preserves the existing cyclic placement. Explicit maps are used by
  // the fixed-stage OctoPipe tuner and the common evaluator, not CP-SAT search.
  std::vector<Index> stage_to_worker;

  void Validate() const;
  [[nodiscard]] Index WorkerForStage(Index stage) const;
  [[nodiscard]] bool HasCostProfile() const;
  [[nodiscard]] bool HasAffineCostProfile() const;
  [[nodiscard]] bool HasRangeCostProfile() const;
  [[nodiscard]] Index OperationCount() const;
  [[nodiscard]] Index StageBeginLayer(Index stage,
                                      const std::vector<Tick>& split) const;
  [[nodiscard]] Index StageEndLayer(Index stage,
                                    const std::vector<Tick>& split) const;
  [[nodiscard]] Tick ForwardDuration(Index stage, Index begin_layer,
                                     Index end_layer) const;
  [[nodiscard]] Tick BackwardDuration(Index stage, Index begin_layer,
                                      Index end_layer) const;
  [[nodiscard]] Tick Duration(Index stage, bool backward,
                              const std::vector<Tick>& split) const;
  [[nodiscard]] Tick EdgeDelay(Index from_worker, Index to_worker) const;
};

void ValidateSplit(const Instance& instance, const std::vector<Tick>& split);
[[nodiscard]] std::vector<Tick> UniformSplit(const Instance& instance);
[[nodiscard]] std::vector<Tick> WorkerLayerTotals(
    const Instance& instance, const std::vector<Tick>& split);
[[nodiscard]] std::uint64_t CountValidSplitsCapped(
    const Instance& instance, std::uint64_t cap);

}  // namespace slackpipe
