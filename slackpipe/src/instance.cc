#include "slackpipe/instance.h"

#include <numeric>

namespace slackpipe {

namespace {

Index StageRoleIndex(Index stage, Index stages) {
  if (stage == 0) return 0;
  if (stage == stages - 1) return 2;
  return 1;
}

}  // namespace

Tick CheckedAdd(Tick a, Tick b, const char* context) {
  if ((b > 0 && a > kTickMax - b) ||
      (b < 0 && a < std::numeric_limits<Tick>::min() - b)) {
    throw Error(std::string("int64 addition overflow in ") + context);
  }
  return a + b;
}

Tick CheckedMul(Tick a, Tick b, const char* context) {
  if (a < 0 || b < 0) {
    throw Error(std::string("negative multiplication operand in ") + context);
  }
  if (a != 0 && b > kTickMax / a) {
    throw Error(std::string("int64 multiplication overflow in ") + context);
  }
  return a * b;
}

Tick CheckedDivExact(Tick numerator, Tick denominator, const char* context) {
  if (denominator <= 0) {
    throw Error(std::string("non-positive denominator in ") + context);
  }
  if (numerator % denominator != 0) {
    throw Error(std::string("non-integral tick duration in ") + context);
  }
  return numerator / denominator;
}

Index Instance::WorkerForStage(Index stage) const {
  if (stage < 0 || stage >= stages || workers <= 0) {
    throw Error("invalid stage or worker count in placement lookup");
  }
  return stage_to_worker.empty() ? stage % workers : stage_to_worker.at(stage);
}

void Instance::Validate() const {
  if (microbatches <= 0) throw Error("microbatches must be positive");
  if (stages <= 0) throw Error("stages must be positive");
  if (workers <= 0) throw Error("workers must be positive");
  if (!stage_to_worker.empty()) {
    if (stage_to_worker.size() != static_cast<std::size_t>(stages)) {
      throw Error("stage_to_worker must have N entries");
    }
    for (Index worker : stage_to_worker) {
      if (worker < 0 || worker >= workers) throw Error("invalid placement worker");
    }
  }
  if (total_layers <= 0) throw Error("total_layers must be positive");
  if (min_layers <= 0) throw Error("min_layers must be positive");
  if (backward_ratio_num <= 0 || backward_ratio_den <= 0) {
    throw Error("backward ratio must be positive");
  }
  if (communication_ticks < 0) {
    throw Error("communication_ticks must be non-negative");
  }
  if (HasAffineCostProfile()) {
    const auto expected = static_cast<std::size_t>(stages);
    if (profile_forward_slope_ticks.size() != expected ||
        profile_backward_slope_ticks.size() != expected ||
        profile_forward_bias_ticks.size() != expected ||
        profile_backward_bias_ticks.size() != expected) {
      throw Error("cost profile arrays must have one entry per stage");
    }
    for (Index stage = 0; stage < stages; ++stage) {
      const auto index = static_cast<std::size_t>(stage);
      if (profile_forward_slope_ticks[index] <= 0 ||
          profile_backward_slope_ticks[index] <= 0) {
        throw Error("cost profile slopes must be positive ticks");
      }
      if (profile_forward_bias_ticks[index] < 0 ||
          profile_backward_bias_ticks[index] < 0) {
        throw Error("cost profile biases must be non-negative ticks");
      }
    }
  }
  if (HasRangeCostProfile()) {
    const auto expected = static_cast<std::size_t>(total_layers + 1);
    if (profile_prefix_forward_ticks.size() != expected ||
        profile_prefix_backward_ticks.size() != expected) {
      throw Error(
          "range cost profile prefix arrays must have total_layers + 1 "
          "entries");
    }
    if (profile_role_forward_bias_ticks.size() != 3 ||
        profile_role_backward_bias_ticks.size() != 3) {
      throw Error(
          "range cost profile role biases must have first/middle/last entries");
    }
    if (profile_prefix_forward_ticks.front() != 0 ||
        profile_prefix_backward_ticks.front() != 0) {
      throw Error("range cost profile prefix arrays must start at zero");
    }
    for (std::size_t i = 1; i < expected; ++i) {
      if (profile_prefix_forward_ticks[i] <
              profile_prefix_forward_ticks[i - 1] ||
          profile_prefix_backward_ticks[i] <
              profile_prefix_backward_ticks[i - 1]) {
        throw Error("range cost profile prefix arrays must be non-decreasing");
      }
    }
    for (Tick value : profile_role_forward_bias_ticks) {
      if (value < 0)
        throw Error("range cost profile forward biases must be non-negative");
    }
    for (Tick value : profile_role_backward_bias_ticks) {
      if (value < 0)
        throw Error("range cost profile backward biases must be non-negative");
    }
  }
  if (HasAffineCostProfile() && HasRangeCostProfile()) {
    throw Error(
        "cost profile cannot contain both affine and range cost models");
  }
  (void)CheckedMul(microbatches, CheckedMul(2, stages, "operation count"),
                   "operation count");
  if (CheckedMul(stages, min_layers, "minimum layers") > total_layers) {
    throw Error("minimum layer requirement exceeds total layers");
  }
}

bool Instance::HasCostProfile() const {
  return HasAffineCostProfile() || HasRangeCostProfile();
}

bool Instance::HasAffineCostProfile() const {
  return !profile_forward_slope_ticks.empty() ||
         !profile_backward_slope_ticks.empty() ||
         !profile_forward_bias_ticks.empty() ||
         !profile_backward_bias_ticks.empty();
}

bool Instance::HasRangeCostProfile() const {
  return !profile_prefix_forward_ticks.empty() ||
         !profile_prefix_backward_ticks.empty() ||
         !profile_role_forward_bias_ticks.empty() ||
         !profile_role_backward_bias_ticks.empty();
}

Index Instance::OperationCount() const {
  Validate();
  return CheckedMul(microbatches, CheckedMul(2, stages, "operation count"),
                    "operation count");
}

Index Instance::StageBeginLayer(Index stage,
                                const std::vector<Tick>& split) const {
  if (split.size() != static_cast<std::size_t>(stages)) {
    throw Error("split length does not match stage count");
  }
  if (stage < 0 || stage >= stages) throw Error("stage out of range");
  Tick begin = 0;
  for (Index s = 0; s < stage; ++s) {
    begin = CheckedAdd(begin, split[static_cast<std::size_t>(s)],
                       "stage begin layer");
  }
  return begin;
}

Index Instance::StageEndLayer(Index stage,
                              const std::vector<Tick>& split) const {
  return CheckedAdd(StageBeginLayer(stage, split),
                    split[static_cast<std::size_t>(stage)], "stage end layer");
}

Tick Instance::ForwardDuration(Index stage, Index begin_layer,
                               Index end_layer) const {
  Validate();
  if (stage < 0 || stage >= stages) throw Error("stage out of range");
  if (begin_layer < 0 || end_layer <= begin_layer || end_layer > total_layers) {
    throw Error("layer range out of range");
  }
  if (HasRangeCostProfile()) {
    const auto begin = static_cast<std::size_t>(begin_layer);
    const auto end = static_cast<std::size_t>(end_layer);
    const Tick layer_cost =
        profile_prefix_forward_ticks[end] - profile_prefix_forward_ticks[begin];
    const Tick bias = profile_role_forward_bias_ticks[static_cast<std::size_t>(
        StageRoleIndex(stage, stages))];
    return CheckedAdd(layer_cost, bias, "range profile forward duration");
  }
  const Tick layers = end_layer - begin_layer;
  if (HasAffineCostProfile()) {
    const auto index = static_cast<std::size_t>(stage);
    return CheckedAdd(CheckedMul(layers, profile_forward_slope_ticks[index],
                                 "profile forward duration"),
                      profile_forward_bias_ticks[index],
                      "profile forward duration");
  }
  return CheckedMul(layers, backward_ratio_den, "forward duration");
}

Tick Instance::BackwardDuration(Index stage, Index begin_layer,
                                Index end_layer) const {
  Validate();
  if (stage < 0 || stage >= stages) throw Error("stage out of range");
  if (begin_layer < 0 || end_layer <= begin_layer || end_layer > total_layers) {
    throw Error("layer range out of range");
  }
  if (HasRangeCostProfile()) {
    const auto begin = static_cast<std::size_t>(begin_layer);
    const auto end = static_cast<std::size_t>(end_layer);
    const Tick layer_cost = profile_prefix_backward_ticks[end] -
                            profile_prefix_backward_ticks[begin];
    const Tick bias = profile_role_backward_bias_ticks[static_cast<std::size_t>(
        StageRoleIndex(stage, stages))];
    return CheckedAdd(layer_cost, bias, "range profile backward duration");
  }
  const Tick layers = end_layer - begin_layer;
  if (HasAffineCostProfile()) {
    const auto index = static_cast<std::size_t>(stage);
    return CheckedAdd(CheckedMul(layers, profile_backward_slope_ticks[index],
                                 "profile backward duration"),
                      profile_backward_bias_ticks[index],
                      "profile backward duration");
  }
  return CheckedMul(layers, backward_ratio_num, "backward duration");
}

Tick Instance::Duration(Index stage, bool backward,
                        const std::vector<Tick>& split) const {
  Validate();
  if (split.size() != static_cast<std::size_t>(stages)) {
    throw Error("split length does not match stage count");
  }
  if (stage < 0 || stage >= stages) throw Error("stage out of range");
  const Index begin = StageBeginLayer(stage, split);
  const Index end = StageEndLayer(stage, split);
  return backward ? BackwardDuration(stage, begin, end)
                  : ForwardDuration(stage, begin, end);
}

Tick Instance::EdgeDelay(Index from_worker, Index to_worker) const {
  if (from_worker < 0 || from_worker >= workers || to_worker < 0 ||
      to_worker >= workers) {
    throw Error("worker out of range");
  }
  return from_worker == to_worker ? 0 : communication_ticks;
}

void ValidateSplit(const Instance& instance, const std::vector<Tick>& split) {
  instance.Validate();
  if (split.size() != static_cast<std::size_t>(instance.stages)) {
    throw Error("split length does not match stage count");
  }
  Tick sum = 0;
  for (Tick value : split) {
    if (value < instance.min_layers) {
      throw Error("split violates minimum layer requirement");
    }
    sum = CheckedAdd(sum, value, "split sum");
  }
  if (sum != instance.total_layers) {
    throw Error("split sum does not equal total layers");
  }
  for (Index s = 0; s < instance.stages; ++s) {
    (void)instance.Duration(s, false, split);
    (void)instance.Duration(s, true, split);
  }
}

std::vector<Tick> UniformSplit(const Instance& instance) {
  instance.Validate();
  const Tick minimum_total =
      CheckedMul(instance.stages, instance.min_layers, "uniform split minimum");
  if (instance.total_layers < minimum_total) {
    throw Error("uniform split requires L >= N * min_layers");
  }
  const Tick remaining = instance.total_layers - minimum_total;
  const Tick q = remaining / instance.stages;
  const Tick remainder = remaining % instance.stages;
  std::vector<Tick> split(static_cast<std::size_t>(instance.stages),
                          instance.min_layers + q);
  for (Index s = 0; s < remainder; ++s) {
    ++split[static_cast<std::size_t>(s)];
  }
  ValidateSplit(instance, split);
  return split;
}

std::vector<Tick> WorkerLayerTotals(const Instance& instance,
                                    const std::vector<Tick>& split) {
  instance.Validate();
  ValidateSplit(instance, split);
  std::vector<Tick> totals(static_cast<std::size_t>(instance.workers), 0);
  for (Index s = 0; s < instance.stages; ++s) {
    totals[static_cast<std::size_t>(instance.WorkerForStage(s))] +=
        split[static_cast<std::size_t>(s)];
  }
  return totals;
}

std::uint64_t CountValidSplitsCapped(const Instance &instance,
                                     std::uint64_t cap) {
  instance.Validate();
  if (cap == 0) return 0;

  const Tick slack =
      instance.total_layers -
      CheckedMul(instance.stages, instance.min_layers, "split count slack");
  if (slack < 0) return 0;
  const std::uint64_t n = static_cast<std::uint64_t>(
      CheckedAdd(slack, instance.stages - 1, "split count binomial n"));
  std::uint64_t k = static_cast<std::uint64_t>(instance.stages - 1);
  if (k > n - k) k = n - k;

  __uint128_t value = 1;
  for (std::uint64_t i = 1; i <= k; ++i) {
    value *= (n - k + i);
    value /= i;
    if (value >= cap) return cap;
  }
  return static_cast<std::uint64_t>(value);
}

}  // namespace slackpipe
