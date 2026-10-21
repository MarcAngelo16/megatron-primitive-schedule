#pragma once

#include <array>
#include <memory>
#include <optional>

#include "slackpipe/slack_heuristic.h"

// Internal primitives are exposed here for direct, non-stochastic regression
// tests. They are not a second solver or a public plan interchange format.
namespace slackpipe::heuristic_internal {
using Split = std::vector<Tick>;
struct Edge { Index target; Tick lag; };
struct Graph {
  MachineOrders orders;
  std::vector<std::array<Edge, 3>> edges;
  std::vector<int> degree;
  std::vector<Index> topo;
};
struct Pair {
  Split partition;
  std::shared_ptr<const Graph> graph;
  std::vector<Tick> start, end, duration;
  Tick makespan = 0, backward_sum = 0;
};
struct Float {
  std::vector<Tick> slack, critical;
};
struct OrderAction {
  Index worker = -1, source = -1, anchor = -1;
  bool stream = false;
  long double score = 0;
};

std::vector<Tick> PhaseCosts(const Instance&, const Split&);
std::shared_ptr<const Graph> Compile(const Instance&, const MachineOrders&);
std::optional<Pair> Replay(const Instance&, const Split&,
                           std::shared_ptr<const Graph>,
                           const std::vector<Tick>&);
Float TotalFloat(const Pair&);
bool Better(const Pair&, const Pair&);
std::string ImprovementReason(const Pair&, const Pair&);
bool ApplyOrder(const Instance&, const OrderAction&, MachineOrders&);
std::vector<OrderAction> OrderActions(const Instance&, const Pair&, const Float&, Index round,
                                     const std::function<bool()>& stopped = {});
// The returned stale count changes only on strict makespan improvement.
Index AcceptChild(const Pair& child, Pair& parent, Index stale);
}  // namespace slackpipe::heuristic_internal
