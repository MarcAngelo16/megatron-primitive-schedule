#include "slackpipe/dag_evaluator.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <tuple>

namespace slackpipe {

namespace {
struct Edge {
  Index to = 0;
  Tick lag = 0;
};

void AddEdge(std::vector<std::vector<Edge>>& graph,
             std::vector<Index>& indegree, Index from, Index to, Tick lag) {
  graph[static_cast<std::size_t>(from)].push_back(Edge{to, lag});
  ++indegree[static_cast<std::size_t>(to)];
}

std::vector<std::string> ValidateOrders(const Instance& instance,
                                        const MachineOrders& orders,
                                        bool fifo_ordering) {
  std::vector<std::string> errors;
  if (orders.size() != static_cast<std::size_t>(instance.workers)) {
    errors.push_back(
        "MISSING_OP machine order count does not match worker count");
    return errors;
  }

  std::vector<Index> seen(static_cast<std::size_t>(instance.OperationCount()),
                          0);
  for (Index w = 0; w < instance.workers; ++w) {
    std::vector<Index> last_batch(static_cast<std::size_t>(2 * instance.stages), -1);
    for (OperationId id : orders[static_cast<std::size_t>(w)]) {
      try {
        const OperationView view = DecodeOperation(instance, id);
        if (fifo_ordering) {
          Index& previous = last_batch[static_cast<std::size_t>(view.chain_index)];
          if (previous >= view.microbatch) {
            errors.push_back("FIFO_ORDER_VIOLATION worker order reverses " +
                             OperationName(instance, id));
          }
          previous = view.microbatch;
        }
        ++seen[static_cast<std::size_t>(id.value)];
        if (seen[static_cast<std::size_t>(id.value)] > 1) {
          std::ostringstream msg;
          msg << "DUPLICATE_OP b=" << view.microbatch
              << " n=" << view.chain_index << " id=" << id.value;
          errors.push_back(msg.str());
          continue;
        }
        if (view.worker != w) {
          std::ostringstream msg;
          msg << "WRONG_WORKER b=" << view.microbatch
              << " n=" << view.chain_index << " expected_worker=" << view.worker
              << " actual_worker=" << w;
          errors.push_back(msg.str());
        }
      } catch (const Error& error) {
        std::ostringstream msg;
        msg << "MISSING_OP " << error.what();
        errors.push_back(msg.str());
      }
    }
  }

  for (Index id_value = 0; id_value < instance.OperationCount(); ++id_value) {
    if (seen[static_cast<std::size_t>(id_value)] == 0) {
      const OperationView view =
          DecodeOperation(instance, OperationId{id_value});
      std::ostringstream msg;
      msg << "MISSING_OP b=" << view.microbatch << " n=" << view.chain_index
          << " id=" << id_value;
      errors.push_back(msg.str());
    }
  }
  return errors;
}

std::vector<std::string> ValidateMachinePredecessors(
    const Instance& instance, const MachinePredecessors& predecessors,
    bool fifo_ordering) {
  std::vector<std::string> errors;
  for (const auto& [id_value, predecessor] : predecessors) {
    try {
      const OperationId id{id_value};
      const OperationView current = DecodeOperation(instance, id);
      const OperationView pred = DecodeOperation(instance, predecessor);
      if (pred.worker != current.worker) {
        std::ostringstream msg;
        msg << "WRONG_WORKER " << OperationName(instance, predecessor)
            << " cannot serialize " << OperationName(instance, id)
            << " because workers differ";
        errors.push_back(msg.str());
      }
      if (fifo_ordering && pred.chain_index == current.chain_index) {
        std::ostringstream msg;
        msg << "DUPLICATE_OP " << OperationName(instance, predecessor)
            << " cannot serialize " << OperationName(instance, id)
            << " because operation positions are identical";
        errors.push_back(msg.str());
      }
      if (IsDataPredecessor(instance, predecessor, id)) {
        std::ostringstream msg;
        msg << "DATA_DEP_VIOLATION " << OperationName(instance, predecessor)
            << " duplicates data predecessor for "
            << OperationName(instance, id);
        errors.push_back(msg.str());
      }
    } catch (const Error& error) {
      errors.push_back(error.what());
    }
  }
  return errors;
}

EvaluationResult EvaluateScheduleGraph(const Instance& instance,
                                       const std::vector<Tick>& split,
                                       const MachineOrders& orders,
                                       const MachinePredecessors& predecessors,
                                       std::vector<std::string> errors,
                                       bool fifo_ordering) {
  EvaluationResult result;
  result.schedule.split = split;
  result.schedule.orders = orders;
  result.schedule.validation_errors = std::move(errors);
  if (!result.schedule.validation_errors.empty()) return result;

  const Index count = instance.OperationCount();
  std::vector<std::vector<Edge>> graph(static_cast<std::size_t>(count));
  std::vector<Index> indegree(static_cast<std::size_t>(count), 0);
  std::vector<Tick> duration(static_cast<std::size_t>(count), 0);
  result.schedule.operations_by_id.resize(static_cast<std::size_t>(count));

  for (Index id_value = 0; id_value < count; ++id_value) {
    OperationId id{id_value};
    const OperationView view = DecodeOperation(instance, id);
    duration[static_cast<std::size_t>(id_value)] =
        instance.Duration(view.stage, view.backward, split);
    result.schedule.operations_by_id[static_cast<std::size_t>(id_value)] =
        ScheduledOperation{id, 0, 0,
                           duration[static_cast<std::size_t>(id_value)],
                           view.worker};
    if (view.chain_index + 1 < 2 * instance.stages) {
      OperationId next =
          EncodeOperation(instance, view.microbatch, view.chain_index + 1);
      const OperationView next_view = DecodeOperation(instance, next);
      AddEdge(graph, indegree, id.value, next.value,
              instance.EdgeDelay(view.worker, next_view.worker));
    }
    if (fifo_ordering) {
      const std::optional<OperationId> fifo = FifoPredecessor(instance, id);
      if (fifo) {
        AddEdge(graph, indegree, fifo->value, id.value, 0);
      }
    }
  }

  for (const auto& [id_value, predecessor] : predecessors) {
    AddEdge(graph, indegree, predecessor.value, id_value, 0);
  }

  std::queue<Index> ready;
  for (Index i = 0; i < count; ++i) {
    if (indegree[static_cast<std::size_t>(i)] == 0) ready.push(i);
  }

  std::vector<Tick> start(static_cast<std::size_t>(count), 0);
  Index visited = 0;
  while (!ready.empty()) {
    const Index current = ready.front();
    ready.pop();
    ++visited;
    const Tick end =
        CheckedAdd(start[static_cast<std::size_t>(current)],
                   duration[static_cast<std::size_t>(current)], "schedule end");
    result.schedule.operations_by_id[static_cast<std::size_t>(current)].start =
        start[static_cast<std::size_t>(current)];
    result.schedule.operations_by_id[static_cast<std::size_t>(current)].end =
        end;
    result.schedule.makespan = std::max(result.schedule.makespan, end);

    for (const Edge& edge : graph[static_cast<std::size_t>(current)]) {
      const Tick successor_start = CheckedAdd(end, edge.lag, "successor start");
      start[static_cast<std::size_t>(edge.to)] =
          std::max(start[static_cast<std::size_t>(edge.to)], successor_start);
      --indegree[static_cast<std::size_t>(edge.to)];
      if (indegree[static_cast<std::size_t>(edge.to)] == 0) ready.push(edge.to);
    }
  }

  if (visited != count) {
    result.schedule.validation_errors.push_back(
        fifo_ordering ? "CYCLE_OR_UNRESOLVED cycle detected among data, FIFO, "
                        "and worker-local predecessor edges"
                      : "CYCLE_OR_UNRESOLVED cycle detected among data and "
                        "worker-local predecessor edges");
    result.schedule.operations_by_id.clear();
    result.schedule.makespan = 0;
  }
  return result;
}
}  // namespace

Tick TotalUsefulWork(const Instance& instance) {
  instance.Validate();
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "total useful work ratio sum");
  return CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                               "total useful work"),
                    ratio_sum, "total useful work");
}

MachinePredecessors ExtractMachinePredecessors(const Instance& instance,
                                               const MachineOrders& orders,
                                               bool fifo_ordering) {
  instance.Validate();
  MachinePredecessors predecessors;
  for (const auto& worker_order : orders) {
    for (std::size_t i = 1; i < worker_order.size(); ++i) {
      const OperationView next = DecodeOperation(instance, worker_order[i]);
      const OperationView prev = DecodeOperation(instance, worker_order[i - 1]);
      // Only forward FIFO edges are redundant. Dropping a reversed edge
      // would silently turn an impossible worker order into a feasible DAG.
      if (fifo_ordering && prev.chain_index == next.chain_index &&
          prev.microbatch >= next.microbatch) {
        throw Error("FIFO_ORDER_VIOLATION same-position worker order reverses FIFO");
      }
      if (prev.worker == next.worker &&
          (!fifo_ordering || prev.chain_index != next.chain_index) &&
          !IsDataPredecessor(instance, worker_order[i - 1], worker_order[i])) {
        predecessors.emplace(worker_order[i].value, worker_order[i - 1]);
      }
    }
  }
  return predecessors;
}

EvaluationResult EvaluateScheduleWithPredecessors(
    const Instance& instance, const std::vector<Tick>& split,
    const MachinePredecessors& predecessors, bool fifo_ordering) {
  try {
    instance.Validate();
    ValidateSplit(instance, split);
    std::vector<std::string> errors =
        ValidateMachinePredecessors(instance, predecessors, fifo_ordering);
    return EvaluateScheduleGraph(
        instance, split,
        MachineOrders(static_cast<std::size_t>(instance.workers)), predecessors,
        std::move(errors), fifo_ordering);
  } catch (const Error& error) {
    EvaluationResult result;
    result.schedule.split = split;
    result.schedule.validation_errors.push_back(error.what());
    return result;
  }
}

EvaluationResult EvaluateSchedule(const Instance& instance,
                                  const std::vector<Tick>& split,
                                  const MachineOrders& orders,
                                  bool fifo_ordering) {
  try {
    instance.Validate();
    ValidateSplit(instance, split);
    std::vector<std::string> errors = ValidateOrders(instance, orders, fifo_ordering);
    const MachinePredecessors predecessors =
        errors.empty()
            ? ExtractMachinePredecessors(instance, orders, fifo_ordering)
            : MachinePredecessors{};
    if (errors.empty()) {
      errors =
          ValidateMachinePredecessors(instance, predecessors, fifo_ordering);
    }
    return EvaluateScheduleGraph(instance, split, orders, predecessors,
                                 std::move(errors), fifo_ordering);
  } catch (const Error& error) {
    EvaluationResult result;
    result.schedule.split = split;
    result.schedule.orders = orders;
    result.schedule.validation_errors.push_back(error.what());
    return result;
  }
}

ScheduleMetrics ComputeScheduleMetrics(const Instance& instance,
                                       const ScheduleSolution& schedule,
                                       bool fifo_ordering) {
  instance.Validate();
  ScheduleMetrics metrics;
  metrics.simulated_iteration_time = schedule.makespan;
  metrics.total_useful_work = TotalUsefulWork(instance);
  metrics.per_worker_busy_time.assign(
      static_cast<std::size_t>(instance.workers), 0);
  metrics.per_worker_idle_time.assign(
      static_cast<std::size_t>(instance.workers), 0);
  metrics.communication_blocked_time = 0;
  if (!schedule.ok() || schedule.operations_by_id.empty() ||
      schedule.operations_by_id.size() !=
          static_cast<std::size_t>(instance.OperationCount())) {
    return metrics;
  }

  for (const ScheduledOperation& op : schedule.operations_by_id) {
    if (op.worker < 0 || op.worker >= instance.workers) {
      throw Error("scheduled operation worker out of range");
    }
    metrics.per_worker_busy_time[static_cast<std::size_t>(op.worker)] =
        CheckedAdd(
            metrics.per_worker_busy_time[static_cast<std::size_t>(op.worker)],
            op.duration, "worker busy time");
  }
  // Profile durations can depend on the selected global layer ranges/roles.
  // Report actual replay work, not the abstract uniform-cost formula.
  if (instance.HasCostProfile()) {
    metrics.total_useful_work = 0;
    for (const auto& op : schedule.operations_by_id)
      metrics.total_useful_work = CheckedAdd(metrics.total_useful_work, op.duration, "profile useful work");
  }
  for (Index w = 0; w < instance.workers; ++w) {
    const Tick busy = metrics.per_worker_busy_time[static_cast<std::size_t>(w)];
    metrics.max_worker_load = std::max(metrics.max_worker_load, busy);
    metrics.per_worker_idle_time[static_cast<std::size_t>(w)] =
        std::max<Tick>(0, schedule.makespan - busy);
  }
  if (schedule.makespan > 0) {
    metrics.pipeline_utilization =
        static_cast<double>(metrics.total_useful_work) /
        static_cast<double>(CheckedMul(instance.workers, schedule.makespan,
                                       "pipeline utilization denominator"));
  }

  if (instance.stages > 0) {
    const OperationId first_backward =
        EncodeOperation(instance, 0, instance.stages);
    metrics.pipeline_fill_time =
        schedule
            .operations_by_id[static_cast<std::size_t>(first_backward.value)]
            .start;
    const OperationId last_backward_start =
        EncodeOperation(instance, instance.microbatches - 1, instance.stages);
    metrics.pipeline_drain_time = std::max<Tick>(
        0, schedule.makespan - schedule
                                   .operations_by_id[static_cast<std::size_t>(
                                       last_backward_start.value)]
                                   .start);
  }

  std::vector<Tick> machine_ready(
      static_cast<std::size_t>(instance.OperationCount()), 0);
  MachineOrders orders = schedule.orders;
  if (orders.empty()) {
    orders.assign(static_cast<std::size_t>(instance.workers), {});
    for (const ScheduledOperation& op : schedule.operations_by_id) {
      const OperationView view = DecodeOperation(instance, op.id);
      orders[static_cast<std::size_t>(view.worker)].push_back(op.id);
    }
    for (std::vector<OperationId>& worker_order : orders) {
      std::sort(
          worker_order.begin(), worker_order.end(),
          [&](OperationId a, OperationId b) {
            const ScheduledOperation& lhs =
                schedule.operations_by_id[static_cast<std::size_t>(a.value)];
            const ScheduledOperation& rhs =
                schedule.operations_by_id[static_cast<std::size_t>(b.value)];
            return std::tuple<Tick, Tick, Index>{lhs.start, lhs.end, a.value} <
                   std::tuple<Tick, Tick, Index>{rhs.start, rhs.end, b.value};
          });
    }
  }
  for (const std::vector<OperationId>& worker_order : orders) {
    for (std::size_t i = 1; i < worker_order.size(); ++i) {
      const OperationId previous = worker_order[i - 1];
      const OperationId current = worker_order[i];
      machine_ready[static_cast<std::size_t>(current.value)] =
          schedule.operations_by_id[static_cast<std::size_t>(previous.value)]
              .end;
    }
  }

  Tick communication_blocked = 0;
  for (const ScheduledOperation& op : schedule.operations_by_id) {
    const OperationView view = DecodeOperation(instance, op.id);
    Tick data_ready_without_comm = 0;
    Tick data_ready_with_comm = 0;
    if (const std::optional<OperationId> data =
            DataPredecessor(instance, op.id)) {
      const OperationView data_view = DecodeOperation(instance, *data);
      const Tick predecessor_end =
          schedule.operations_by_id[static_cast<std::size_t>(data->value)].end;
      data_ready_without_comm = predecessor_end;
      data_ready_with_comm = CheckedAdd(
          predecessor_end, instance.EdgeDelay(data_view.worker, view.worker),
          "communication ready time");
    }
    Tick fifo_ready = 0;
    if (fifo_ordering) {
      if (const std::optional<OperationId> fifo =
              FifoPredecessor(instance, op.id)) {
        fifo_ready =
            schedule.operations_by_id[static_cast<std::size_t>(fifo->value)]
                .end;
      }
    }
    const Tick ready_without_comm =
        std::max({machine_ready[static_cast<std::size_t>(op.id.value)],
                  fifo_ready, data_ready_without_comm});
    const Tick ready_with_comm =
        std::max({machine_ready[static_cast<std::size_t>(op.id.value)],
                  fifo_ready, data_ready_with_comm});
    if (ready_with_comm > ready_without_comm) {
      communication_blocked = CheckedAdd(communication_blocked,
                                         ready_with_comm - ready_without_comm,
                                         "communication blocked time");
    }
  }
  metrics.communication_blocked_time = communication_blocked;
  return metrics;
}

}  // namespace slackpipe
