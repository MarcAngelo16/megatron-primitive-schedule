#include "slackpipe/result_validator.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string_view>

#include "slackpipe/operation.h"

namespace slackpipe {
namespace {

using Clock = std::chrono::steady_clock;

struct ValidationEdge {
  OperationId from;
  OperationId to;
  Tick lag = 0;
  std::string type;
};

std::string JsonEscape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '"':
      case '\\':
        out << '\\' << ch;
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

template <typename T>
void WriteOptionalNumber(std::ostringstream& out,
                         const std::optional<T>& value) {
  if (value) {
    out << *value;
  } else {
    out << "null";
  }
}

void WriteOptionalString(std::ostringstream& out,
                         const std::optional<std::string>& value) {
  if (value) {
    out << "\"" << JsonEscape(*value) << "\"";
  } else {
    out << "null";
  }
}

double ElapsedSeconds(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

Index OperationPositionCountRaw(const Instance& instance) {
  return CheckedMul(2, instance.stages, "validator operation positions");
}

Index OperationCountRaw(const Instance& instance) {
  return CheckedMul(instance.microbatches, OperationPositionCountRaw(instance),
                    "validator operation count");
}

bool IsTerminalFeasibleStatusText(const std::optional<std::string>& status) {
  if (!status) return false;
  return *status == "OPTIMAL" || *status == "FEASIBLE";
}

bool InRangeOperation(const Instance& instance, OperationId id) {
  return id.value >= 0 && id.value < OperationCountRaw(instance);
}

StableOperationKey KeyForIdUnchecked(const Instance& instance, OperationId id) {
  const Index positions = OperationPositionCountRaw(instance);
  StableOperationKey key;
  key.id = id;
  key.microbatch = id.value / positions;
  key.operation_position = id.value % positions;
  const bool backward = key.operation_position >= instance.stages;
  key.phase = backward ? "B" : "F";
  key.stage = backward ? positions - 1 - key.operation_position
                       : key.operation_position;
  key.worker = instance.WorkerForStage(key.stage);
  return key;
}

std::string OperationNameUnchecked(const Instance& instance, OperationId id) {
  return StableOperationName(KeyForIdUnchecked(instance, id));
}

std::optional<OperationId> ParseOperationName(const Instance& instance,
                                              const std::string& name,
                                              std::string* error) {
  if (name.size() < 5 || (name[0] != 'F' && name[0] != 'B')) {
    if (error) *error = "operation name must start with F or B";
    return std::nullopt;
  }
  std::size_t cursor = 1;
  if (cursor >= name.size() ||
      !std::isdigit(static_cast<unsigned char>(name[cursor]))) {
    if (error) *error = "operation name is missing stage";
    return std::nullopt;
  }
  Index stage = 0;
  while (cursor < name.size() &&
         std::isdigit(static_cast<unsigned char>(name[cursor]))) {
    stage = CheckedAdd(CheckedMul(stage, 10, "operation name stage"),
                       name[cursor] - '0', "operation name stage");
    ++cursor;
  }
  if (cursor + 2 >= name.size() || name[cursor] != '(' ||
      name[cursor + 1] != 'b') {
    if (error) *error = "operation name must contain (b<index>)";
    return std::nullopt;
  }
  cursor += 2;
  if (cursor >= name.size() ||
      !std::isdigit(static_cast<unsigned char>(name[cursor]))) {
    if (error) *error = "operation name is missing microbatch";
    return std::nullopt;
  }
  Index microbatch = 0;
  while (cursor < name.size() &&
         std::isdigit(static_cast<unsigned char>(name[cursor]))) {
    microbatch = CheckedAdd(CheckedMul(microbatch, 10, "operation name batch"),
                            name[cursor] - '0', "operation name batch");
    ++cursor;
  }
  if (cursor + 1 != name.size() || name[cursor] != ')') {
    if (error) *error = "operation name has trailing text";
    return std::nullopt;
  }
  if (stage < 0 || stage >= instance.stages || microbatch < 0 ||
      microbatch >= instance.microbatches) {
    if (error) *error = "operation name refers to an out-of-range operation";
    return std::nullopt;
  }
  const Index position =
      name[0] == 'F' ? stage : OperationPositionCountRaw(instance) - 1 - stage;
  return OperationId{
      CheckedAdd(CheckedMul(microbatch, OperationPositionCountRaw(instance),
                            "operation name encoding"),
                 position, "operation name encoding")};
}

std::optional<OperationId> DataPredecessorRaw(const Instance& instance,
                                              OperationId id) {
  const StableOperationKey key = KeyForIdUnchecked(instance, id);
  if (key.operation_position == 0) return std::nullopt;
  return OperationId{id.value - 1};
}

bool IsDataPredecessorRaw(const Instance& instance, OperationId predecessor,
                          OperationId id) {
  const std::optional<OperationId> data = DataPredecessorRaw(instance, id);
  return data && *data == predecessor;
}

Tick EdgeDelayRaw(const Instance& instance, OperationId from, OperationId to) {
  const StableOperationKey from_key = KeyForIdUnchecked(instance, from);
  const StableOperationKey to_key = KeyForIdUnchecked(instance, to);
  return from_key.worker == to_key.worker ? 0 : instance.communication_ticks;
}

Tick DurationRaw(const Instance& instance, OperationId id,
                 const std::vector<Tick>& split) {
  const StableOperationKey key = KeyForIdUnchecked(instance, id);
  return instance.Duration(key.stage, key.phase == "B", split);
}

std::string EdgeLabel(const Instance& instance, const ValidationEdge& edge) {
  std::ostringstream out;
  out << OperationNameUnchecked(instance, edge.from) << " -" << edge.type
      << "-> " << OperationNameUnchecked(instance, edge.to);
  return out.str();
}

ResultValidationResult MakeBaseResult(const ResultValidationInput& input) {
  ResultValidationResult result;
  result.reported_makespan = input.reported_makespan;
  return result;
}

ResultValidationResult Fail(ResultValidationResult result,
                            Clock::time_point started, std::string error_code,
                            std::string error_category, std::string message) {
  result.passed = false;
  result.error_code = std::move(error_code);
  result.error_category = std::move(error_category);
  result.message = std::move(message);
  result.validation_runtime_seconds = ElapsedSeconds(started);
  return result;
}

ResultValidationResult Pass(ResultValidationResult result,
                            Clock::time_point started) {
  result.passed = true;
  result.error_code.clear();
  result.error_category.clear();
  result.message.clear();
  result.validation_runtime_seconds = ElapsedSeconds(started);
  return result;
}

ResultValidationResult FailCount(ResultValidationResult result,
                                 Clock::time_point started,
                                 const std::string& code,
                                 const std::string& category,
                                 const std::string& message, Index expected,
                                 Index actual) {
  result.expected_count = expected;
  result.actual_count = actual;
  return Fail(std::move(result), started, code, category, message);
}

std::optional<std::string> ValidateNameMatches(const Instance& instance,
                                               OperationId id,
                                               const std::string& name) {
  std::string parse_error;
  const std::optional<OperationId> parsed =
      ParseOperationName(instance, name, &parse_error);
  if (!parsed) return parse_error;
  if (*parsed != id) return "operation name/id mismatch";
  return std::nullopt;
}

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool bool_value = false;
  std::string number_text;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::map<std::string, JsonValue> object_value;

  const JsonValue* Find(const std::string& key) const {
    if (type != Type::kObject) return nullptr;
    const auto it = object_value.find(key);
    return it == object_value.end() ? nullptr : &it->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  JsonValue Parse() {
    JsonValue value = ParseValue();
    SkipWhitespace();
    if (cursor_ != text_.size()) throw Error("trailing JSON content");
    return value;
  }

 private:
  void SkipWhitespace() {
    while (cursor_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[cursor_]))) {
      ++cursor_;
    }
  }

  char Peek() {
    SkipWhitespace();
    if (cursor_ >= text_.size()) throw Error("unexpected end of JSON");
    return text_[cursor_];
  }

  bool Consume(char expected) {
    SkipWhitespace();
    if (cursor_ < text_.size() && text_[cursor_] == expected) {
      ++cursor_;
      return true;
    }
    return false;
  }

  void Expect(char expected) {
    if (!Consume(expected)) {
      throw Error(std::string("expected JSON character ") + expected);
    }
  }

  JsonValue ParseValue() {
    const char ch = Peek();
    if (ch == 'n') return ParseLiteral("null", JsonValue::Type::kNull);
    if (ch == 't') return ParseBool("true", true);
    if (ch == 'f') return ParseBool("false", false);
    if (ch == '"') return ParseStringValue();
    if (ch == '[') return ParseArray();
    if (ch == '{') return ParseObject();
    return ParseNumber();
  }

  JsonValue ParseLiteral(std::string_view literal, JsonValue::Type type) {
    if (text_.substr(cursor_, literal.size()) != literal) {
      throw Error("invalid JSON literal");
    }
    cursor_ += literal.size();
    JsonValue value;
    value.type = type;
    return value;
  }

  JsonValue ParseBool(std::string_view literal, bool bool_value) {
    JsonValue value = ParseLiteral(literal, JsonValue::Type::kBool);
    value.bool_value = bool_value;
    return value;
  }

  std::string ParseStringRaw() {
    Expect('"');
    std::string out;
    while (cursor_ < text_.size()) {
      const char ch = text_[cursor_++];
      if (ch == '"') return out;
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (cursor_ >= text_.size()) throw Error("unterminated JSON escape");
      const char escaped = text_[cursor_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          out.push_back(escaped);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u':
          if (cursor_ + 4 > text_.size())
            throw Error("short JSON unicode escape");
          out.push_back('?');
          cursor_ += 4;
          break;
        default:
          throw Error("invalid JSON escape");
      }
    }
    throw Error("unterminated JSON string");
  }

  JsonValue ParseStringValue() {
    JsonValue value;
    value.type = JsonValue::Type::kString;
    value.string_value = ParseStringRaw();
    return value;
  }

  JsonValue ParseNumber() {
    SkipWhitespace();
    const std::size_t begin = cursor_;
    if (cursor_ < text_.size() && text_[cursor_] == '-') ++cursor_;
    if (cursor_ >= text_.size() ||
        !std::isdigit(static_cast<unsigned char>(text_[cursor_]))) {
      throw Error("invalid JSON number");
    }
    while (cursor_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[cursor_]))) {
      ++cursor_;
    }
    if (cursor_ < text_.size() && text_[cursor_] == '.') {
      ++cursor_;
      while (cursor_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[cursor_]))) {
        ++cursor_;
      }
    }
    if (cursor_ < text_.size() &&
        (text_[cursor_] == 'e' || text_[cursor_] == 'E')) {
      ++cursor_;
      if (cursor_ < text_.size() &&
          (text_[cursor_] == '+' || text_[cursor_] == '-')) {
        ++cursor_;
      }
      while (cursor_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[cursor_]))) {
        ++cursor_;
      }
    }
    JsonValue value;
    value.type = JsonValue::Type::kNumber;
    value.number_text = std::string(text_.substr(begin, cursor_ - begin));
    return value;
  }

  JsonValue ParseArray() {
    JsonValue value;
    value.type = JsonValue::Type::kArray;
    Expect('[');
    if (Consume(']')) return value;
    while (true) {
      value.array_value.push_back(ParseValue());
      if (Consume(']')) return value;
      Expect(',');
    }
  }

  JsonValue ParseObject() {
    JsonValue value;
    value.type = JsonValue::Type::kObject;
    Expect('{');
    if (Consume('}')) return value;
    while (true) {
      const std::string key = ParseStringRaw();
      Expect(':');
      value.object_value[key] = ParseValue();
      if (Consume('}')) return value;
      Expect(',');
    }
  }

  std::string_view text_;
  std::size_t cursor_ = 0;
};

bool IsNull(const JsonValue* value) {
  return value == nullptr || value->type == JsonValue::Type::kNull;
}

std::optional<std::string> JsonString(const JsonValue* value) {
  if (IsNull(value)) return std::nullopt;
  if (value->type != JsonValue::Type::kString) {
    throw Error("expected JSON string");
  }
  return value->string_value;
}

std::optional<double> JsonDouble(const JsonValue* value) {
  if (IsNull(value)) return std::nullopt;
  if (value->type != JsonValue::Type::kNumber) {
    throw Error("expected JSON number");
  }
  return std::stod(value->number_text);
}

std::optional<Tick> JsonTick(const JsonValue* value) {
  if (IsNull(value)) return std::nullopt;
  if (value->type != JsonValue::Type::kNumber) {
    throw Error("expected JSON integer");
  }
  if (value->number_text.find_first_of(".eE") != std::string::npos) {
    const double numeric = std::stod(value->number_text);
    const double rounded = std::round(numeric);
    if (std::fabs(numeric - rounded) > 1e-9) {
      throw Error("expected integral JSON number");
    }
    return static_cast<Tick>(rounded);
  }
  return std::stoll(value->number_text);
}

std::optional<bool> JsonBool(const JsonValue* value) {
  if (IsNull(value)) return std::nullopt;
  if (value->type != JsonValue::Type::kBool) {
    throw Error("expected JSON bool");
  }
  return value->bool_value;
}

std::optional<Tick> FieldTick(const JsonValue& object, const std::string& key) {
  return JsonTick(object.Find(key));
}

std::optional<double> FieldDouble(const JsonValue& object,
                                  const std::string& key) {
  return JsonDouble(object.Find(key));
}

std::optional<std::string> FieldString(const JsonValue& object,
                                       const std::string& key) {
  return JsonString(object.Find(key));
}

std::optional<bool> FieldBool(const JsonValue& object, const std::string& key) {
  return JsonBool(object.Find(key));
}

std::optional<std::vector<Tick>> FieldTickArray(const JsonValue& object,
                                                const std::string& key) {
  const JsonValue* value = object.Find(key);
  if (IsNull(value)) return std::nullopt;
  if (value->type != JsonValue::Type::kArray) {
    throw Error("expected JSON array for " + key);
  }
  std::vector<Tick> values;
  values.reserve(value->array_value.size());
  for (const JsonValue& element : value->array_value) {
    values.push_back(*JsonTick(&element));
  }
  return values;
}

std::optional<std::vector<std::vector<std::string>>> FieldStringMatrix(
    const JsonValue& object, const std::string& key) {
  const JsonValue* value = object.Find(key);
  if (IsNull(value)) return std::nullopt;
  if (value->type != JsonValue::Type::kArray) {
    throw Error("expected JSON array for " + key);
  }
  std::vector<std::vector<std::string>> matrix;
  matrix.reserve(value->array_value.size());
  for (const JsonValue& row : value->array_value) {
    if (row.type != JsonValue::Type::kArray) {
      throw Error("expected nested JSON array for " + key);
    }
    std::vector<std::string> names;
    names.reserve(row.array_value.size());
    for (const JsonValue& element : row.array_value) {
      names.push_back(*JsonString(&element));
    }
    matrix.push_back(std::move(names));
  }
  return matrix;
}

ResultValidationInput InputFromJson(const JsonValue& root) {
  if (root.type != JsonValue::Type::kObject) {
    throw Error("result JSON root must be an object");
  }
  const JsonValue* canonical = root.Find("canonical_result");
  const JsonValue& source =
      canonical != nullptr && canonical->type == JsonValue::Type::kObject
          ? *canonical
          : root;

  ResultValidationInput input;
  input.instance.microbatches =
      FieldTick(source, "micro_batches")
          .value_or(FieldTick(root, "microbatches")
                        .value_or(FieldTick(root, "B").value_or(0)));
  input.instance.stages =
      FieldTick(source, "logical_stages")
          .value_or(FieldTick(root, "stages")
                        .value_or(FieldTick(root, "N").value_or(0)));
  input.instance.workers =
      FieldTick(source, "physical_workers")
          .value_or(FieldTick(root, "workers")
                        .value_or(FieldTick(root, "J").value_or(
                            FieldTick(root, "num_workers").value_or(0))));
  input.instance.total_layers =
      FieldTick(source, "total_layers")
          .value_or(FieldTick(root, "total_layers")
                        .value_or(FieldTick(root, "L").value_or(0)));
  input.instance.min_layers =
      FieldTick(source, "min_layers")
          .value_or(FieldTick(root, "min_layers").value_or(1));
  if (const auto* mapping = source.Find("stage_to_worker_mapping")) {
    if (mapping->type != JsonValue::Type::kArray ||
        mapping->array_value.size() != static_cast<std::size_t>(input.instance.stages)) {
      throw Error("invalid stage_to_worker_mapping");
    }
    input.instance.stage_to_worker.assign(input.instance.stages, -1);
    for (const auto& entry : mapping->array_value) {
      const Index stage = FieldTick(entry, "stage").value_or(-1);
      if (stage < 0 || stage >= input.instance.stages ||
          input.instance.stage_to_worker[stage] != -1) {
        throw Error("invalid or duplicate placement stage");
      }
      input.instance.stage_to_worker[stage] = FieldTick(entry, "worker").value_or(-1);
    }
  }

  const Tick forward_num =
      FieldTick(source, "forward_cost_ratio_numerator")
          .value_or(FieldTick(root, "ratio_den").value_or(1));
  const Tick forward_den =
      FieldTick(source, "forward_cost_ratio_denominator").value_or(1);
  const Tick backward_num =
      FieldTick(source, "backward_cost_ratio_numerator")
          .value_or(FieldTick(root, "ratio_num").value_or(1));
  const Tick backward_den =
      FieldTick(source, "backward_cost_ratio_denominator").value_or(1);
  input.instance.backward_ratio_den = forward_num;
  input.instance.backward_ratio_num = backward_num;
  input.instance.communication_ticks =
      FieldTick(source, "communication_ticks")
          .value_or(FieldTick(root, "communication_ticks").value_or(0));

  if (const auto* costs = root.Find("duration_model_ticks")) {
    if (costs->type != JsonValue::Type::kObject) throw Error("invalid duration_model_ticks");
    auto required = [&](const char* name) {
      auto values = FieldTickArray(*costs, name);
      if (!values) throw Error(std::string("missing duration_model_ticks.") + name);
      return *values;
    };
    input.instance.profile_forward_slope_ticks = required("forward_slope");
    input.instance.profile_backward_slope_ticks = required("backward_slope");
    input.instance.profile_forward_bias_ticks = required("forward_bias");
    input.instance.profile_backward_bias_ticks = required("backward_bias");
    input.instance.profile_prefix_forward_ticks = required("prefix_forward");
    input.instance.profile_prefix_backward_ticks = required("prefix_backward");
    input.instance.profile_role_forward_bias_ticks = required("role_forward_bias");
    input.instance.profile_role_backward_bias_ticks = required("role_backward_bias");
  }

  input.communication_model = FieldString(source, "communication_model")
                                  .value_or(input.communication_model);
  if (forward_den != 1 || backward_den != 1) {
    input.communication_model = "unsupported_rational_duration";
  }
  input.communication_alpha = FieldDouble(source, "communication_alpha");
  input.communication_beta = FieldDouble(source, "communication_beta");
  input.communication_payload = FieldString(source, "communication_payload");
  input.method_name = FieldString(source, "canonical_method");
  input.solver_status_raw = FieldString(source, "solver_status_raw");
  input.reported_status = FieldString(source, "reported_status");
  input.fifo_ordering_requested = FieldBool(source, "fifo_ordering_requested");
  if (!input.fifo_ordering_requested) {
    input.fifo_ordering_requested = FieldBool(root, "fifo_ordering_requested");
  }
  input.fifo_ordering_effective = FieldBool(source, "fifo_ordering_effective");
  if (!input.fifo_ordering_effective) {
    input.fifo_ordering_effective = FieldBool(root, "fifo_ordering_effective");
  }
  input.fifo_constraint_count = FieldTick(source, "fifo_constraint_count");
  if (!input.fifo_constraint_count) {
    input.fifo_constraint_count = FieldTick(root, "fifo_constraint_count");
  }
  input.fifo_ordering = input.fifo_ordering_effective.value_or(
      FieldBool(source, "fifo_ordering")
          .value_or(FieldBool(root, "fifo_ordering").value_or(true)));
  input.feasible_claimed =
      FieldBool(source, "feasible")
          .value_or(FieldBool(root, "ok")
                        .value_or(IsTerminalFeasibleStatusText(
                            input.reported_status)));
  input.reported_makespan = FieldTick(source, "makespan");
  if (!input.reported_makespan) {
    input.reported_makespan = FieldTick(root, "makespan_ticks");
  }
  if (!input.reported_makespan) {
    input.reported_makespan = FieldTick(root, "makespan");
  }
  if (!input.reported_makespan) {
    input.reported_makespan = FieldTick(root, "simulated_iteration_time");
  }
  input.selected_partition = FieldTickArray(source, "selected_partition");
  if (!input.selected_partition) {
    input.selected_partition = FieldTickArray(root, "split");
  }
  input.worker_order_names =
      FieldStringMatrix(source, "worker_local_operation_order");
  if (!input.worker_order_names) {
    input.worker_order_names =
        FieldStringMatrix(root, "worker_local_operation_order");
  }

  if (const JsonValue* predecessors =
          source.Find("derived_worker_predecessors");
      !IsNull(predecessors)) {
    if (predecessors->type != JsonValue::Type::kArray) {
      throw Error("derived_worker_predecessors must be an array");
    }
    std::vector<SerializedWorkerPredecessorRecord> records;
    records.reserve(predecessors->array_value.size());
    for (const JsonValue& element : predecessors->array_value) {
      if (element.type != JsonValue::Type::kObject) {
        throw Error("derived predecessor record must be an object");
      }
      SerializedWorkerPredecessorRecord record;
      record.operation_id =
          OperationId{FieldTick(element, "operation_id").value_or(-1)};
      record.operation_name = FieldString(element, "operation");
      record.predecessor_id =
          OperationId{FieldTick(element, "predecessor_id").value_or(-1)};
      record.predecessor_name = FieldString(element, "predecessor");
      record.worker = FieldTick(element, "worker");
      records.push_back(std::move(record));
    }
    input.derived_worker_predecessors = std::move(records);
    input.require_serialized_worker_predecessors = true;
  } else if (canonical != nullptr) {
    input.require_serialized_worker_predecessors = true;
  }

  if (const JsonValue* operations = root.Find("operations");
      !IsNull(operations)) {
    if (operations->type != JsonValue::Type::kArray) {
      throw Error("operations must be an array");
    }
    std::vector<SerializedOperationRecord> records;
    records.reserve(operations->array_value.size());
    for (const JsonValue& element : operations->array_value) {
      if (element.type != JsonValue::Type::kObject) {
        throw Error("operation record must be an object");
      }
      SerializedOperationRecord record;
      record.id = OperationId{FieldTick(element, "id").value_or(-1)};
      record.name = FieldString(element, "name");
      record.microbatch = FieldTick(element, "microbatch");
      record.operation_position = FieldTick(element, "chain_index");
      record.stage = FieldTick(element, "stage");
      record.phase = FieldString(element, "kind");
      record.worker = FieldTick(element, "worker");
      record.start = FieldTick(element, "start");
      record.end = FieldTick(element, "end");
      record.duration = FieldTick(element, "duration");
      records.push_back(std::move(record));
    }
    input.serialized_operations = std::move(records);
    input.require_serialized_operations = true;
  } else if (canonical != nullptr && input.feasible_claimed) {
    input.require_serialized_operations = true;
  }
  return input;
}

std::string MismatchMessage(OperationId operation, OperationId expected,
                            OperationId actual, const Instance& instance) {
  std::ostringstream out;
  out << "derived predecessor for "
      << OperationNameUnchecked(instance, operation) << " expected "
      << OperationNameUnchecked(instance, expected) << " but found "
      << OperationNameUnchecked(instance, actual);
  return out.str();
}

}  // namespace

StableOperationKey ExpectedOperationKey(const Instance& instance,
                                        OperationId id) {
  instance.Validate();
  if (!InRangeOperation(instance, id)) throw Error("operation id out of range");
  return KeyForIdUnchecked(instance, id);
}

std::string StableOperationName(const StableOperationKey& key) {
  std::ostringstream out;
  out << key.phase << key.stage << "(b" << key.microbatch << ")";
  return out.str();
}

ResultValidationInput ValidationInputFromSchedule(
    const Instance& instance, const ScheduleSolution& schedule,
    const std::string& status, std::optional<std::string> method_name,
    bool fifo_ordering) {
  ResultValidationInput input;
  input.instance = instance;
  input.fifo_ordering = fifo_ordering;
  input.fifo_ordering_requested = fifo_ordering;
  input.fifo_ordering_effective = fifo_ordering;
  input.selected_partition = schedule.split;
  input.worker_orders = schedule.orders;
  input.serialized_operations.emplace();
  input.serialized_operations->reserve(schedule.operations_by_id.size());
  for (const ScheduledOperation& operation : schedule.operations_by_id) {
    SerializedOperationRecord record;
    record.id = operation.id;
    record.worker = operation.worker;
    record.start = operation.start;
    record.end = operation.end;
    record.duration = operation.duration;
    input.serialized_operations->push_back(record);
  }
  input.reported_makespan = schedule.makespan;
  input.feasible_claimed =
      schedule.ok() || IsTerminalFeasibleStatusText(status);
  input.solver_status_raw = status;
  input.reported_status = status;
  input.method_name = std::move(method_name);
  return input;
}

ResultValidationResult ValidateResult(const ResultValidationInput& input) {
  const auto started = Clock::now();
  ResultValidationResult result = MakeBaseResult(input);

  try {
    input.instance.Validate();
    result.expected_operation_count = OperationCountRaw(input.instance);
  } catch (const std::exception& error) {
    return Fail(std::move(result), started, "instance_invalid", "instance",
                error.what());
  }

  if (input.communication_model != "constant_inter_worker_delay") {
    return Fail(std::move(result), started, "unsupported_communication_model",
                "communication",
                "validator supports only constant_inter_worker_delay");
  }
  if (input.communication_alpha || input.communication_beta ||
      input.communication_payload) {
    return Fail(std::move(result), started, "unsupported_communication_model",
                "communication",
                "alpha-beta communication metadata is not supported by this "
                "implementation");
  }

  if (!input.feasible_claimed) {
    if (input.reported_makespan) {
      result.actual_tick = *input.reported_makespan;
      return Fail(std::move(result), started, "infeasible_has_makespan",
                  "outcome",
                  "a non-feasible result must not report a makespan");
    }
    result.warnings.push_back("no_feasible_solution_to_validate");
    return Pass(std::move(result), started);
  }

  if (!input.reported_makespan) {
    return Fail(std::move(result), started, "makespan_missing", "outcome",
                "a feasible result must report a makespan");
  }

  if (!input.selected_partition) {
    return Fail(std::move(result), started, "partition_missing", "partition",
                "selected partition is missing");
  }
  const std::vector<Tick>& split = *input.selected_partition;
  if (split.size() != static_cast<std::size_t>(input.instance.stages)) {
    return FailCount(std::move(result), started, "partition_length_mismatch",
                     "partition",
                     "partition length does not match logical stage count",
                     input.instance.stages, static_cast<Index>(split.size()));
  }
  Tick layer_sum = 0;
  for (Index stage = 0; stage < input.instance.stages; ++stage) {
    const Tick value = split[static_cast<std::size_t>(stage)];
    if (value < input.instance.min_layers) {
      result.offending_operations.push_back("stage " + std::to_string(stage));
      result.expected_tick = input.instance.min_layers;
      result.actual_tick = value;
      return Fail(std::move(result), started, "partition_empty_stage",
                  "partition",
                  "partition violates the minimum layers per stage");
    }
    layer_sum = CheckedAdd(layer_sum, value, "validator partition sum");
  }
  if (layer_sum != input.instance.total_layers) {
    result.expected_tick = input.instance.total_layers;
    result.actual_tick = layer_sum;
    return Fail(std::move(result), started, "partition_sum_mismatch",
                "partition",
                "partition layer sum does not match total layer count");
  }
  for (Index stage = 0; stage < input.instance.stages; ++stage) {
    (void)input.instance.Duration(stage, false, split);
    (void)input.instance.Duration(stage, true, split);
  }
  if (input.fixed_partition_reference) {
    if (input.fixed_partition_reference->size() != split.size()) {
      return FailCount(
          std::move(result), started, "partition_fixed_value_mismatch",
          "partition",
          "fixed partition reference length does not match selected partition",
          static_cast<Index>(input.fixed_partition_reference->size()),
          static_cast<Index>(split.size()));
    }
    for (std::size_t i = 0; i < split.size(); ++i) {
      if ((*input.fixed_partition_reference)[i] != split[i]) {
        result.offending_operations.push_back("stage " + std::to_string(i));
        result.expected_tick = (*input.fixed_partition_reference)[i];
        result.actual_tick = split[i];
        return Fail(std::move(result), started,
                    "partition_fixed_value_mismatch", "partition",
                    "selected partition differs from fixed reference "
                    "partition");
      }
    }
  }

  MachineOrders orders;
  if (input.worker_orders) {
    orders = *input.worker_orders;
  }
  if (input.worker_order_names) {
    if (input.worker_order_names->size() !=
        static_cast<std::size_t>(input.instance.workers)) {
      return FailCount(std::move(result), started,
                       "worker_order_worker_count_mismatch", "schedule",
                       "worker order matrix does not match worker count",
                       input.instance.workers,
                       static_cast<Index>(input.worker_order_names->size()));
    }
    MachineOrders parsed_orders(
        static_cast<std::size_t>(input.instance.workers));
    for (Index worker = 0; worker < input.instance.workers; ++worker) {
      const std::vector<std::string>& names =
          (*input.worker_order_names)[static_cast<std::size_t>(worker)];
      parsed_orders[static_cast<std::size_t>(worker)].reserve(names.size());
      for (const std::string& name : names) {
        std::string parse_error;
        const std::optional<OperationId> id =
            ParseOperationName(input.instance, name, &parse_error);
        if (!id) {
          result.offending_operations.push_back(name);
          return Fail(std::move(result), started, "operation_name_malformed",
                      "schedule", parse_error);
        }
        parsed_orders[static_cast<std::size_t>(worker)].push_back(*id);
      }
    }
    if (input.worker_orders && parsed_orders != *input.worker_orders) {
      return Fail(std::move(result), started, "worker_order_name_mismatch",
                  "schedule",
                  "worker order names do not match numeric operation ids");
    }
    orders = std::move(parsed_orders);
  }
  if (orders.empty()) {
    return Fail(std::move(result), started, "worker_order_missing", "schedule",
                "worker-local operation order is missing");
  }
  if (orders.size() != static_cast<std::size_t>(input.instance.workers)) {
    return FailCount(std::move(result), started,
                     "worker_order_worker_count_mismatch", "schedule",
                     "worker order matrix does not match worker count",
                     input.instance.workers, static_cast<Index>(orders.size()));
  }

  const Index expected_count = OperationCountRaw(input.instance);
  std::vector<int> seen(static_cast<std::size_t>(expected_count), 0);
  for (Index worker = 0; worker < input.instance.workers; ++worker) {
    for (OperationId id : orders[static_cast<std::size_t>(worker)]) {
      ++result.actual_operation_count;
      if (!InRangeOperation(input.instance, id)) {
        result.offending_operations.push_back("id " + std::to_string(id.value));
        return Fail(std::move(result), started, "operation_unknown", "schedule",
                    "worker order contains unknown operation id");
      }
      const StableOperationKey key = KeyForIdUnchecked(input.instance, id);
      if (key.worker != worker) {
        result.offending_worker = worker;
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, id));
        return Fail(std::move(result), started, "worker_order_wrong_worker",
                    "schedule",
                    "operation appears in the wrong worker-local order");
      }
      int& count = seen[static_cast<std::size_t>(id.value)];
      ++count;
      if (count > 1) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, id));
        return Fail(std::move(result), started, "operation_duplicate",
                    "schedule", "operation appears more than once");
      }
    }
  }
  if (result.actual_operation_count != expected_count) {
    for (Index id = 0; id < expected_count; ++id) {
      if (seen[static_cast<std::size_t>(id)] == 0) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, OperationId{id}));
        break;
      }
    }
    return FailCount(std::move(result), started, "operation_missing",
                     "schedule", "worker orders do not contain every operation",
                     expected_count, result.actual_operation_count);
  }

  std::vector<ValidationEdge> edges;
  edges.reserve(static_cast<std::size_t>(expected_count * 3));
  const Index positions = OperationPositionCountRaw(input.instance);
  for (Index microbatch = 0; microbatch < input.instance.microbatches;
       ++microbatch) {
    for (Index position = 1; position < positions; ++position) {
      const OperationId from{microbatch * positions + position - 1};
      const OperationId to{microbatch * positions + position};
      edges.push_back(ValidationEdge{
          from, to, EdgeDelayRaw(input.instance, from, to), "data"});
      ++result.data_edge_count;
    }
  }
  if (input.fifo_ordering) {
    for (Index microbatch = 1; microbatch < input.instance.microbatches;
         ++microbatch) {
      for (Index position = 0; position < positions; ++position) {
        const OperationId from{(microbatch - 1) * positions + position};
        const OperationId to{microbatch * positions + position};
        edges.push_back(ValidationEdge{from, to, 0, "fifo"});
        ++result.fifo_edge_count;
      }
    }
  }

  std::map<Index, OperationId> expected_worker_predecessors;
  for (Index worker = 0; worker < input.instance.workers; ++worker) {
    const std::vector<OperationId>& order =
        orders[static_cast<std::size_t>(worker)];
    for (std::size_t i = 1; i < order.size(); ++i) {
      const OperationId predecessor = order[i - 1];
      const OperationId id = order[i];
      if (predecessor == id) {
        result.offending_worker = worker;
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, id));
        return Fail(std::move(result), started, "malformed_self_edge",
                    "schedule", "worker order contains a self edge");
      }
      const StableOperationKey predecessor_key =
          KeyForIdUnchecked(input.instance, predecessor);
      const StableOperationKey key = KeyForIdUnchecked(input.instance, id);
      if (predecessor_key.operation_position == key.operation_position) {
        if (input.fifo_ordering &&
            predecessor_key.microbatch >= key.microbatch) {
          result.offending_worker = worker;
          result.offending_operations.push_back(
              OperationNameUnchecked(input.instance, predecessor));
          result.offending_operations.push_back(
              OperationNameUnchecked(input.instance, id));
          result.offending_edge_type = "fifo";
          return Fail(std::move(result), started, "worker_order_reverses_fifo",
                      "schedule",
                      "same-position worker order contradicts FIFO order");
        }
        if (input.fifo_ordering) continue;
      }
      if (IsDataPredecessorRaw(input.instance, predecessor, id)) continue;
      expected_worker_predecessors[id.value] = predecessor;
      edges.push_back(ValidationEdge{predecessor, id, 0, "worker"});
      ++result.worker_edge_count;
    }
  }

  if (input.require_serialized_worker_predecessors &&
      !input.derived_worker_predecessors) {
    return Fail(std::move(result), started, "derived_predecessors_missing",
                "provenance",
                "canonical result is missing derived worker predecessors");
  }
  if (input.derived_worker_predecessors) {
    std::map<Index, OperationId> actual_worker_predecessors;
    for (const SerializedWorkerPredecessorRecord& record :
         *input.derived_worker_predecessors) {
      if (!InRangeOperation(input.instance, record.operation_id) ||
          !InRangeOperation(input.instance, record.predecessor_id)) {
        return Fail(std::move(result), started, "worker_predecessor_unknown_op",
                    "provenance",
                    "derived worker predecessor references unknown operation");
      }
      if (record.operation_id == record.predecessor_id) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.operation_id));
        return Fail(std::move(result), started, "malformed_self_edge",
                    "provenance",
                    "derived worker predecessor contains a self edge");
      }
      if (record.operation_name) {
        if (const std::optional<std::string> mismatch = ValidateNameMatches(
                input.instance, record.operation_id, *record.operation_name)) {
          result.offending_operations.push_back(*record.operation_name);
          return Fail(std::move(result), started, "operation_name_malformed",
                      "provenance", *mismatch);
        }
      }
      if (record.predecessor_name) {
        if (const std::optional<std::string> mismatch =
                ValidateNameMatches(input.instance, record.predecessor_id,
                                    *record.predecessor_name)) {
          result.offending_operations.push_back(*record.predecessor_name);
          return Fail(std::move(result), started, "operation_name_malformed",
                      "provenance", *mismatch);
        }
      }
      const StableOperationKey operation_key =
          KeyForIdUnchecked(input.instance, record.operation_id);
      const StableOperationKey predecessor_key =
          KeyForIdUnchecked(input.instance, record.predecessor_id);
      if (operation_key.worker != predecessor_key.worker ||
          (record.worker && *record.worker != operation_key.worker)) {
        result.offending_worker = record.worker;
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.predecessor_id));
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.operation_id));
        return Fail(std::move(result), started,
                    "worker_predecessor_cross_worker", "provenance",
                    "derived worker predecessor is not worker-local");
      }
      if (input.fifo_ordering && operation_key.operation_position ==
                                     predecessor_key.operation_position) {
        return Fail(std::move(result), started,
                    "worker_predecessor_same_position", "provenance",
                    "same-position FIFO edges must not be serialized as "
                    "worker predecessors");
      }
      if (IsDataPredecessorRaw(input.instance, record.predecessor_id,
                               record.operation_id)) {
        return Fail(std::move(result), started,
                    "worker_predecessor_data_duplicate", "provenance",
                    "data predecessors must not be duplicated as worker "
                    "predecessors");
      }
      actual_worker_predecessors[record.operation_id.value] =
          record.predecessor_id;
    }
    if (actual_worker_predecessors.size() !=
        expected_worker_predecessors.size()) {
      return FailCount(
          std::move(result), started, "derived_predecessor_mismatch",
          "provenance",
          "serialized derived worker predecessor count does not match worker "
          "order",
          static_cast<Index>(expected_worker_predecessors.size()),
          static_cast<Index>(actual_worker_predecessors.size()));
    }
    for (const auto& [id_value, predecessor] : expected_worker_predecessors) {
      const auto it = actual_worker_predecessors.find(id_value);
      if (it == actual_worker_predecessors.end()) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, OperationId{id_value}));
        return Fail(std::move(result), started, "derived_predecessor_mismatch",
                    "provenance",
                    "serialized derived worker predecessors are missing an "
                    "expected edge");
      }
      if (it->second != predecessor) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, OperationId{id_value}));
        return Fail(std::move(result), started, "derived_predecessor_mismatch",
                    "provenance",
                    MismatchMessage(OperationId{id_value}, predecessor,
                                    it->second, input.instance));
      }
    }
  }
  result.total_edge_count = result.data_edge_count + result.fifo_edge_count +
                            result.worker_edge_count;

  std::vector<Tick> durations(static_cast<std::size_t>(expected_count), 0);
  for (Index id = 0; id < expected_count; ++id) {
    durations[static_cast<std::size_t>(id)] =
        DurationRaw(input.instance, OperationId{id}, split);
  }

  std::vector<std::vector<ValidationEdge>> outgoing(
      static_cast<std::size_t>(expected_count));
  std::vector<Index> indegree(static_cast<std::size_t>(expected_count), 0);
  for (const ValidationEdge& edge : edges) {
    if (edge.from == edge.to) {
      result.offending_operations.push_back(
          OperationNameUnchecked(input.instance, edge.from));
      result.offending_edge_type = edge.type;
      return Fail(std::move(result), started, "malformed_self_edge", "graph",
                  "dependency graph contains a self edge");
    }
    outgoing[static_cast<std::size_t>(edge.from.value)].push_back(edge);
    ++indegree[static_cast<std::size_t>(edge.to.value)];
  }
  for (std::vector<ValidationEdge>& node_edges : outgoing) {
    std::sort(node_edges.begin(), node_edges.end(),
              [](const ValidationEdge& a, const ValidationEdge& b) {
                if (a.to.value != b.to.value) return a.to.value < b.to.value;
                if (a.type != b.type) return a.type < b.type;
                return a.lag < b.lag;
              });
  }

  std::priority_queue<Index, std::vector<Index>, std::greater<Index>> ready;
  for (Index id = 0; id < expected_count; ++id) {
    if (indegree[static_cast<std::size_t>(id)] == 0) ready.push(id);
  }
  std::vector<Tick> earliest_start(static_cast<std::size_t>(expected_count), 0);
  std::vector<Tick> earliest_end(static_cast<std::size_t>(expected_count), 0);
  std::vector<Index> topo_order;
  topo_order.reserve(static_cast<std::size_t>(expected_count));
  while (!ready.empty()) {
    const Index id_value = ready.top();
    ready.pop();
    topo_order.push_back(id_value);
    earliest_end[static_cast<std::size_t>(id_value)] =
        CheckedAdd(earliest_start[static_cast<std::size_t>(id_value)],
                   durations[static_cast<std::size_t>(id_value)],
                   "validator earliest end");
    for (const ValidationEdge& edge :
         outgoing[static_cast<std::size_t>(id_value)]) {
      const Tick candidate =
          CheckedAdd(earliest_end[static_cast<std::size_t>(id_value)], edge.lag,
                     "validator edge lag");
      Tick& current = earliest_start[static_cast<std::size_t>(edge.to.value)];
      if (candidate > current) current = candidate;
      Index& degree = indegree[static_cast<std::size_t>(edge.to.value)];
      --degree;
      if (degree == 0) ready.push(edge.to.value);
    }
  }
  if (static_cast<Index>(topo_order.size()) != expected_count) {
    std::set<Index> cyclic;
    for (Index id = 0; id < expected_count; ++id) {
      if (indegree[static_cast<std::size_t>(id)] > 0) cyclic.insert(id);
    }
    for (Index id : cyclic) {
      for (const ValidationEdge& edge :
           outgoing[static_cast<std::size_t>(id)]) {
        if (cyclic.count(edge.to.value) != 0) {
          result.cycle_witness.push_back(EdgeLabel(input.instance, edge));
          if (!result.offending_edge_type)
            result.offending_edge_type = edge.type;
          if (result.cycle_witness.size() >= 8) break;
        }
      }
      if (result.cycle_witness.size() >= 8) break;
    }
    if (result.cycle_witness.empty()) {
      for (Index id : cyclic) {
        result.cycle_witness.push_back(
            OperationNameUnchecked(input.instance, OperationId{id}));
        if (result.cycle_witness.size() >= 8) break;
      }
    }
    return Fail(std::move(result), started, "dependency_cycle", "graph",
                "data, FIFO, and worker predecessor edges are cyclic");
  }
  Tick reconstructed_makespan = 0;
  for (Tick end : earliest_end)
    reconstructed_makespan = std::max(reconstructed_makespan, end);
  result.reconstructed_makespan = reconstructed_makespan;

  std::vector<std::optional<SerializedOperationRecord>> intervals(
      static_cast<std::size_t>(expected_count));
  if (input.require_serialized_operations && !input.serialized_operations) {
    return Fail(std::move(result), started, "operations_missing",
                "serialization",
                "canonical result is missing serialized operation timings");
  }
  if (input.serialized_operations) {
    result.intervals_checked =
        static_cast<Index>(input.serialized_operations->size());
    std::vector<int> operation_seen(static_cast<std::size_t>(expected_count),
                                    0);
    for (const SerializedOperationRecord& record :
         *input.serialized_operations) {
      if (!InRangeOperation(input.instance, record.id)) {
        result.offending_operations.push_back("id " +
                                              std::to_string(record.id.value));
        return Fail(std::move(result), started, "operation_unknown",
                    "serialization",
                    "serialized operation id is outside the instance");
      }
      int& count = operation_seen[static_cast<std::size_t>(record.id.value)];
      ++count;
      if (count > 1) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.id));
        return Fail(std::move(result), started, "operation_duplicate",
                    "serialization",
                    "serialized operation appears more than once");
      }
      const StableOperationKey key =
          KeyForIdUnchecked(input.instance, record.id);
      if (record.name) {
        if (const std::optional<std::string> mismatch =
                ValidateNameMatches(input.instance, record.id, *record.name)) {
          result.offending_operations.push_back(*record.name);
          return Fail(std::move(result), started, "operation_name_malformed",
                      "serialization", *mismatch);
        }
      }
      if (record.microbatch && *record.microbatch != key.microbatch) {
        return Fail(std::move(result), started, "operation_microbatch_mismatch",
                    "serialization",
                    "serialized microbatch does not match operation id");
      }
      if (record.operation_position &&
          *record.operation_position != key.operation_position) {
        return Fail(
            std::move(result), started, "operation_position_mismatch",
            "serialization",
            "serialized operation position does not match operation id");
      }
      if (record.stage && *record.stage != key.stage) {
        return Fail(std::move(result), started, "operation_stage_mismatch",
                    "serialization",
                    "serialized stage does not match operation id");
      }
      if (record.phase && *record.phase != key.phase) {
        return Fail(std::move(result), started, "operation_phase_mismatch",
                    "serialization",
                    "serialized phase does not match operation id");
      }
      if (record.worker && *record.worker != key.worker) {
        result.offending_worker = record.worker;
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.id));
        return Fail(std::move(result), started, "operation_worker_mismatch",
                    "serialization",
                    "serialized worker does not match stage mapping");
      }
      const Tick expected_duration =
          durations[static_cast<std::size_t>(record.id.value)];
      if (record.duration && *record.duration != expected_duration) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.id));
        result.expected_tick = expected_duration;
        result.actual_tick = *record.duration;
        return Fail(std::move(result), started, "operation_duration_mismatch",
                    "serialization",
                    "serialized duration does not match partition cost model");
      }
      if (!record.start || !record.end || !record.duration) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.id));
        return Fail(std::move(result), started, "operation_timing_missing",
                    "serialization",
                    "serialized operation is missing start, end, or duration");
      }
      if (*record.start < 0 || *record.end < *record.start) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.id));
        return Fail(std::move(result), started, "operation_timing_invalid",
                    "serialization",
                    "serialized operation interval is invalid");
      }
      if (*record.end - *record.start != expected_duration) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, record.id));
        result.expected_tick = expected_duration;
        result.actual_tick = *record.end - *record.start;
        return Fail(std::move(result), started, "operation_duration_mismatch",
                    "serialization",
                    "serialized end-start duration does not match cost model");
      }
      intervals[static_cast<std::size_t>(record.id.value)] = record;
    }
    for (Index id = 0; id < expected_count; ++id) {
      if (operation_seen[static_cast<std::size_t>(id)] == 0) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, OperationId{id}));
        return FailCount(
            std::move(result), started, "operation_missing", "serialization",
            "serialized operations do not contain every operation",
            expected_count,
            static_cast<Index>(input.serialized_operations->size()));
      }
    }

    for (const ValidationEdge& edge : edges) {
      const SerializedOperationRecord& from =
          *intervals[static_cast<std::size_t>(edge.from.value)];
      const SerializedOperationRecord& to =
          *intervals[static_cast<std::size_t>(edge.to.value)];
      const Tick required_start =
          CheckedAdd(*from.end, edge.lag, "validator edge time check");
      if (*to.start < required_start) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, edge.from));
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, edge.to));
        result.offending_edge_type = edge.type;
        result.expected_tick = required_start;
        result.actual_tick = *to.start;
        const std::string code =
            edge.type == "data"   ? "data_dependency_violation"
            : edge.type == "fifo" ? "fifo_dependency_violation"
                                  : "worker_serialization_violation";
        return Fail(
            std::move(result), started, code, "schedule",
            "serialized timings violate a " + edge.type + " dependency");
      }
    }

    for (Index id = 0; id < expected_count; ++id) {
      const SerializedOperationRecord& record =
          *intervals[static_cast<std::size_t>(id)];
      if (*record.start != earliest_start[static_cast<std::size_t>(id)] ||
          *record.end != earliest_end[static_cast<std::size_t>(id)]) {
        result.offending_operations.push_back(
            OperationNameUnchecked(input.instance, OperationId{id}));
        result.expected_tick = earliest_start[static_cast<std::size_t>(id)];
        result.actual_tick = *record.start;
        return Fail(std::move(result), started, "serialized_timing_mismatch",
                    "schedule",
                    "serialized timings are not the independent earliest-start "
                    "schedule for the returned order");
      }
    }

    std::vector<std::vector<SerializedOperationRecord>> by_worker(
        static_cast<std::size_t>(input.instance.workers));
    Tick serialized_makespan = 0;
    for (const SerializedOperationRecord& record :
         *input.serialized_operations) {
      const StableOperationKey key =
          KeyForIdUnchecked(input.instance, record.id);
      by_worker[static_cast<std::size_t>(key.worker)].push_back(record);
      serialized_makespan = std::max(serialized_makespan, *record.end);
    }
    result.serialized_makespan = serialized_makespan;
    for (Index worker = 0; worker < input.instance.workers; ++worker) {
      std::vector<SerializedOperationRecord>& worker_intervals =
          by_worker[static_cast<std::size_t>(worker)];
      std::sort(worker_intervals.begin(), worker_intervals.end(),
                [](const SerializedOperationRecord& a,
                   const SerializedOperationRecord& b) {
                  if (*a.start != *b.start) return *a.start < *b.start;
                  if (*a.end != *b.end) return *a.end < *b.end;
                  return a.id.value < b.id.value;
                });
      for (std::size_t i = 1; i < worker_intervals.size(); ++i) {
        const SerializedOperationRecord& previous = worker_intervals[i - 1];
        const SerializedOperationRecord& current = worker_intervals[i];
        if (*current.start < *previous.end) {
          result.offending_worker = worker;
          result.offending_operations.push_back(
              OperationNameUnchecked(input.instance, previous.id));
          result.offending_operations.push_back(
              OperationNameUnchecked(input.instance, current.id));
          return Fail(std::move(result), started, "worker_interval_overlap",
                      "schedule",
                      "serialized operation intervals overlap on a worker");
        }
      }
    }
    if (serialized_makespan != reconstructed_makespan) {
      result.expected_tick = reconstructed_makespan;
      result.actual_tick = serialized_makespan;
      return Fail(
          std::move(result), started, "serialized_makespan_mismatch",
          "objective",
          "serialized makespan differs from independent reconstruction");
    }
  }

  if (*input.reported_makespan != reconstructed_makespan) {
    result.expected_tick = reconstructed_makespan;
    result.actual_tick = *input.reported_makespan;
    return Fail(std::move(result), started, "reported_makespan_mismatch",
                "objective",
                "reported makespan differs from independent reconstruction");
  }

  return Pass(std::move(result), started);
}

ResultValidationResult ValidateScheduleSolutionIndependent(
    const Instance& instance, const ScheduleSolution& schedule,
    const std::string& status, std::optional<std::string> method_name,
    bool fifo_ordering) {
  return ValidateResult(ValidationInputFromSchedule(
      instance, schedule, status, method_name, fifo_ordering));
}

ResultValidationInput ValidationInputFromResultJsonText(
    const std::string& json_text) {
  JsonParser parser(json_text);
  return InputFromJson(parser.Parse());
}

ScheduleSolution ScheduleSolutionFromValidationInput(
    const ResultValidationInput& input) {
  if (!input.selected_partition) {
    throw Error("validated result does not contain a selected partition");
  }
  if (!input.serialized_operations) {
    throw Error("validated result does not contain serialized operations");
  }
  ScheduleSolution schedule;
  schedule.split = *input.selected_partition;
  if (input.worker_orders) {
    schedule.orders = *input.worker_orders;
  }
  const Index expected_count = OperationCountRaw(input.instance);
  schedule.operations_by_id.resize(static_cast<std::size_t>(expected_count));
  std::vector<bool> seen(static_cast<std::size_t>(expected_count), false);
  for (const SerializedOperationRecord& record : *input.serialized_operations) {
    if (!InRangeOperation(input.instance, record.id)) {
      throw Error("serialized operation id is out of range");
    }
    if (!record.start || !record.end || !record.duration || !record.worker) {
      throw Error("serialized operation timing record is incomplete");
    }
    const std::size_t index = static_cast<std::size_t>(record.id.value);
    schedule.operations_by_id[index] =
        ScheduledOperation{record.id, *record.start, *record.end,
                           *record.duration, *record.worker};
    schedule.makespan = std::max(schedule.makespan, *record.end);
    seen[index] = true;
  }
  for (bool present : seen) {
    if (!present) {
      throw Error("serialized operations do not contain every operation");
    }
  }
  return schedule;
}

ResultValidationResult ValidateResultJsonText(const std::string& json_text) {
  const auto started = Clock::now();
  try {
    ResultValidationInput input = ValidationInputFromResultJsonText(json_text);
    return ValidateResult(input);
  } catch (const std::exception& error) {
    ResultValidationResult result;
    result.passed = false;
    result.error_code = "result_json_unreadable";
    result.error_category = "serialization";
    result.message = error.what();
    result.validation_runtime_seconds = ElapsedSeconds(started);
    return result;
  }
}

std::string ResultValidationToJson(const ResultValidationResult& result,
                                   const std::string& indent) {
  const std::string nested = indent + "  ";
  std::ostringstream out;
  out << "{\n";
  out << nested << "\"validation_version\": " << result.validation_version
      << ",\n";
  out << nested << "\"passed\": " << (result.passed ? "true" : "false")
      << ",\n";
  out << nested << "\"error_code\": ";
  WriteOptionalString(out, result.error_code.empty()
                               ? std::nullopt
                               : std::optional<std::string>(result.error_code));
  out << ",\n";
  out << nested << "\"error_category\": ";
  WriteOptionalString(out,
                      result.error_category.empty()
                          ? std::nullopt
                          : std::optional<std::string>(result.error_category));
  out << ",\n";
  out << nested << "\"message\": ";
  WriteOptionalString(out, result.message.empty()
                               ? std::nullopt
                               : std::optional<std::string>(result.message));
  out << ",\n";
  out << nested << "\"offending_operations\": [";
  for (std::size_t i = 0; i < result.offending_operations.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(result.offending_operations[i]) << "\"";
  }
  out << "],\n";
  out << nested << "\"offending_worker\": ";
  WriteOptionalNumber(out, result.offending_worker);
  out << ",\n";
  out << nested << "\"offending_edge_type\": ";
  WriteOptionalString(out, result.offending_edge_type);
  out << ",\n";
  out << nested << "\"expected_tick\": ";
  WriteOptionalNumber(out, result.expected_tick);
  out << ",\n";
  out << nested << "\"actual_tick\": ";
  WriteOptionalNumber(out, result.actual_tick);
  out << ",\n";
  out << nested << "\"expected_count\": ";
  WriteOptionalNumber(out, result.expected_count);
  out << ",\n";
  out << nested << "\"actual_count\": ";
  WriteOptionalNumber(out, result.actual_count);
  out << ",\n";
  out << nested << "\"reconstructed_makespan\": ";
  WriteOptionalNumber(out, result.reconstructed_makespan);
  out << ",\n";
  out << nested << "\"serialized_makespan\": ";
  WriteOptionalNumber(out, result.serialized_makespan);
  out << ",\n";
  out << nested << "\"reported_makespan\": ";
  WriteOptionalNumber(out, result.reported_makespan);
  out << ",\n";
  out << nested
      << "\"expected_operation_count\": " << result.expected_operation_count
      << ",\n";
  out << nested
      << "\"actual_operation_count\": " << result.actual_operation_count
      << ",\n";
  out << nested << "\"data_edge_count\": " << result.data_edge_count << ",\n";
  out << nested << "\"fifo_edge_count\": " << result.fifo_edge_count << ",\n";
  out << nested << "\"worker_edge_count\": " << result.worker_edge_count
      << ",\n";
  out << nested << "\"total_edge_count\": " << result.total_edge_count << ",\n";
  out << nested << "\"intervals_checked\": " << result.intervals_checked
      << ",\n";
  out << nested << "\"cycle_witness\": [";
  for (std::size_t i = 0; i < result.cycle_witness.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(result.cycle_witness[i]) << "\"";
  }
  out << "],\n";
  out << nested << "\"warnings\": [";
  for (std::size_t i = 0; i < result.warnings.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(result.warnings[i]) << "\"";
  }
  out << "],\n";
  out << nested
      << "\"validation_runtime_seconds\": " << result.validation_runtime_seconds
      << "\n";
  out << indent << "}";
  return out.str();
}

}  // namespace slackpipe
