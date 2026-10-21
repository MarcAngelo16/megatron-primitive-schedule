#pragma once

#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace slackpipe {

class Deadline {
 public:
  using Clock = std::chrono::steady_clock;

  explicit Deadline(double requested_seconds = 0.0);

  [[nodiscard]] static Deadline FromSeconds(double requested_seconds) {
    return Deadline(requested_seconds);
  }

  [[nodiscard]] bool bounded() const { return bounded_; }
  [[nodiscard]] double requested_seconds() const { return requested_seconds_; }
  [[nodiscard]] Clock::time_point started_at() const { return started_; }
  [[nodiscard]] double elapsed_seconds() const;
  [[nodiscard]] double remaining_seconds() const;
  [[nodiscard]] double remaining_seconds_for_reporting() const;
  [[nodiscard]] bool expired() const;
  [[nodiscard]] double clamp_solver_limit(double requested_phase_limit) const;

 private:
  Clock::time_point started_;
  double requested_seconds_ = 0.0;
  bool bounded_ = false;
};

struct DeadlinePhaseSnapshot {
  std::string phase;
  std::optional<double> requested_limit_seconds;
  std::optional<double> effective_limit_seconds;
  std::optional<double> remaining_before_seconds;
  std::optional<double> remaining_after_seconds;
  double runtime_seconds = 0.0;
  bool expired_before_start = false;
  std::string status;
};

class DeadlinePhaseTimer {
 public:
  DeadlinePhaseTimer(const Deadline& deadline, std::string phase,
                     std::optional<double> requested_limit_seconds);

  [[nodiscard]] double effective_limit_seconds() const {
    return effective_limit_seconds_.value_or(0.0);
  }
  [[nodiscard]] bool expired_before_start() const {
    return expired_before_start_;
  }
  [[nodiscard]] DeadlinePhaseSnapshot Finish(
      const std::string& status = "") const;

 private:
  const Deadline& deadline_;
  std::string phase_;
  std::optional<double> requested_limit_seconds_;
  std::optional<double> effective_limit_seconds_;
  std::optional<double> remaining_before_seconds_;
  Deadline::Clock::time_point started_;
  bool expired_before_start_ = false;
};

}  // namespace slackpipe
