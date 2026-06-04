#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <utility>

namespace dlcuda {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kRuntimeError,
  kIoError,
  kNotFound,
  kUnsupported,
};

[[nodiscard]] inline const char *StatusCodeName(StatusCode code) {
  switch (code) {
  case StatusCode::kOk:
    return "Ok";
  case StatusCode::kInvalidArgument:
    return "InvalidArgument";
  case StatusCode::kRuntimeError:
    return "RuntimeError";
  case StatusCode::kIoError:
    return "IoError";
  case StatusCode::kNotFound:
    return "NotFound";
  case StatusCode::kUnsupported:
    return "Unsupported";
  }
  return "Unknown";
}

class [[nodiscard]] Status {
public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}
  Status(const Status &) = default;
  Status(Status &&) noexcept = default;

  void operator=(const Status &other) {
    code_ = other.code_;
    message_ = other.message_;
  }

  void operator=(Status &&other) noexcept {
    code_ = other.code_;
    message_ = std::move(other.message_);
  }

  [[nodiscard]] static Status Ok() {
    return Status();
  }
  [[nodiscard]] static Status InvalidArgument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }
  [[nodiscard]] static Status RuntimeError(std::string message) {
    return Status(StatusCode::kRuntimeError, std::move(message));
  }
  [[nodiscard]] static Status IoError(std::string message) {
    return Status(StatusCode::kIoError, std::move(message));
  }
  [[nodiscard]] static Status NotFound(std::string message) {
    return Status(StatusCode::kNotFound, std::move(message));
  }
  [[nodiscard]] static Status Unsupported(std::string message) {
    return Status(StatusCode::kUnsupported, std::move(message));
  }

  [[nodiscard]] bool ok() const {
    return code_ == StatusCode::kOk;
  }
  [[nodiscard]] StatusCode code() const {
    return code_;
  }
  [[nodiscard]] const std::string &message() const {
    return message_;
  }
  [[nodiscard]] const char *code_name() const {
    return StatusCodeName(code_);
  }
  [[nodiscard]] std::string ToString() const {
    if (message_.empty()) {
      return code_name();
    }
    return std::string(code_name()) + ": " + message_;
  }

private:
  StatusCode code_;
  std::string message_;
};

template <typename T> class [[nodiscard]] Result {
public:
  Result(const T &value) : value_(value), status_(Status::Ok()) {}
  Result(T &&value) : value_(std::move(value)), status_(Status::Ok()) {}
  Result(const Status &status) : status_(status) {
    assert(!status.ok());
  }
  Result(Status &&status) : status_(std::move(status)) {
    assert(!status_.ok());
  }

  [[nodiscard]] bool ok() const {
    return status_.ok();
  }
  [[nodiscard]] const Status &status() const {
    return status_;
  }

  [[nodiscard]] const T &value() const {
    assert(ok() && value_.has_value());
    return *value_;
  }
  [[nodiscard]] T &value() {
    assert(ok() && value_.has_value());
    return *value_;
  }

private:
  std::optional<T> value_;
  Status status_;
};

#define DLCUDA_RETURN_IF_ERROR(expr)                                                               \
  do {                                                                                             \
    ::dlcuda::Status _status = (expr);                                                             \
    if (!_status.ok()) {                                                                           \
      return _status;                                                                              \
    }                                                                                              \
  } while (0)

} // namespace dlcuda
