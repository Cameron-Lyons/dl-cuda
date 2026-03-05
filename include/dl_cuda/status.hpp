#pragma once

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

class Status {
public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }
  static Status InvalidArgument(const std::string &message) {
    return Status(StatusCode::kInvalidArgument, message);
  }
  static Status RuntimeError(const std::string &message) {
    return Status(StatusCode::kRuntimeError, message);
  }
  static Status IoError(const std::string &message) {
    return Status(StatusCode::kIoError, message);
  }
  static Status NotFound(const std::string &message) {
    return Status(StatusCode::kNotFound, message);
  }
  static Status Unsupported(const std::string &message) {
    return Status(StatusCode::kUnsupported, message);
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string &message() const { return message_; }

private:
  StatusCode code_;
  std::string message_;
};

template <typename T> class Result {
public:
  Result(const T &value) : value_(value), status_(Status::Ok()) {}
  Result(T &&value) : value_(std::move(value)), status_(Status::Ok()) {}
  Result(const Status &status) : status_(status) {}

  bool ok() const { return status_.ok(); }
  const Status &status() const { return status_; }

  const T &value() const { return value_; }
  T &value() { return value_; }

private:
  T value_{};
  Status status_;
};

#define DLCUDA_RETURN_IF_ERROR(expr)                                           \
  do {                                                                         \
    ::dlcuda::Status _status = (expr);                                         \
    if (!_status.ok()) {                                                       \
      return _status;                                                          \
    }                                                                          \
  } while (0)

} // namespace dlcuda
