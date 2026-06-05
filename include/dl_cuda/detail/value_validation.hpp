#pragma once

#include "dl_cuda/status.hpp"

#include <cmath>
#include <string>

namespace dlcuda::detail {

inline Status ValidatePositiveFinite(float value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0f)) {
    return Status::InvalidArgument(std::string(name) + " must be finite and > 0");
  }
  return Status::Ok();
}

inline Status ValidateNonNegativeFinite(float value, const char *name) {
  if (!std::isfinite(value) || value < 0.0f) {
    return Status::InvalidArgument(std::string(name) + " must be finite and >= 0");
  }
  return Status::Ok();
}

inline Status ValidateRate(float value, const char *name) {
  if (!std::isfinite(value) || value < 0.0f || value >= 1.0f) {
    return Status::InvalidArgument(std::string(name) + " must be finite and in [0, 1)");
  }
  return Status::Ok();
}

} // namespace dlcuda::detail
