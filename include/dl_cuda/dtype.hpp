#pragma once

#include <cstddef>
#include <cstdint>

namespace dlcuda {

enum class DType : uint32_t {
  kFloat32 = 0,
  kInt32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
};

[[nodiscard]] inline constexpr size_t DTypeSize(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return sizeof(float);
  case DType::kInt32:
    return sizeof(int32_t);
  case DType::kFloat16:
  case DType::kBFloat16:
    return 2;
  }
  return 0;
}

[[nodiscard]] inline constexpr const char *DTypeName(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return "float32";
  case DType::kInt32:
    return "int32";
  case DType::kFloat16:
    return "float16";
  case DType::kBFloat16:
    return "bfloat16";
  }
  return "unknown";
}

[[nodiscard]] inline constexpr bool IsFloatingPointDType(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
  case DType::kFloat16:
  case DType::kBFloat16:
    return true;
  case DType::kInt32:
    return false;
  }
  return false;
}

} // namespace dlcuda
