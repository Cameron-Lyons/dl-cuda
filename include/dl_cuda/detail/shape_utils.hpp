#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dlcuda::detail {

inline std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= shape[i - 1];
  }
  return strides;
}

} // namespace dlcuda::detail
