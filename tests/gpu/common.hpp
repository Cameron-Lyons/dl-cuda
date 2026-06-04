#pragma once

#include "dl_cuda.hpp"
#include "dl_cuda/detail/cuda_dtype.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

namespace dlcuda::gpu_tests {

inline bool HasCudaDevice() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  return status == cudaSuccess && count > 0;
}

inline bool AlmostEqual(float actual, float expected, float tolerance = 1e-4f) {
  return std::fabs(actual - expected) <= tolerance;
}

inline float GELUValue(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

inline float GELUGrad(float x) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
  return 0.5f * (1.0f + std::erf(x * kInvSqrt2)) + x * std::exp(-0.5f * x * x) * kInvSqrt2Pi;
}

inline bool CheckCloseVector(const std::vector<float> &actual, const std::vector<float> &expected,
                             const char *label, float tolerance = 1e-4f) {
  if (actual.size() != expected.size()) {
    std::fprintf(stderr, "%s size mismatch\n", label);
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!AlmostEqual(actual[i], expected[i], tolerance)) {
      std::fprintf(stderr, "%s value mismatch at %zu: got %.6f expected %.6f\n", label, i,
                   actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

inline bool CopyFloatsToTensor(dlcuda::RuntimeContext &ctx, dlcuda::Tensor *tensor,
                               const std::vector<float> &values, const char *label) {
  if (tensor == nullptr || !tensor->defined() ||
      tensor->numel() != static_cast<int64_t>(values.size())) {
    std::fprintf(stderr, "%s copy received invalid tensor\n", label);
    return false;
  }
  dlcuda::Status status;
  switch (tensor->dtype()) {
  case dlcuda::DType::kFloat32:
    status = tensor->CopyFromHost(values.data(), values.size() * sizeof(float), ctx.stream());
    break;
  case dlcuda::DType::kFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = dlcuda::detail::FloatToFloat16Bits(values[i]);
    }
    status =
        tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), ctx.stream());
    break;
  }
  case dlcuda::DType::kBFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = dlcuda::detail::FloatToBFloat16Bits(values[i]);
    }
    status =
        tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), ctx.stream());
    break;
  }
  case dlcuda::DType::kInt32:
    std::fprintf(stderr, "%s copy does not support int32\n", label);
    return false;
  }
  if (!status.ok()) {
    std::fprintf(stderr, "%s copy failed: %s\n", label, status.ToString().c_str());
    return false;
  }
  return true;
}

inline bool CopyTensorToFloats(dlcuda::RuntimeContext &ctx, const dlcuda::Tensor &tensor,
                               std::vector<float> *values, const char *label) {
  if (values == nullptr || !tensor.defined()) {
    std::fprintf(stderr, "%s read received invalid tensor\n", label);
    return false;
  }
  values->resize(static_cast<size_t>(tensor.numel()));
  dlcuda::Status status;
  switch (tensor.dtype()) {
  case dlcuda::DType::kFloat32:
    status = tensor.CopyToHost(values->data(), values->size() * sizeof(float), ctx.stream());
    break;
  case dlcuda::DType::kFloat16: {
    std::vector<uint16_t> raw(values->size());
    status = tensor.CopyToHost(raw.data(), raw.size() * sizeof(uint16_t), ctx.stream());
    if (status.ok()) {
      for (size_t i = 0; i < raw.size(); ++i) {
        (*values)[i] = dlcuda::detail::Float16BitsToFloat(raw[i]);
      }
    }
    break;
  }
  case dlcuda::DType::kBFloat16: {
    std::vector<uint16_t> raw(values->size());
    status = tensor.CopyToHost(raw.data(), raw.size() * sizeof(uint16_t), ctx.stream());
    if (status.ok()) {
      for (size_t i = 0; i < raw.size(); ++i) {
        (*values)[i] = dlcuda::detail::BFloat16BitsToFloat(raw[i]);
      }
    }
    break;
  }
  case dlcuda::DType::kInt32:
    std::fprintf(stderr, "%s read does not support int32\n", label);
    return false;
  }
  if (!status.ok() || !ctx.Synchronize().ok()) {
    std::fprintf(stderr, "%s read failed\n", label);
    return false;
  }
  return true;
}

bool RunMixedPrecisionSmoke(dlcuda::RuntimeContext &ctx, dlcuda::DType dtype);
bool RunAutogradSmoke(dlcuda::RuntimeContext &ctx);
bool RunLayerCoverageSmoke(dlcuda::RuntimeContext &ctx);
bool RunOptimizerCoverageSmoke(dlcuda::RuntimeContext &ctx);

} // namespace dlcuda::gpu_tests
