#pragma once

#include "dl_cuda/nn.hpp"

#include "dl_cuda/detail/cuda_dtype.cuh"
#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/detail/tensor_validation.hpp"
#include "dl_cuda/tensor_ops.hpp"

#if defined(DLCUDA_HAS_CUBLASLT)
#include <cublasLt.h>
#endif
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dlcuda {
namespace {

[[maybe_unused]] constexpr int kCudaThreads = 256;
[[maybe_unused]] constexpr int kLinearTile = 16;

using CudaBlockReduce = cub::BlockReduce<float, kCudaThreads>;
using detail::EnsureDType;
using detail::EnsureSameShapeAndType;
using detail::ValidateFloatingTensor;
using detail::ValidateIntTensor;
using detail::ValidateRank;

struct FloatMaxReduce {
  __host__ __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

inline Status ValidateFloatingDType(DType dtype, const char *name) {
  if (!IsFloatingPointDType(dtype)) {
    return Status::InvalidArgument(std::string(name) + " dtype must be floating point");
  }
  return Status::Ok();
}

inline Status CopyHostFloatsToTensor(Tensor *tensor, const std::vector<float> &values,
                                     cudaStream_t stream) {
  if (tensor == nullptr || !tensor->defined()) {
    return Status::InvalidArgument("CopyHostFloatsToTensor received undefined tensor");
  }
  if (tensor->numel() != static_cast<int64_t>(values.size())) {
    return Status::InvalidArgument("CopyHostFloatsToTensor size mismatch");
  }
  switch (tensor->dtype()) {
  case DType::kFloat32:
    return tensor->CopyFromHost(values.data(), values.size() * sizeof(float), stream);
  case DType::kFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = detail::FloatToFloat16Bits(values[i]);
    }
    return tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), stream);
  }
  case DType::kBFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = detail::FloatToBFloat16Bits(values[i]);
    }
    return tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), stream);
  }
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("CopyHostFloatsToTensor requires a floating-point tensor");
}

inline Status FillKaimingNormal(RuntimeContext &ctx, Tensor *weight, float fan_in) {
  if (weight == nullptr) {
    return Status::InvalidArgument("FillKaimingNormal received a null tensor");
  }
  if (!(fan_in > 0.0f)) {
    return Status::InvalidArgument("FillKaimingNormal fan_in must be positive");
  }

  std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
  std::vector<float> host_weight(static_cast<size_t>(weight->numel()));
  for (float &v : host_weight) {
    v = dist(rng);
  }

  return CopyHostFloatsToTensor(weight, host_weight, ctx.stream());
}

inline Status InitializeWeightBiasAndGradients(RuntimeContext &ctx, Tensor *weight, Tensor *bias,
                                               Tensor *grad_weight, Tensor *grad_bias,
                                               float fan_in) {
  if (bias == nullptr || grad_weight == nullptr || grad_bias == nullptr) {
    return Status::InvalidArgument("InitializeWeightBiasAndGradients received a null tensor");
  }
  DLCUDA_RETURN_IF_ERROR(FillKaimingNormal(ctx, weight, fan_in));
  DLCUDA_RETURN_IF_ERROR(bias->FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(grad_weight->FillZero(ctx.stream()));
  return grad_bias->FillZero(ctx.stream());
}

inline Result<int64_t> SpatialOutputSize(int64_t input, int64_t kernel, int64_t stride,
                                         int64_t padding, const char *name) {
  if (input < 0) {
    return Status::InvalidArgument(std::string(name) + " input size must be non-negative");
  }
  if (kernel <= 0 || stride <= 0 || padding < 0) {
    return Status::InvalidArgument(std::string(name) +
                                   " kernel/stride/padding parameters are invalid");
  }
  int64_t numerator = input + 2 * padding - kernel;
  if (numerator < 0) {
    return Status::InvalidArgument(std::string(name) + " output size is non-positive");
  }
  return numerator / stride + 1;
}

inline std::string JoinParameterName(const std::string &prefix, const char *name) {
  if (prefix.empty()) {
    return std::string(name);
  }
  return prefix + "." + name;
}

} // namespace
} // namespace dlcuda
