#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

__device__ uint32_t DropoutHash(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<uint32_t>(value >> 32);
}

template <typename Codec>
__global__ void DropoutForwardKernel(const typename Codec::Storage *input,
                                     typename Codec::Storage *output, float *mask, int64_t size,
                                     float probability, uint64_t seed) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float keep_probability = 1.0f - probability;
    float uniform = static_cast<float>(DropoutHash(seed ^ static_cast<uint64_t>(idx)) >> 8) *
                    (1.0f / 16777216.0f);
    float multiplier = uniform < keep_probability ? 1.0f / keep_probability : 0.0f;
    mask[idx] = multiplier;
    Codec::Store(output, idx, Codec::Load(input, idx) * multiplier);
  }
}

template <typename Codec>
__global__ void DropoutBackwardKernel(const typename Codec::Storage *grad_output, const float *mask,
                                      typename Codec::Storage *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    Codec::Store(grad_input, idx, Codec::Load(grad_output, idx) * mask[idx]);
  }
}

template <typename Codec>
__global__ void TensorCopyKernel(const typename Codec::Storage *input,
                                 typename Codec::Storage *output, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    Codec::Store(output, idx, Codec::Load(input, idx));
  }
}

template <typename Codec>
Status LaunchTensorCopyKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                              int blocks) {
  TensorCopyKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      input.numel());
  return detail::CheckKernelLaunch("Tensor copy kernel");
}

Status LaunchTensorCopyKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                              int blocks) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchTensorCopyKernel<detail::Float32Codec>(ctx, input, output, blocks);
  case DType::kFloat16:
    return LaunchTensorCopyKernel<detail::Float16Codec>(ctx, input, output, blocks);
  case DType::kBFloat16:
    return LaunchTensorCopyKernel<detail::BFloat16Codec>(ctx, input, output, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Tensor copy does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchDropoutForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  Tensor *mask, int blocks, float probability, uint64_t seed) {
  DropoutForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      mask->data_as<float>(), input.numel(), probability, seed);
  return detail::CheckKernelLaunch("Dropout forward kernel");
}

Status LaunchDropoutForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  Tensor *mask, int blocks, float probability, uint64_t seed) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchDropoutForwardKernel<detail::Float32Codec>(ctx, input, output, mask, blocks,
                                                            probability, seed);
  case DType::kFloat16:
    return LaunchDropoutForwardKernel<detail::Float16Codec>(ctx, input, output, mask, blocks,
                                                            probability, seed);
  case DType::kBFloat16:
    return LaunchDropoutForwardKernel<detail::BFloat16Codec>(ctx, input, output, mask, blocks,
                                                             probability, seed);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Dropout does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchDropoutBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &mask, Tensor *grad_input, int blocks) {
  DropoutBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), mask.data_as<float>(),
      grad_input->data_as<typename Codec::Storage>(), grad_output.numel());
  return detail::CheckKernelLaunch("Dropout backward kernel");
}

Status LaunchDropoutBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &mask, Tensor *grad_input, int blocks) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchDropoutBackwardKernel<detail::Float32Codec>(ctx, grad_output, mask, grad_input,
                                                             blocks);
  case DType::kFloat16:
    return LaunchDropoutBackwardKernel<detail::Float16Codec>(ctx, grad_output, mask, grad_input,
                                                             blocks);
  case DType::kBFloat16:
    return LaunchDropoutBackwardKernel<detail::BFloat16Codec>(ctx, grad_output, mask, grad_input,
                                                              blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Dropout backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
}

} // namespace
} // namespace dlcuda
