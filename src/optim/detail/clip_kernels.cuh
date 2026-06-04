#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename Codec>
__global__ void AccumulateNormSqKernel(const typename Codec::Storage *grads, int64_t n,
                                       float *total_norm_sq) {
  __shared__ typename OptimizerBlockReduce::TempStorage reduce_storage;
  int tid = threadIdx.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

  float local = 0.0f;
  for (int64_t i = idx; i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float v = Codec::Load(grads, i);
    local += v * v;
  }
  float block_sum = OptimizerBlockReduce(reduce_storage).Sum(local);

  if (tid == 0) {
    atomicAdd(total_norm_sq, block_sum);
  }
}

__global__ void ComputeClipScaleKernel(const float *total_norm_sq, float max_norm,
                                       float *clip_scale) {
  float total_norm = sqrtf(total_norm_sq[0]);
  clip_scale[0] = total_norm > max_norm ? max_norm / (total_norm + 1e-6f) : 1.0f;
}

template <typename Codec>
__global__ void ScaleByFactorKernel(typename Codec::Storage *data, const float *clip_scale,
                                    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    Codec::Store(data, idx, Codec::Load(data, idx) * clip_scale[0]);
  }
}
template <typename Codec>
Status LaunchAccumulateNormSq(RuntimeContext &ctx, const Tensor &grad, Tensor *total_norm_sq_buffer,
                              int blocks) {
  AccumulateNormSqKernel<Codec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      grad.data_as<typename Codec::Storage>(), grad.numel(),
      total_norm_sq_buffer->data_as<float>());
  return detail::CheckKernelLaunch("AccumulateNormSqKernel");
}

Status LaunchAccumulateNormSq(RuntimeContext &ctx, const Tensor &grad, Tensor *total_norm_sq_buffer,
                              int blocks) {
  switch (grad.dtype()) {
  case DType::kFloat32:
    return LaunchAccumulateNormSq<detail::Float32Codec>(ctx, grad, total_norm_sq_buffer, blocks);
  case DType::kFloat16:
    return LaunchAccumulateNormSq<detail::Float16Codec>(ctx, grad, total_norm_sq_buffer, blocks);
  case DType::kBFloat16:
    return LaunchAccumulateNormSq<detail::BFloat16Codec>(ctx, grad, total_norm_sq_buffer, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ClipGradNorm does not support dtype " +
                                 std::string(DTypeName(grad.dtype())));
}

template <typename Codec>
Status LaunchScaleByFactor(RuntimeContext &ctx, Tensor *grad, Tensor *clip_scale_buffer,
                           int blocks) {
  ScaleByFactorKernel<Codec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      grad->data_as<typename Codec::Storage>(), clip_scale_buffer->data_as<float>(), grad->numel());
  return detail::CheckKernelLaunch("ScaleByFactorKernel");
}

Status LaunchScaleByFactor(RuntimeContext &ctx, Tensor *grad, Tensor *clip_scale_buffer,
                           int blocks) {
  switch (grad->dtype()) {
  case DType::kFloat32:
    return LaunchScaleByFactor<detail::Float32Codec>(ctx, grad, clip_scale_buffer, blocks);
  case DType::kFloat16:
    return LaunchScaleByFactor<detail::Float16Codec>(ctx, grad, clip_scale_buffer, blocks);
  case DType::kBFloat16:
    return LaunchScaleByFactor<detail::BFloat16Codec>(ctx, grad, clip_scale_buffer, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ClipGradNorm does not support dtype " +
                                 std::string(DTypeName(grad->dtype())));
}

} // namespace
} // namespace dlcuda
