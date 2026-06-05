#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename Codec>
__global__ void NormSqPartialsKernel(const typename Codec::Storage *grads, int64_t n,
                                     float *partials, int64_t partial_offset) {
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
    partials[partial_offset + blockIdx.x] = block_sum;
  }
}

__global__ void FinalizeNormSqKernel(const float *partials, float *total_norm_sq,
                                     int64_t partial_count) {
  __shared__ typename OptimizerBlockReduce::TempStorage reduce_storage;
  int tid = threadIdx.x;

  float local = 0.0f;
  for (int64_t i = tid; i < partial_count; i += blockDim.x) {
    local += partials[i];
  }
  float total = OptimizerBlockReduce(reduce_storage).Sum(local);

  if (tid == 0) {
    total_norm_sq[0] = total;
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
Status LaunchNormSqPartials(RuntimeContext &ctx, const Tensor &grad, Tensor *partial_norm_sq_buffer,
                            int64_t partial_offset, int blocks) {
  NormSqPartialsKernel<Codec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      grad.data_as<typename Codec::Storage>(), grad.numel(),
      partial_norm_sq_buffer->data_as<float>(), partial_offset);
  return detail::CheckKernelLaunch("NormSqPartialsKernel");
}

Status LaunchNormSqPartials(RuntimeContext &ctx, const Tensor &grad, Tensor *partial_norm_sq_buffer,
                            int64_t partial_offset, int blocks) {
  switch (grad.dtype()) {
  case DType::kFloat32:
    return LaunchNormSqPartials<detail::Float32Codec>(ctx, grad, partial_norm_sq_buffer,
                                                      partial_offset, blocks);
  case DType::kFloat16:
    return LaunchNormSqPartials<detail::Float16Codec>(ctx, grad, partial_norm_sq_buffer,
                                                      partial_offset, blocks);
  case DType::kBFloat16:
    return LaunchNormSqPartials<detail::BFloat16Codec>(ctx, grad, partial_norm_sq_buffer,
                                                       partial_offset, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ClipGradNorm does not support dtype " +
                                 std::string(DTypeName(grad.dtype())));
}

Status LaunchFinalizeNormSq(RuntimeContext &ctx, const Tensor &partial_norm_sq_buffer,
                            Tensor *total_norm_sq_buffer, int64_t partial_count) {
  FinalizeNormSqKernel<<<1, kOptimizerThreads, 0, ctx.stream()>>>(
      partial_norm_sq_buffer.data_as<float>(), total_norm_sq_buffer->data_as<float>(),
      partial_count);
  return detail::CheckKernelLaunch("FinalizeNormSqKernel");
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
