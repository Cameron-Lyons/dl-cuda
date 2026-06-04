#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename ParamCodec, typename GradCodec>
__global__ void SGDUpdateKernel(typename ParamCodec::Storage *params,
                                const typename GradCodec::Storage *grads, float *momentum_buffer,
                                bool has_momentum, float lr, float momentum, float weight_decay,
                                float dampening, bool nesterov, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = ParamCodec::Load(params, idx);
    float g = GradCodec::Load(grads, idx);
    if (weight_decay != 0.0f) {
      g += weight_decay * p;
    }

    float update = g;
    if (has_momentum) {
      float buffer = momentum * momentum_buffer[idx] + (1.0f - dampening) * g;
      momentum_buffer[idx] = buffer;
      update = nesterov ? g + momentum * buffer : buffer;
    }

    ParamCodec::Store(params, idx, p - lr * update);
  }
}
template <typename ParamCodec, typename GradCodec>
Status LaunchSGDUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *momentum_buffer,
                       bool has_momentum, float lr, float momentum, float weight_decay,
                       float dampening, bool nesterov, int blocks) {
  SGDUpdateKernel<ParamCodec, GradCodec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      param.value->data_as<typename ParamCodec::Storage>(),
      param.grad->data_as<typename GradCodec::Storage>(),
      has_momentum ? momentum_buffer->data_as<float>() : nullptr, has_momentum, lr, momentum,
      weight_decay, dampening, nesterov, param.value->numel());
  return detail::CheckKernelLaunch("SGD update kernel");
}

template <typename ParamCodec>
Status LaunchSGDUpdateForParam(RuntimeContext &ctx, const ParameterRef &param,
                               Tensor *momentum_buffer, bool has_momentum, float lr, float momentum,
                               float weight_decay, float dampening, bool nesterov, int blocks) {
  switch (param.grad->dtype()) {
  case DType::kFloat32:
    return LaunchSGDUpdate<ParamCodec, detail::Float32Codec>(
        ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay, dampening, nesterov,
        blocks);
  case DType::kFloat16:
    return LaunchSGDUpdate<ParamCodec, detail::Float16Codec>(
        ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay, dampening, nesterov,
        blocks);
  case DType::kBFloat16:
    return LaunchSGDUpdate<ParamCodec, detail::BFloat16Codec>(
        ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay, dampening, nesterov,
        blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("SGD does not support grad dtype " +
                                 std::string(DTypeName(param.grad->dtype())));
}

Status LaunchSGDUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *momentum_buffer,
                       bool has_momentum, float lr, float momentum, float weight_decay,
                       float dampening, bool nesterov, int blocks) {
  switch (param.value->dtype()) {
  case DType::kFloat32:
    return LaunchSGDUpdateForParam<detail::Float32Codec>(ctx, param, momentum_buffer, has_momentum,
                                                         lr, momentum, weight_decay, dampening,
                                                         nesterov, blocks);
  case DType::kFloat16:
    return LaunchSGDUpdateForParam<detail::Float16Codec>(ctx, param, momentum_buffer, has_momentum,
                                                         lr, momentum, weight_decay, dampening,
                                                         nesterov, blocks);
  case DType::kBFloat16:
    return LaunchSGDUpdateForParam<detail::BFloat16Codec>(ctx, param, momentum_buffer, has_momentum,
                                                          lr, momentum, weight_decay, dampening,
                                                          nesterov, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("SGD does not support parameter dtype " +
                                 std::string(DTypeName(param.value->dtype())));
}

} // namespace
} // namespace dlcuda
