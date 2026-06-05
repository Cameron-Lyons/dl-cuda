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

struct SGDUpdateLauncher {
  RuntimeContext &ctx;
  const ParameterRef &param;
  Tensor *momentum_buffer = nullptr;
  bool has_momentum = false;
  float lr = 0.0f;
  float momentum = 0.0f;
  float weight_decay = 0.0f;
  float dampening = 0.0f;
  bool nesterov = false;
  int blocks = 0;

  template <typename ParamCodec, typename GradCodec> Status operator()() const {
    return LaunchSGDUpdate<ParamCodec, GradCodec>(ctx, param, momentum_buffer, has_momentum, lr,
                                                  momentum, weight_decay, dampening, nesterov,
                                                  blocks);
  }
};

Status LaunchSGDUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *momentum_buffer,
                       bool has_momentum, float lr, float momentum, float weight_decay,
                       float dampening, bool nesterov, int blocks) {
  return DispatchOptimizerParamGradDTypes(
      param, "SGD",
      SGDUpdateLauncher{ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay,
                        dampening, nesterov, blocks});
}

} // namespace
} // namespace dlcuda
