#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename ParamCodec, typename GradCodec>
__global__ void RMSPropUpdateKernel(typename ParamCodec::Storage *params,
                                    const typename GradCodec::Storage *grads, float *square_avg,
                                    float *momentum_buffer, float *grad_avg, bool has_momentum,
                                    bool centered, float lr, float alpha, float epsilon,
                                    float momentum, float weight_decay, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = ParamCodec::Load(params, idx);
    float g = GradCodec::Load(grads, idx);
    if (weight_decay != 0.0f) {
      g += weight_decay * p;
    }

    float square = alpha * square_avg[idx] + (1.0f - alpha) * g * g;
    square_avg[idx] = square;

    float avg = square;
    if (centered) {
      float mean = alpha * grad_avg[idx] + (1.0f - alpha) * g;
      grad_avg[idx] = mean;
      avg -= mean * mean;
    }
    float update = g / sqrtf(fmaxf(avg, 0.0f) + epsilon);
    if (has_momentum) {
      float buffer = momentum * momentum_buffer[idx] + update;
      momentum_buffer[idx] = buffer;
      update = buffer;
    }

    ParamCodec::Store(params, idx, p - lr * update);
  }
}
template <typename ParamCodec, typename GradCodec>
Status LaunchRMSPropUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *square_avg,
                           Tensor *momentum_buffer, Tensor *grad_avg, bool has_momentum,
                           bool centered, float lr, float alpha, float epsilon, float momentum,
                           float weight_decay, int blocks) {
  RMSPropUpdateKernel<ParamCodec, GradCodec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      param.value->data_as<typename ParamCodec::Storage>(),
      param.grad->data_as<typename GradCodec::Storage>(), square_avg->data_as<float>(),
      has_momentum ? momentum_buffer->data_as<float>() : nullptr,
      centered ? grad_avg->data_as<float>() : nullptr, has_momentum, centered, lr, alpha, epsilon,
      momentum, weight_decay, param.value->numel());
  return detail::CheckKernelLaunch("RMSProp update kernel");
}

struct RMSPropUpdateLauncher {
  RuntimeContext &ctx;
  const ParameterRef &param;
  Tensor *square_avg = nullptr;
  Tensor *momentum_buffer = nullptr;
  Tensor *grad_avg = nullptr;
  bool has_momentum = false;
  bool centered = false;
  float lr = 0.0f;
  float alpha = 0.0f;
  float epsilon = 0.0f;
  float momentum = 0.0f;
  float weight_decay = 0.0f;
  int blocks = 0;

  template <typename ParamCodec, typename GradCodec> Status operator()() const {
    return LaunchRMSPropUpdate<ParamCodec, GradCodec>(ctx, param, square_avg, momentum_buffer,
                                                      grad_avg, has_momentum, centered, lr, alpha,
                                                      epsilon, momentum, weight_decay, blocks);
  }
};

Status LaunchRMSPropUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *square_avg,
                           Tensor *momentum_buffer, Tensor *grad_avg, bool has_momentum,
                           bool centered, float lr, float alpha, float epsilon, float momentum,
                           float weight_decay, int blocks) {
  return DispatchOptimizerParamGradDTypes(
      param, "RMSProp",
      RMSPropUpdateLauncher{ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum,
                            centered, lr, alpha, epsilon, momentum, weight_decay, blocks});
}

} // namespace
} // namespace dlcuda
