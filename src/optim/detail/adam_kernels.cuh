#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename ParamCodec, typename GradCodec>
__global__ void AdamUpdateKernel(typename ParamCodec::Storage *params,
                                 const typename GradCodec::Storage *grads, float *m, float *v,
                                 float lr, float beta1, float beta2, float epsilon,
                                 float inv_bias_correction1, float inv_bias_correction2,
                                 float weight_decay, bool decoupled_weight_decay, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = ParamCodec::Load(params, idx);
    float g = GradCodec::Load(grads, idx);
    if (weight_decay != 0.0f && !decoupled_weight_decay) {
      g += weight_decay * p;
    }

    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    float m_hat = m_new * inv_bias_correction1;
    float v_hat = v_new * inv_bias_correction2;
    float updated = p - lr * (m_hat / (sqrtf(v_hat) + epsilon));
    if (weight_decay != 0.0f && decoupled_weight_decay) {
      updated -= lr * weight_decay * p;
    }
    ParamCodec::Store(params, idx, updated);
  }
}
template <typename ParamCodec, typename GradCodec>
Status LaunchAdamUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *m, Tensor *v,
                        float lr, float beta1, float beta2, float epsilon,
                        float inv_bias_correction1, float inv_bias_correction2, float weight_decay,
                        bool decoupled_weight_decay, int blocks) {
  AdamUpdateKernel<ParamCodec, GradCodec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      param.value->data_as<typename ParamCodec::Storage>(),
      param.grad->data_as<typename GradCodec::Storage>(), m->data_as<float>(), v->data_as<float>(),
      lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2, weight_decay,
      decoupled_weight_decay, param.value->numel());
  return detail::CheckKernelLaunch("Adam update kernel");
}

template <typename ParamCodec>
Status LaunchAdamUpdateForParam(RuntimeContext &ctx, const ParameterRef &param, Tensor *m,
                                Tensor *v, float lr, float beta1, float beta2, float epsilon,
                                float inv_bias_correction1, float inv_bias_correction2,
                                float weight_decay, bool decoupled_weight_decay, int blocks) {
  switch (param.grad->dtype()) {
  case DType::kFloat32:
    return LaunchAdamUpdate<ParamCodec, detail::Float32Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kFloat16:
    return LaunchAdamUpdate<ParamCodec, detail::Float16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kBFloat16:
    return LaunchAdamUpdate<ParamCodec, detail::BFloat16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Adam does not support grad dtype " +
                                 std::string(DTypeName(param.grad->dtype())));
}

Status LaunchAdamUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *m, Tensor *v,
                        float lr, float beta1, float beta2, float epsilon,
                        float inv_bias_correction1, float inv_bias_correction2, float weight_decay,
                        bool decoupled_weight_decay, int blocks) {
  switch (param.value->dtype()) {
  case DType::kFloat32:
    return LaunchAdamUpdateForParam<detail::Float32Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kFloat16:
    return LaunchAdamUpdateForParam<detail::Float16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kBFloat16:
    return LaunchAdamUpdateForParam<detail::BFloat16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Adam does not support parameter dtype " +
                                 std::string(DTypeName(param.value->dtype())));
}

} // namespace
} // namespace dlcuda
