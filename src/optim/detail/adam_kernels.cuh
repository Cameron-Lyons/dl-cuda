#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

struct AdamUpdateBlock {
  void *params = nullptr;
  const void *grads = nullptr;
  float *m = nullptr;
  float *v = nullptr;
  int64_t start = 0;
  int64_t n = 0;
  DType param_dtype = DType::kFloat32;
  DType grad_dtype = DType::kFloat32;
  float lr = 0.0f;
  float weight_decay = 0.0f;
};

__device__ float LoadAdamValue(const void *data, DType dtype, int64_t index) {
  switch (dtype) {
  case DType::kFloat32:
    return static_cast<const float *>(data)[index];
  case DType::kFloat16:
    return detail::Float16BitsToFloat(static_cast<const uint16_t *>(data)[index]);
  case DType::kBFloat16:
    return detail::BFloat16BitsToFloat(static_cast<const uint16_t *>(data)[index]);
  case DType::kInt32:
    break;
  }
  return 0.0f;
}

__device__ void StoreAdamParam(void *data, DType dtype, int64_t index, float value) {
  switch (dtype) {
  case DType::kFloat32:
    static_cast<float *>(data)[index] = value;
    return;
  case DType::kFloat16:
    static_cast<uint16_t *>(data)[index] = detail::FloatToFloat16Bits(value);
    return;
  case DType::kBFloat16:
    static_cast<uint16_t *>(data)[index] = detail::FloatToBFloat16Bits(value);
    return;
  case DType::kInt32:
    break;
  }
}

__global__ void AdamUpdateBlocksKernel(const AdamUpdateBlock *blocks, int block_count, float beta1,
                                       float beta2, float epsilon, float inv_bias_correction1,
                                       float inv_bias_correction2, bool decoupled_weight_decay) {
  int block_id = static_cast<int>(blockIdx.x);
  if (block_id >= block_count) {
    return;
  }

  AdamUpdateBlock block = blocks[block_id];
  int64_t idx = block.start + threadIdx.x;
  if (idx >= block.n) {
    return;
  }

  float p = LoadAdamValue(block.params, block.param_dtype, idx);
  float g = LoadAdamValue(block.grads, block.grad_dtype, idx);
  if (block.weight_decay != 0.0f && !decoupled_weight_decay) {
    g += block.weight_decay * p;
  }

  float m_new = beta1 * block.m[idx] + (1.0f - beta1) * g;
  float v_new = beta2 * block.v[idx] + (1.0f - beta2) * g * g;
  block.m[idx] = m_new;
  block.v[idx] = v_new;

  float m_hat = m_new * inv_bias_correction1;
  float v_hat = v_new * inv_bias_correction2;
  float updated = p - block.lr * (m_hat / (sqrtf(v_hat) + epsilon));
  if (block.weight_decay != 0.0f && decoupled_weight_decay) {
    updated -= block.lr * block.weight_decay * p;
  }
  StoreAdamParam(block.params, block.param_dtype, idx, updated);
}

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

struct AdamUpdateLauncher {
  RuntimeContext &ctx;
  const ParameterRef &param;
  Tensor *m = nullptr;
  Tensor *v = nullptr;
  float lr = 0.0f;
  float beta1 = 0.0f;
  float beta2 = 0.0f;
  float epsilon = 0.0f;
  float inv_bias_correction1 = 0.0f;
  float inv_bias_correction2 = 0.0f;
  float weight_decay = 0.0f;
  bool decoupled_weight_decay = false;
  int blocks = 0;

  template <typename ParamCodec, typename GradCodec> Status operator()() const {
    return LaunchAdamUpdate<ParamCodec, GradCodec>(ctx, param, m, v, lr, beta1, beta2, epsilon,
                                                   inv_bias_correction1, inv_bias_correction2,
                                                   weight_decay, decoupled_weight_decay, blocks);
  }
};

Status LaunchAdamUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *m, Tensor *v,
                        float lr, float beta1, float beta2, float epsilon,
                        float inv_bias_correction1, float inv_bias_correction2, float weight_decay,
                        bool decoupled_weight_decay, int blocks) {
  return DispatchOptimizerParamGradDTypes(
      param, "Adam",
      AdamUpdateLauncher{ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1,
                         inv_bias_correction2, weight_decay, decoupled_weight_decay, blocks});
}

Status LaunchAdamUpdateBlocks(RuntimeContext &ctx, Tensor *blocks, int block_count, float beta1,
                              float beta2, float epsilon, float inv_bias_correction1,
                              float inv_bias_correction2, bool decoupled_weight_decay) {
  AdamUpdateBlocksKernel<<<block_count, kOptimizerThreads, 0, ctx.stream()>>>(
      blocks->data_as<AdamUpdateBlock>(), block_count, beta1, beta2, epsilon, inv_bias_correction1,
      inv_bias_correction2, decoupled_weight_decay);
  return detail::CheckKernelLaunch("Adam multi-tensor update kernel");
}

} // namespace
} // namespace dlcuda
