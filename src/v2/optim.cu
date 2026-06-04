#include "dl_cuda/optim.hpp"
#include "dl_cuda/trainer.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>

namespace dlcuda {
namespace {

constexpr int kOptimizerThreads = 256;
constexpr int kNormReductionMaxBlocks = 4096;

Status ValidatePositiveFinite(float value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0f)) {
    return Status::InvalidArgument(std::string(name) + " must be finite and > 0");
  }
  return Status::Ok();
}

Status ValidateAdamHyperparameters(float beta1, float beta2, float epsilon) {
  if (!std::isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f) {
    return Status::InvalidArgument("Adam beta1 must be finite and in [0, 1)");
  }
  if (!std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f) {
    return Status::InvalidArgument("Adam beta2 must be finite and in [0, 1)");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(epsilon, "Adam epsilon"));
  return Status::Ok();
}

Status ValidateGradient(const ParameterRef &param, const char *op_name) {
  if (param.grad == nullptr || !param.grad->defined()) {
    return Status::InvalidArgument(std::string(op_name) + ": undefined grad tensor for " +
                                   param.name);
  }
  if (param.grad->dtype() != DType::kFloat32) {
    return Status::InvalidArgument(std::string(op_name) + " only supports float32 grads");
  }
  return Status::Ok();
}

Status ValidateParameterAndGradient(const ParameterRef &param, const char *op_name) {
  if (param.value == nullptr || !param.value->defined()) {
    return Status::InvalidArgument(std::string(op_name) + ": undefined parameter for " +
                                   param.name);
  }
  DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, op_name));
  if (param.value->dtype() != DType::kFloat32) {
    return Status::InvalidArgument(std::string(op_name) + " only supports float32 parameters");
  }
  if (param.value->shape() != param.grad->shape()) {
    return Status::InvalidArgument(std::string(op_name) + " shape mismatch for " + param.name);
  }
  return Status::Ok();
}

Status ZeroGradients(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, "ZeroGrad"));
    DLCUDA_RETURN_IF_ERROR(param.grad->FillZero(ctx.stream()));
  }
  return Status::Ok();
}

__global__ void AdamUpdateKernel(float *params, const float *grads, float *m, float *v, float lr,
                                 float beta1, float beta2, float epsilon,
                                 float inv_bias_correction1, float inv_bias_correction2,
                                 int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = grads[idx];
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    float m_hat = m_new * inv_bias_correction1;
    float v_hat = v_new * inv_bias_correction2;
    params[idx] -= lr * (m_hat / (sqrtf(v_hat) + epsilon));
  }
}

__global__ void AccumulateNormSqKernel(const float *grads, int64_t n, float *total_norm_sq) {
  __shared__ float shared[kOptimizerThreads];
  int tid = threadIdx.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

  float local = 0.0f;
  for (int64_t i = idx; i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float v = grads[i];
    local += v * v;
  }

  shared[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(total_norm_sq, shared[0]);
  }
}

__global__ void ComputeClipScaleKernel(const float *total_norm_sq, float max_norm,
                                       float *clip_scale) {
  float total_norm = sqrtf(total_norm_sq[0]);
  clip_scale[0] = total_norm > max_norm ? max_norm / (total_norm + 1e-6f) : 1.0f;
}

__global__ void ScaleByFactorKernel(float *data, const float *clip_scale, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= clip_scale[0];
  }
}

} // namespace

Status AdamOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  std::unordered_set<const Tensor *> active_params;
  active_params.reserve(params.size());

  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateParameterAndGradient(param, "Adam"));
    active_params.insert(param.value);

    auto m_it = m_state_.find(param.value);
    auto v_it = v_state_.find(param.value);

    bool needs_init = (m_it == m_state_.end() || v_it == v_state_.end());
    if (!needs_init) {
      needs_init =
          (m_it->second.shape() != param.value->shape() ||
           v_it->second.shape() != param.value->shape() ||
           m_it->second.dtype() != DType::kFloat32 || v_it->second.dtype() != DType::kFloat32);
    }

    if (needs_init) {
      auto m = Tensor::Allocate(param.value->shape(), DType::kFloat32);
      if (!m.ok()) {
        return m.status();
      }
      auto v = Tensor::Allocate(param.value->shape(), DType::kFloat32);
      if (!v.ok()) {
        return v.status();
      }
      m_state_[param.value] = m.value();
      v_state_[param.value] = v.value();
      DLCUDA_RETURN_IF_ERROR(m_state_[param.value].FillZero(ctx.stream()));
      DLCUDA_RETURN_IF_ERROR(v_state_[param.value].FillZero(ctx.stream()));
    }
  }

  for (auto it = m_state_.begin(); it != m_state_.end();) {
    if (active_params.find(it->first) == active_params.end()) {
      it = m_state_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = v_state_.begin(); it != v_state_.end();) {
    if (active_params.find(it->first) == active_params.end()) {
      it = v_state_.erase(it);
    } else {
      ++it;
    }
  }

  return Status::Ok();
}

Status AdamOptimizer::ZeroGrad(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  return ZeroGradients(ctx, params);
}

Status AdamOptimizer::Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float lr) {
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(lr, "Adam lr"));
  DLCUDA_RETURN_IF_ERROR(ValidateAdamHyperparameters(beta1_, beta2_, epsilon_));
  if (params.empty()) {
    return Status::Ok();
  }
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));

  beta1_power_ *= beta1_;
  beta2_power_ *= beta2_;
  float inv_bias_correction1 = 1.0f / (1.0f - beta1_power_);
  float inv_bias_correction2 = 1.0f / (1.0f - beta2_power_);

  for (const auto &param : params) {
    Tensor &m = m_state_.at(param.value);
    Tensor &v = v_state_.at(param.value);

    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      AdamUpdateKernel<<<blocks.value(), kOptimizerThreads, 0, ctx.stream()>>>(
          param.value->data_as<float>(), param.grad->data_as<float>(), m.data_as<float>(),
          v.data_as<float>(), lr, beta1_, beta2_, epsilon_, inv_bias_correction1,
          inv_bias_correction2, param.value->numel());
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Adam update kernel"));
    }
  }

  return Status::Ok();
}

Status ClipGradNorm(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float max_norm,
                    float *total_norm) {
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(max_norm, "max_norm"));

  bool has_grad_elements = false;
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, "ClipGradNorm"));
    if (param.grad->numel() > 0) {
      has_grad_elements = true;
    }
  }
  if (!has_grad_elements) {
    if (total_norm != nullptr) {
      *total_norm = 0.0f;
    }
    return Status::Ok();
  }

  auto total_norm_sq_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.total_norm_sq", {1}, DType::kFloat32);
  if (!total_norm_sq_tensor.ok()) {
    return total_norm_sq_tensor.status();
  }
  Tensor total_norm_sq_buffer = total_norm_sq_tensor.value();
  DLCUDA_RETURN_IF_ERROR(total_norm_sq_buffer.FillZero(ctx.stream()));

  auto clip_scale_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.clip_scale", {1}, DType::kFloat32);
  if (!clip_scale_tensor.ok()) {
    return clip_scale_tensor.status();
  }
  Tensor clip_scale_buffer = clip_scale_tensor.value();

  for (const auto &param : params) {
    int64_t n = param.grad->numel();
    auto blocks = detail::CappedBlocksForElements(n, kOptimizerThreads, kNormReductionMaxBlocks);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() <= 0) {
      continue;
    }
    AccumulateNormSqKernel<<<blocks.value(), kOptimizerThreads, 0, ctx.stream()>>>(
        param.grad->data_as<float>(), n, total_norm_sq_buffer.data_as<float>());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("AccumulateNormSqKernel"));
  }

  ComputeClipScaleKernel<<<1, 1, 0, ctx.stream()>>>(total_norm_sq_buffer.data_as<float>(), max_norm,
                                                    clip_scale_buffer.data_as<float>());
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ComputeClipScaleKernel"));

  for (const auto &param : params) {
    auto blocks = detail::BlocksForElements(param.grad->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() <= 0) {
      continue;
    }
    ScaleByFactorKernel<<<blocks.value(), kOptimizerThreads, 0, ctx.stream()>>>(
        param.grad->data_as<float>(), clip_scale_buffer.data_as<float>(), param.grad->numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ScaleByFactorKernel"));
  }

  if (total_norm != nullptr) {
    float total_norm_sq = 0.0f;
    DLCUDA_RETURN_IF_ERROR(
        total_norm_sq_buffer.CopyToHost(&total_norm_sq, sizeof(total_norm_sq), ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());
    *total_norm = std::sqrt(total_norm_sq);
  }

  return Status::Ok();
}

} // namespace dlcuda
