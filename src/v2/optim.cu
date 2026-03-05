#include "dl_cuda/optim.hpp"
#include "dl_cuda/trainer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace dlcuda {
namespace {

Status FromCuda(cudaError_t err, const std::string &context) {
  if (err == cudaSuccess) {
    return Status::Ok();
  }
  return Status::RuntimeError(context + ": " + cudaGetErrorString(err));
}

__global__ void SgdUpdateKernel(float *params, const float *grads, float lr,
                                int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    params[idx] -= lr * grads[idx];
  }
}

__global__ void AdamUpdateKernel(float *params, const float *grads, float *m,
                                 float *v, float lr, float beta1,
                                 float beta2, float epsilon,
                                 float inv_bias_correction1,
                                 float inv_bias_correction2, int64_t n) {
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

__global__ void AccumulateNormSqKernel(const float *grads, int64_t n,
                                       float *total_norm_sq) {
  __shared__ float shared[256];
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

__global__ void ScaleKernel(float *data, float scale, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

} // namespace

Status SGDOptimizer::ZeroGrad(RuntimeContext &ctx,
                              const std::vector<ParameterRef> &params) {
  for (const auto &param : params) {
    if (param.grad == nullptr || !param.grad->defined()) {
      return Status::InvalidArgument("ZeroGrad: undefined grad tensor for " +
                                     param.name);
    }
    DLCUDA_RETURN_IF_ERROR(param.grad->FillZero(ctx.stream()));
  }
  return Status::Ok();
}

Status SGDOptimizer::Step(RuntimeContext &ctx,
                          const std::vector<ParameterRef> &params, float lr) {
  if (!(lr > 0.0f)) {
    return Status::InvalidArgument("SGD lr must be > 0");
  }
  for (const auto &param : params) {
    if (param.value == nullptr || param.grad == nullptr || !param.value->defined() ||
        !param.grad->defined()) {
      return Status::InvalidArgument("SGD step: undefined parameter or gradient for " +
                                     param.name);
    }
    if (param.value->dtype() != DType::kFloat32 ||
        param.grad->dtype() != DType::kFloat32) {
      return Status::InvalidArgument("SGD supports float32 parameters only");
    }
    if (param.value->shape() != param.grad->shape()) {
      return Status::InvalidArgument("SGD shape mismatch for " + param.name);
    }

    int blocks = static_cast<int>((param.value->numel() + 255) / 256);
    SgdUpdateKernel<<<blocks, 256, 0, ctx.stream()>>>(
        param.value->data_as<float>(), param.grad->data_as<float>(), lr,
        param.value->numel());
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "SGD update kernel"));
  }
  return Status::Ok();
}

Status AdamOptimizer::EnsureState(RuntimeContext &ctx,
                                  const std::vector<ParameterRef> &params) {
  for (const auto &param : params) {
    if (param.value == nullptr || !param.value->defined() ||
        param.grad == nullptr || !param.grad->defined()) {
      return Status::InvalidArgument("Adam state requires defined value/grad for " +
                                     param.name);
    }
    if (param.value->dtype() != DType::kFloat32 ||
        param.grad->dtype() != DType::kFloat32) {
      return Status::InvalidArgument("Adam supports float32 tensors only");
    }
    if (param.value->shape() != param.grad->shape()) {
      return Status::InvalidArgument("Adam shape mismatch for " + param.name);
    }

    auto m_it = m_state_.find(param.name);
    auto v_it = v_state_.find(param.name);

    bool needs_init = (m_it == m_state_.end() || v_it == v_state_.end());
    if (!needs_init) {
      needs_init = (m_it->second.shape() != param.value->shape());
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
      m_state_[param.name] = m.value();
      v_state_[param.name] = v.value();
      DLCUDA_RETURN_IF_ERROR(m_state_[param.name].FillZero(ctx.stream()));
      DLCUDA_RETURN_IF_ERROR(v_state_[param.name].FillZero(ctx.stream()));
    }
  }

  return Status::Ok();
}

Status AdamOptimizer::ZeroGrad(RuntimeContext &ctx,
                               const std::vector<ParameterRef> &params) {
  for (const auto &param : params) {
    if (param.grad == nullptr || !param.grad->defined()) {
      return Status::InvalidArgument("ZeroGrad: undefined grad tensor for " +
                                     param.name);
    }
    DLCUDA_RETURN_IF_ERROR(param.grad->FillZero(ctx.stream()));
  }
  return Status::Ok();
}

Status AdamOptimizer::Step(RuntimeContext &ctx,
                           const std::vector<ParameterRef> &params, float lr) {
  if (!(lr > 0.0f)) {
    return Status::InvalidArgument("Adam lr must be > 0");
  }
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));

  ++step_;
  float inv_bias_correction1 =
      1.0f / (1.0f - std::pow(beta1_, static_cast<float>(step_)));
  float inv_bias_correction2 =
      1.0f / (1.0f - std::pow(beta2_, static_cast<float>(step_)));

  for (const auto &param : params) {
    Tensor &m = m_state_[param.name];
    Tensor &v = v_state_[param.name];

    int blocks = static_cast<int>((param.value->numel() + 255) / 256);
    AdamUpdateKernel<<<blocks, 256, 0, ctx.stream()>>>(
        param.value->data_as<float>(), param.grad->data_as<float>(),
        m.data_as<float>(), v.data_as<float>(), lr, beta1_, beta2_, epsilon_,
        inv_bias_correction1, inv_bias_correction2, param.value->numel());
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Adam update kernel"));
  }

  return Status::Ok();
}

Result<float> ClipGradNorm(RuntimeContext &ctx,
                           const std::vector<ParameterRef> &params,
                           float max_norm) {
  if (!(max_norm > 0.0f)) {
    return Status::InvalidArgument("max_norm must be > 0");
  }

  float *d_total_norm_sq = nullptr;
  cudaError_t alloc_status = cudaMalloc(&d_total_norm_sq, sizeof(float));
  if (alloc_status != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMalloc failed in ClipGradNorm: ") +
                                cudaGetErrorString(alloc_status));
  }
  cudaError_t memset_status =
      cudaMemsetAsync(d_total_norm_sq, 0, sizeof(float), ctx.stream());
  if (memset_status != cudaSuccess) {
    cudaFree(d_total_norm_sq);
    return Status::RuntimeError(std::string("cudaMemsetAsync failed in ClipGradNorm: ") +
                                cudaGetErrorString(memset_status));
  }

  for (const auto &param : params) {
    if (param.grad == nullptr || !param.grad->defined()) {
      cudaFree(d_total_norm_sq);
      return Status::InvalidArgument("ClipGradNorm: undefined grad tensor for " +
                                     param.name);
    }
    if (param.grad->dtype() != DType::kFloat32) {
      cudaFree(d_total_norm_sq);
      return Status::InvalidArgument("ClipGradNorm only supports float32 grads");
    }
    int64_t n = param.grad->numel();
    int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 4096));
    if (blocks <= 0) {
      continue;
    }
    AccumulateNormSqKernel<<<blocks, 256, 0, ctx.stream()>>>(
        param.grad->data_as<float>(), n, d_total_norm_sq);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFree(d_total_norm_sq);
      return Status::RuntimeError(std::string("AccumulateNormSqKernel failed: ") +
                                  cudaGetErrorString(err));
    }
  }

  float total_norm_sq = 0.0f;
  cudaError_t copy_status = cudaMemcpyAsync(&total_norm_sq, d_total_norm_sq,
                                            sizeof(float), cudaMemcpyDeviceToHost,
                                            ctx.stream());
  if (copy_status != cudaSuccess) {
    cudaFree(d_total_norm_sq);
    return Status::RuntimeError(std::string("cudaMemcpyAsync failed in ClipGradNorm: ") +
                                cudaGetErrorString(copy_status));
  }
  cudaError_t sync_status = cudaStreamSynchronize(ctx.stream());
  if (sync_status != cudaSuccess) {
    cudaFree(d_total_norm_sq);
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(sync_status));
  }

  cudaFree(d_total_norm_sq);

  float total_norm = std::sqrt(total_norm_sq);
  if (total_norm > max_norm) {
    float scale = max_norm / (total_norm + 1e-6f);
    for (const auto &param : params) {
      int blocks = static_cast<int>((param.grad->numel() + 255) / 256);
      ScaleKernel<<<blocks, 256, 0, ctx.stream()>>>(param.grad->data_as<float>(),
                                                     scale,
                                                     param.grad->numel());
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        return Status::RuntimeError(std::string("ScaleKernel failed: ") +
                                    cudaGetErrorString(err));
      }
    }
  }

  return total_norm;
}

} // namespace dlcuda
