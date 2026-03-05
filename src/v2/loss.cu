#include "dl_cuda/loss.hpp"

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

Status ValidateFloat2D(const Tensor &tensor, const char *name) {
  if (!tensor.defined()) {
    return Status::InvalidArgument(std::string(name) + " is undefined");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Status::InvalidArgument(std::string(name) + " must be float32");
  }
  if (tensor.rank() != 2) {
    return Status::InvalidArgument(std::string(name) + " must have rank 2");
  }
  return Status::Ok();
}

__global__ void BinaryCrossEntropyKernel(const float *targets,
                                         const float *predictions,
                                         float *loss_sum, float *grads,
                                         int64_t n, float epsilon) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float y = targets[idx];
    float p = predictions[idx];
    p = fmaxf(epsilon, fminf(1.0f - epsilon, p));

    float loss = -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    atomicAdd(loss_sum, loss);

    grads[idx] = (-y / p + (1.0f - y) / (1.0f - p)) / static_cast<float>(n);
  }
}

__global__ void CategoricalMetricsKernel(const int32_t *target_ids,
                                         const float *probabilities,
                                         float *loss_sum,
                                         float *correct_sum, int64_t n,
                                         int64_t classes, float epsilon) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < n) {
    int32_t target = target_ids[row];
    const float *row_probs = probabilities + row * classes;

    if (target < 0 || static_cast<int64_t>(target) >= classes) {
      return;
    }

    float p = row_probs[target];
    p = fmaxf(epsilon, p);
    atomicAdd(loss_sum, -logf(p));

    int best = 0;
    float best_p = row_probs[0];
    for (int64_t c = 1; c < classes; ++c) {
      float v = row_probs[c];
      if (v > best_p) {
        best = static_cast<int>(c);
        best_p = v;
      }
    }
    if (best == static_cast<int>(target)) {
      atomicAdd(correct_sum, 1.0f);
    }
  }
}

__global__ void CategoricalGradKernel(const int32_t *target_ids,
                                      const float *probabilities, float *grads,
                                      int64_t n, int64_t classes,
                                      float epsilon) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = n * classes;
  if (idx < total) {
    int64_t row = idx / classes;
    int64_t col = idx % classes;
    int32_t target = target_ids[row];
    if (target < 0 || static_cast<int64_t>(target) >= classes) {
      grads[idx] = 0.0f;
      return;
    }
    if (col == static_cast<int64_t>(target)) {
      float p = probabilities[idx];
      p = fmaxf(epsilon, p);
      grads[idx] = -1.0f / (p * static_cast<float>(n));
    } else {
      grads[idx] = 0.0f;
    }
  }
}

} // namespace

Result<float> BinaryCrossEntropy(RuntimeContext &ctx, const Tensor &targets,
                                 const Tensor &predictions,
                                 Tensor *prediction_grads) {
  if (prediction_grads == nullptr) {
    return Status::InvalidArgument("BinaryCrossEntropy: prediction_grads is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(targets, "targets"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(predictions, "predictions"));

  if (targets.shape() != predictions.shape()) {
    return Status::InvalidArgument("BinaryCrossEntropy: targets/predictions shape mismatch");
  }

  auto grads = Tensor::Allocate(predictions.shape(), DType::kFloat32);
  if (!grads.ok()) {
    return grads.status();
  }

  float *d_loss_sum = nullptr;
  cudaError_t alloc_status = cudaMalloc(&d_loss_sum, sizeof(float));
  if (alloc_status != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMalloc failed: ") +
                                cudaGetErrorString(alloc_status));
  }

  cudaError_t memset_status = cudaMemsetAsync(d_loss_sum, 0, sizeof(float), ctx.stream());
  if (memset_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    return Status::RuntimeError(std::string("cudaMemsetAsync failed: ") +
                                cudaGetErrorString(memset_status));
  }

  int64_t n = targets.numel();
  int blocks = static_cast<int>((n + 255) / 256);
  BinaryCrossEntropyKernel<<<blocks, 256, 0, ctx.stream()>>>(
      targets.data_as<float>(), predictions.data_as<float>(), d_loss_sum,
      grads.value().data_as<float>(), n, 1e-8f);

  cudaError_t kernel_status = cudaGetLastError();
  if (kernel_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    return Status::RuntimeError(std::string("BinaryCrossEntropy kernel failed: ") +
                                cudaGetErrorString(kernel_status));
  }

  float loss_sum = 0.0f;
  cudaError_t copy_status = cudaMemcpyAsync(&loss_sum, d_loss_sum, sizeof(float),
                                            cudaMemcpyDeviceToHost, ctx.stream());
  if (copy_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    return Status::RuntimeError(std::string("cudaMemcpyAsync failed: ") +
                                cudaGetErrorString(copy_status));
  }

  cudaError_t sync_status = cudaStreamSynchronize(ctx.stream());
  if (sync_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(sync_status));
  }

  cudaFree(d_loss_sum);

  *prediction_grads = grads.value();
  return loss_sum / static_cast<float>(n);
}

Result<ClassificationMetrics>
CategoricalCrossEntropyFromIds(RuntimeContext &ctx, const Tensor &target_ids,
                               const Tensor &probabilities,
                               Tensor *probability_grads) {
  if (probability_grads == nullptr) {
    return Status::InvalidArgument(
        "CategoricalCrossEntropyFromIds: probability_grads is null");
  }
  if (!target_ids.defined()) {
    return Status::InvalidArgument("target_ids is undefined");
  }
  if (target_ids.dtype() != DType::kInt32) {
    return Status::InvalidArgument("target_ids must be int32");
  }
  if (target_ids.rank() != 1) {
    return Status::InvalidArgument("target_ids must have rank 1");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(probabilities, "probabilities"));

  int64_t n = target_ids.dim(0);
  int64_t classes = probabilities.dim(1);
  if (probabilities.dim(0) != n) {
    return Status::InvalidArgument("target_ids/probabilities batch mismatch");
  }

  auto grads = Tensor::Allocate(probabilities.shape(), DType::kFloat32);
  if (!grads.ok()) {
    return grads.status();
  }

  float *d_loss_sum = nullptr;
  float *d_correct_sum = nullptr;

  cudaError_t alloc_loss_status = cudaMalloc(&d_loss_sum, sizeof(float));
  if (alloc_loss_status != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMalloc loss_sum failed: ") +
                                cudaGetErrorString(alloc_loss_status));
  }
  cudaError_t alloc_correct_status = cudaMalloc(&d_correct_sum, sizeof(float));
  if (alloc_correct_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    return Status::RuntimeError(std::string("cudaMalloc correct_sum failed: ") +
                                cudaGetErrorString(alloc_correct_status));
  }

  cudaError_t memset_loss_status =
      cudaMemsetAsync(d_loss_sum, 0, sizeof(float), ctx.stream());
  if (memset_loss_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(std::string("cudaMemsetAsync loss_sum failed: ") +
                                cudaGetErrorString(memset_loss_status));
  }
  cudaError_t memset_correct_status =
      cudaMemsetAsync(d_correct_sum, 0, sizeof(float), ctx.stream());
  if (memset_correct_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(
        std::string("cudaMemsetAsync correct_sum failed: ") +
        cudaGetErrorString(memset_correct_status));
  }

  int metric_blocks = static_cast<int>((n + 255) / 256);
  CategoricalMetricsKernel<<<metric_blocks, 256, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), probabilities.data_as<float>(), d_loss_sum,
      d_correct_sum, n, classes, 1e-8f);
  cudaError_t metrics_status = cudaGetLastError();
  if (metrics_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(std::string("Categorical metrics kernel failed: ") +
                                cudaGetErrorString(metrics_status));
  }

  int64_t total = n * classes;
  int grad_blocks = static_cast<int>((total + 255) / 256);
  CategoricalGradKernel<<<grad_blocks, 256, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), probabilities.data_as<float>(),
      grads.value().data_as<float>(), n, classes, 1e-8f);
  cudaError_t grad_status = cudaGetLastError();
  if (grad_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(std::string("Categorical grad kernel failed: ") +
                                cudaGetErrorString(grad_status));
  }

  float loss_sum = 0.0f;
  float correct_sum = 0.0f;

  cudaError_t loss_copy_status = cudaMemcpyAsync(&loss_sum, d_loss_sum, sizeof(float),
                                                 cudaMemcpyDeviceToHost,
                                                 ctx.stream());
  if (loss_copy_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(std::string("loss_sum copy failed: ") +
                                cudaGetErrorString(loss_copy_status));
  }

  cudaError_t correct_copy_status =
      cudaMemcpyAsync(&correct_sum, d_correct_sum, sizeof(float),
                      cudaMemcpyDeviceToHost, ctx.stream());
  if (correct_copy_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(std::string("correct_sum copy failed: ") +
                                cudaGetErrorString(correct_copy_status));
  }

  cudaError_t sync_status = cudaStreamSynchronize(ctx.stream());
  if (sync_status != cudaSuccess) {
    cudaFree(d_loss_sum);
    cudaFree(d_correct_sum);
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(sync_status));
  }

  cudaFree(d_loss_sum);
  cudaFree(d_correct_sum);

  ClassificationMetrics metrics;
  metrics.loss = loss_sum / static_cast<float>(n);
  metrics.accuracy = correct_sum / static_cast<float>(n);

  *probability_grads = grads.value();
  return metrics;
}

} // namespace dlcuda
