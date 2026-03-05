#include "dl_cuda/loss.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace dlcuda {
namespace {

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
                                         const float *predictions, float *grads,
                                         int64_t n, float epsilon) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float y = targets[idx];
    float p = predictions[idx];
    p = fmaxf(epsilon, fminf(1.0f - epsilon, p));
    grads[idx] = (-y / p + (1.0f - y) / (1.0f - p)) / static_cast<float>(n);
  }
}

__global__ void BinaryCrossEntropyMetricKernel(const float *targets,
                                               const float *predictions,
                                               float *loss_sum, int64_t n,
                                               float epsilon) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float y = targets[idx];
    float p = predictions[idx];
    p = fmaxf(epsilon, fminf(1.0f - epsilon, p));
    float loss = -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    atomicAdd(loss_sum, loss);
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

Status BinaryCrossEntropyBackward(RuntimeContext &ctx, const Tensor &targets,
                                  const Tensor &predictions,
                                  Tensor *prediction_grads) {
  if (prediction_grads == nullptr) {
    return Status::InvalidArgument("BinaryCrossEntropyBackward: prediction_grads is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(targets, "targets"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(predictions, "predictions"));

  if (targets.shape() != predictions.shape()) {
    return Status::InvalidArgument(
        "BinaryCrossEntropyBackward: targets/predictions shape mismatch");
  }

  int64_t n = targets.numel();
  if (n <= 0) {
    return Status::InvalidArgument("BinaryCrossEntropyBackward requires non-empty tensors");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(prediction_grads, predictions.shape(), DType::kFloat32));

  int blocks = static_cast<int>((n + 255) / 256);
  BinaryCrossEntropyKernel<<<blocks, 256, 0, ctx.stream()>>>(
      targets.data_as<float>(), predictions.data_as<float>(),
      prediction_grads->data_as<float>(), n, 1e-8f);

  cudaError_t kernel_status = cudaGetLastError();
  if (kernel_status != cudaSuccess) {
    return Status::RuntimeError(std::string("BinaryCrossEntropy kernel failed: ") +
                                cudaGetErrorString(kernel_status));
  }
  return Status::Ok();
}

Result<float> BinaryCrossEntropyLoss(RuntimeContext &ctx, const Tensor &targets,
                                     const Tensor &predictions) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(targets, "targets"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(predictions, "predictions"));
  if (targets.shape() != predictions.shape()) {
    return Status::InvalidArgument(
        "BinaryCrossEntropyLoss: targets/predictions shape mismatch");
  }

  int64_t n = targets.numel();
  if (n <= 0) {
    return Status::InvalidArgument("BinaryCrossEntropyLoss requires non-empty tensors");
  }

  auto loss_sum_tensor =
      ctx.ScratchTensor("loss.binary_cross_entropy.loss_sum", {1}, DType::kFloat32);
  if (!loss_sum_tensor.ok()) {
    return loss_sum_tensor.status();
  }
  Tensor loss_sum_buffer = loss_sum_tensor.value();
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.FillZero(ctx.stream()));

  int blocks = static_cast<int>((n + 255) / 256);
  BinaryCrossEntropyMetricKernel<<<blocks, 256, 0, ctx.stream()>>>(
      targets.data_as<float>(), predictions.data_as<float>(),
      loss_sum_buffer.data_as<float>(), n, 1e-8f);
  cudaError_t kernel_status = cudaGetLastError();
  if (kernel_status != cudaSuccess) {
    return Status::RuntimeError(std::string("BinaryCrossEntropy metric kernel failed: ") +
                                cudaGetErrorString(kernel_status));
  }

  float loss_sum = 0.0f;
  DLCUDA_RETURN_IF_ERROR(
      loss_sum_buffer.CopyToHost(&loss_sum, sizeof(loss_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());
  return loss_sum / static_cast<float>(n);
}

Status CategoricalCrossEntropyBackwardFromIds(RuntimeContext &ctx,
                                              const Tensor &target_ids,
                                              const Tensor &probabilities,
                                              Tensor *probability_grads) {
  if (probability_grads == nullptr) {
    return Status::InvalidArgument(
        "CategoricalCrossEntropyBackwardFromIds: probability_grads is null");
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
  if (n <= 0 || classes <= 0) {
    return Status::InvalidArgument(
        "CategoricalCrossEntropyBackwardFromIds requires non-empty inputs");
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(probability_grads, probabilities.shape(), DType::kFloat32));

  int64_t total = n * classes;
  int grad_blocks = static_cast<int>((total + 255) / 256);
  CategoricalGradKernel<<<grad_blocks, 256, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), probabilities.data_as<float>(),
      probability_grads->data_as<float>(), n, classes, 1e-8f);
  cudaError_t grad_status = cudaGetLastError();
  if (grad_status != cudaSuccess) {
    return Status::RuntimeError(std::string("Categorical grad kernel failed: ") +
                                cudaGetErrorString(grad_status));
  }
  return Status::Ok();
}

Result<ClassificationMetrics>
CategoricalCrossEntropyMetricsFromIds(RuntimeContext &ctx, const Tensor &target_ids,
                                      const Tensor &probabilities) {
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
  if (n <= 0 || classes <= 0) {
    return Status::InvalidArgument(
        "CategoricalCrossEntropyMetricsFromIds requires non-empty inputs");
  }

  auto loss_sum_tensor =
      ctx.ScratchTensor("loss.categorical_cross_entropy.loss_sum", {1},
                        DType::kFloat32);
  if (!loss_sum_tensor.ok()) {
    return loss_sum_tensor.status();
  }
  auto correct_sum_tensor =
      ctx.ScratchTensor("loss.categorical_cross_entropy.correct_sum", {1},
                        DType::kFloat32);
  if (!correct_sum_tensor.ok()) {
    return correct_sum_tensor.status();
  }

  Tensor loss_sum_buffer = loss_sum_tensor.value();
  Tensor correct_sum_buffer = correct_sum_tensor.value();
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(correct_sum_buffer.FillZero(ctx.stream()));

  int metric_blocks = static_cast<int>((n + 255) / 256);
  CategoricalMetricsKernel<<<metric_blocks, 256, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), probabilities.data_as<float>(),
      loss_sum_buffer.data_as<float>(), correct_sum_buffer.data_as<float>(), n,
      classes, 1e-8f);
  cudaError_t metrics_status = cudaGetLastError();
  if (metrics_status != cudaSuccess) {
    return Status::RuntimeError(std::string("Categorical metrics kernel failed: ") +
                                cudaGetErrorString(metrics_status));
  }

  float loss_sum = 0.0f;
  float correct_sum = 0.0f;
  DLCUDA_RETURN_IF_ERROR(
      loss_sum_buffer.CopyToHost(&loss_sum, sizeof(loss_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(
      correct_sum_buffer.CopyToHost(&correct_sum, sizeof(correct_sum),
                                    ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

  ClassificationMetrics metrics;
  metrics.loss = loss_sum / static_cast<float>(n);
  metrics.accuracy = correct_sum / static_cast<float>(n);
  return metrics;
}

} // namespace dlcuda
