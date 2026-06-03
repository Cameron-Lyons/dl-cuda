#include "dl_cuda/loss.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace dlcuda {
namespace {

constexpr int kLossThreads = 256;
constexpr int kLossReductionMaxBlocks = 4096;

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

Status ValidateTargetIds1D(const Tensor &target_ids) {
  if (!target_ids.defined()) {
    return Status::InvalidArgument("target_ids is undefined");
  }
  if (target_ids.dtype() != DType::kInt32) {
    return Status::InvalidArgument("target_ids must be int32");
  }
  if (target_ids.rank() != 1) {
    return Status::InvalidArgument("target_ids must have rank 1");
  }
  return Status::Ok();
}

struct CategoricalShape {
  int64_t rows = 0;
  int64_t classes = 0;
};

Result<CategoricalShape> ValidateCategoricalInputs(const Tensor &target_ids, const Tensor &values,
                                                   const char *values_name, const char *op_name) {
  DLCUDA_RETURN_IF_ERROR(ValidateTargetIds1D(target_ids));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(values, values_name));

  CategoricalShape shape;
  shape.rows = target_ids.dim(0);
  shape.classes = values.dim(1);
  if (values.dim(0) != shape.rows) {
    return Status::InvalidArgument(std::string("target_ids/") + values_name + " batch mismatch");
  }
  if (shape.rows <= 0 || shape.classes <= 0) {
    return Status::InvalidArgument(std::string(op_name) + " requires non-empty inputs");
  }
  if (shape.classes > std::numeric_limits<int>::max()) {
    return Status::InvalidArgument(std::string(op_name) + " class count is outside int range");
  }
  return shape;
}

__global__ void BinaryCrossEntropyKernel(const float *targets, const float *predictions,
                                         float *grads, int64_t n, float epsilon) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float y = targets[idx];
    float p = predictions[idx];
    p = fmaxf(epsilon, fminf(1.0f - epsilon, p));
    grads[idx] = (-y / p + (1.0f - y) / (1.0f - p)) / static_cast<float>(n);
  }
}

__global__ void BinaryCrossEntropyMetricKernel(const float *targets, const float *predictions,
                                               float *loss_sum, int64_t n, float epsilon) {
  __shared__ float shared[kLossThreads];

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int tid = threadIdx.x;
  float local_loss = 0.0f;
  for (int64_t i = idx; i < n; i += stride) {
    float y = targets[i];
    float p = predictions[i];
    p = fmaxf(epsilon, fminf(1.0f - epsilon, p));
    local_loss += -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
  }

  shared[tid] = local_loss;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(loss_sum, shared[0]);
  }
}

__global__ void CategoricalMetricsKernel(const int32_t *target_ids, const float *probabilities,
                                         float *loss_sum, float *correct_sum, int64_t n,
                                         int64_t classes, float epsilon) {
  __shared__ float shared_values[kLossThreads];
  __shared__ int shared_indices[kLossThreads];

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= n) {
    return;
  }

  int32_t target = target_ids[row];
  const float *row_probs = probabilities + row * classes;

  if (target < 0 || static_cast<int64_t>(target) >= classes) {
    return;
  }

  float local_best_p = -FLT_MAX;
  int local_best = 0;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    float v = row_probs[c];
    if (v > local_best_p) {
      local_best = static_cast<int>(c);
      local_best_p = v;
    }
  }
  shared_values[tid] = local_best_p;
  shared_indices[tid] = local_best;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other_value = shared_values[tid + stride];
      int other_index = shared_indices[tid + stride];
      if (other_value > shared_values[tid] ||
          (other_value == shared_values[tid] && other_index < shared_indices[tid])) {
        shared_values[tid] = other_value;
        shared_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float p = row_probs[target];
    p = fmaxf(epsilon, p);
    atomicAdd(loss_sum, -logf(p));
    if (shared_indices[0] == static_cast<int>(target)) {
      atomicAdd(correct_sum, 1.0f);
    }
  }
}

__global__ void CategoricalGradKernel(const int32_t *target_ids, const float *probabilities,
                                      float *grads, int64_t n, int64_t classes, float epsilon) {
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

__global__ void CategoricalLogitsGradKernel(const int32_t *target_ids, const float *logits,
                                            float *grads, int64_t n, int64_t classes) {
  __shared__ float shared[kLossThreads];

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= n) {
    return;
  }

  int32_t target = target_ids[row];
  const float *row_logits = logits + row * classes;
  float *row_grads = grads + row * classes;

  if (target < 0 || static_cast<int64_t>(target) >= classes) {
    for (int64_t c = tid; c < classes; c += blockDim.x) {
      row_grads[c] = 0.0f;
    }
    return;
  }

  float local_max = -FLT_MAX;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    local_max = fmaxf(local_max, row_logits[c]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  float row_max = shared[0];

  float local_sum = 0.0f;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    float e = expf(row_logits[c] - row_max);
    row_grads[c] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  float inv_sum = 1.0f / (shared[0] + 1e-20f);
  float inv_n = 1.0f / static_cast<float>(n);
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    float p = row_grads[c] * inv_sum;
    row_grads[c] = (p - (c == static_cast<int64_t>(target) ? 1.0f : 0.0f)) * inv_n;
  }
}

__global__ void CategoricalLogitsMetricsKernel(const int32_t *target_ids, const float *logits,
                                               float *loss_sum, float *correct_sum, int64_t n,
                                               int64_t classes) {
  __shared__ float shared_values[kLossThreads];
  __shared__ int shared_indices[kLossThreads];

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= n) {
    return;
  }

  int32_t target = target_ids[row];
  const float *row_logits = logits + row * classes;

  if (target < 0 || static_cast<int64_t>(target) >= classes) {
    return;
  }

  float local_max = -FLT_MAX;
  int local_best = 0;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    float v = row_logits[c];
    if (v > local_max) {
      local_max = v;
      local_best = static_cast<int>(c);
    }
  }
  shared_values[tid] = local_max;
  shared_indices[tid] = local_best;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other_value = shared_values[tid + stride];
      int other_index = shared_indices[tid + stride];
      if (other_value > shared_values[tid] ||
          (other_value == shared_values[tid] && other_index < shared_indices[tid])) {
        shared_values[tid] = other_value;
        shared_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }
  float row_max = shared_values[0];
  int best = shared_indices[0];

  float local_sum = 0.0f;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    local_sum += expf(row_logits[c] - row_max);
  }
  shared_values[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_values[tid] += shared_values[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float target_logit = row_logits[target];
    float sum = shared_values[0];
    atomicAdd(loss_sum, logf(sum + 1e-20f) + row_max - target_logit);
    if (best == static_cast<int>(target)) {
      atomicAdd(correct_sum, 1.0f);
    }
  }
}

} // namespace

Status BinaryCrossEntropyBackward(RuntimeContext &ctx, const Tensor &targets,
                                  const Tensor &predictions, Tensor *prediction_grads) {
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
  DLCUDA_RETURN_IF_ERROR(EnsureTensor(prediction_grads, predictions.shape(), DType::kFloat32));

  auto blocks = detail::BlocksForElements(n, kLossThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  BinaryCrossEntropyKernel<<<blocks.value(), kLossThreads, 0, ctx.stream()>>>(
      targets.data_as<float>(), predictions.data_as<float>(), prediction_grads->data_as<float>(), n,
      1e-8f);

  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("BinaryCrossEntropy kernel"));
  return Status::Ok();
}

Result<float> BinaryCrossEntropyLoss(RuntimeContext &ctx, const Tensor &targets,
                                     const Tensor &predictions) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(targets, "targets"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat2D(predictions, "predictions"));
  if (targets.shape() != predictions.shape()) {
    return Status::InvalidArgument("BinaryCrossEntropyLoss: targets/predictions shape mismatch");
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

  auto blocks = detail::CappedBlocksForElements(n, kLossThreads, kLossReductionMaxBlocks);
  if (!blocks.ok()) {
    return blocks.status();
  }
  BinaryCrossEntropyMetricKernel<<<blocks.value(), kLossThreads, 0, ctx.stream()>>>(
      targets.data_as<float>(), predictions.data_as<float>(), loss_sum_buffer.data_as<float>(), n,
      1e-8f);
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("BinaryCrossEntropy metric kernel"));

  float loss_sum = 0.0f;
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.CopyToHost(&loss_sum, sizeof(loss_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());
  return loss_sum / static_cast<float>(n);
}

Status CategoricalCrossEntropyBackwardFromIds(RuntimeContext &ctx, const Tensor &target_ids,
                                              const Tensor &probabilities,
                                              Tensor *probability_grads) {
  if (probability_grads == nullptr) {
    return Status::InvalidArgument(
        "CategoricalCrossEntropyBackwardFromIds: probability_grads is null");
  }
  auto shape = ValidateCategoricalInputs(target_ids, probabilities, "probabilities",
                                         "CategoricalCrossEntropyBackwardFromIds");
  if (!shape.ok()) {
    return shape.status();
  }
  CategoricalShape categorical = shape.value();
  int64_t n = categorical.rows;
  int64_t classes = categorical.classes;

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(probability_grads, probabilities.shape(), DType::kFloat32));

  int64_t total = n * classes;
  auto grad_blocks = detail::BlocksForElements(total, kLossThreads);
  if (!grad_blocks.ok()) {
    return grad_blocks.status();
  }
  CategoricalGradKernel<<<grad_blocks.value(), kLossThreads, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), probabilities.data_as<float>(),
      probability_grads->data_as<float>(), n, classes, 1e-8f);
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Categorical grad kernel"));
  return Status::Ok();
}

Status CategoricalCrossEntropyBackwardFromLogits(RuntimeContext &ctx, const Tensor &target_ids,
                                                 const Tensor &logits, Tensor *logit_grads) {
  if (logit_grads == nullptr) {
    return Status::InvalidArgument(
        "CategoricalCrossEntropyBackwardFromLogits: logit_grads is null");
  }
  auto shape = ValidateCategoricalInputs(target_ids, logits, "logits",
                                         "CategoricalCrossEntropyBackwardFromLogits");
  if (!shape.ok()) {
    return shape.status();
  }
  CategoricalShape categorical = shape.value();
  int64_t n = categorical.rows;
  int64_t classes = categorical.classes;

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(logit_grads, logits.shape(), DType::kFloat32));

  auto rows = detail::RowsForGrid(n, "categorical logits");
  if (!rows.ok()) {
    return rows.status();
  }
  CategoricalLogitsGradKernel<<<rows.value(), kLossThreads, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), logits.data_as<float>(), logit_grads->data_as<float>(), n,
      classes);
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Categorical logits grad kernel"));
  return Status::Ok();
}

Result<ClassificationMetrics> CategoricalCrossEntropyMetricsFromLogits(RuntimeContext &ctx,
                                                                       const Tensor &target_ids,
                                                                       const Tensor &logits) {
  auto shape = ValidateCategoricalInputs(target_ids, logits, "logits",
                                         "CategoricalCrossEntropyMetricsFromLogits");
  if (!shape.ok()) {
    return shape.status();
  }
  CategoricalShape categorical = shape.value();
  int64_t n = categorical.rows;
  int64_t classes = categorical.classes;

  auto loss_sum_tensor =
      ctx.ScratchTensor("loss.categorical_cross_entropy_logits.loss_sum", {1}, DType::kFloat32);
  if (!loss_sum_tensor.ok()) {
    return loss_sum_tensor.status();
  }
  auto correct_sum_tensor =
      ctx.ScratchTensor("loss.categorical_cross_entropy_logits.correct_sum", {1}, DType::kFloat32);
  if (!correct_sum_tensor.ok()) {
    return correct_sum_tensor.status();
  }

  Tensor loss_sum_buffer = loss_sum_tensor.value();
  Tensor correct_sum_buffer = correct_sum_tensor.value();
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(correct_sum_buffer.FillZero(ctx.stream()));

  auto rows = detail::RowsForGrid(n, "categorical logits");
  if (!rows.ok()) {
    return rows.status();
  }
  CategoricalLogitsMetricsKernel<<<rows.value(), kLossThreads, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), logits.data_as<float>(), loss_sum_buffer.data_as<float>(),
      correct_sum_buffer.data_as<float>(), n, classes);
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Categorical logits metrics kernel"));

  float loss_sum = 0.0f;
  float correct_sum = 0.0f;
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.CopyToHost(&loss_sum, sizeof(loss_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(
      correct_sum_buffer.CopyToHost(&correct_sum, sizeof(correct_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

  ClassificationMetrics metrics;
  metrics.loss = loss_sum / static_cast<float>(n);
  metrics.accuracy = correct_sum / static_cast<float>(n);
  return metrics;
}

Result<ClassificationMetrics> CategoricalCrossEntropyMetricsFromIds(RuntimeContext &ctx,
                                                                    const Tensor &target_ids,
                                                                    const Tensor &probabilities) {
  auto shape = ValidateCategoricalInputs(target_ids, probabilities, "probabilities",
                                         "CategoricalCrossEntropyMetricsFromIds");
  if (!shape.ok()) {
    return shape.status();
  }
  CategoricalShape categorical = shape.value();
  int64_t n = categorical.rows;
  int64_t classes = categorical.classes;

  auto loss_sum_tensor =
      ctx.ScratchTensor("loss.categorical_cross_entropy.loss_sum", {1}, DType::kFloat32);
  if (!loss_sum_tensor.ok()) {
    return loss_sum_tensor.status();
  }
  auto correct_sum_tensor =
      ctx.ScratchTensor("loss.categorical_cross_entropy.correct_sum", {1}, DType::kFloat32);
  if (!correct_sum_tensor.ok()) {
    return correct_sum_tensor.status();
  }

  Tensor loss_sum_buffer = loss_sum_tensor.value();
  Tensor correct_sum_buffer = correct_sum_tensor.value();
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(correct_sum_buffer.FillZero(ctx.stream()));

  auto rows = detail::RowsForGrid(n, "categorical probabilities");
  if (!rows.ok()) {
    return rows.status();
  }
  CategoricalMetricsKernel<<<rows.value(), kLossThreads, 0, ctx.stream()>>>(
      target_ids.data_as<int32_t>(), probabilities.data_as<float>(),
      loss_sum_buffer.data_as<float>(), correct_sum_buffer.data_as<float>(), n, classes, 1e-8f);
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Categorical metrics kernel"));

  float loss_sum = 0.0f;
  float correct_sum = 0.0f;
  DLCUDA_RETURN_IF_ERROR(loss_sum_buffer.CopyToHost(&loss_sum, sizeof(loss_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(
      correct_sum_buffer.CopyToHost(&correct_sum, sizeof(correct_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

  ClassificationMetrics metrics;
  metrics.loss = loss_sum / static_cast<float>(n);
  metrics.accuracy = correct_sum / static_cast<float>(n);
  return metrics;
}

} // namespace dlcuda
