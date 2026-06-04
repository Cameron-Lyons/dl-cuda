#include "dl_cuda/loss.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace dlcuda {
namespace {

constexpr int kLossThreads = 256;
constexpr int kLossReductionMaxBlocks = 4096;

using LossBlockReduce = cub::BlockReduce<float, kLossThreads>;

struct BestClass {
  float value;
  int index;
};

struct BestClassReduce {
  __device__ __forceinline__ BestClass operator()(const BestClass &a, const BestClass &b) const {
    if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
      return b;
    }
    return a;
  }
};

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
  __shared__ typename LossBlockReduce::TempStorage reduce_storage;

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
  float block_loss = LossBlockReduce(reduce_storage).Sum(local_loss);

  if (tid == 0) {
    atomicAdd(loss_sum, block_loss);
  }
}

__global__ void CategoricalLogitsGradKernel(const int32_t *target_ids, const float *logits,
                                            float *grads, int64_t n, int64_t classes) {
  __shared__ typename LossBlockReduce::TempStorage max_storage;
  __shared__ typename LossBlockReduce::TempStorage sum_storage;
  __shared__ float row_max_shared;
  __shared__ float row_sum_shared;

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
  float row_max = LossBlockReduce(max_storage).Reduce(local_max, cub::Max());
  if (tid == 0) {
    row_max_shared = row_max;
  }
  __syncthreads();
  row_max = row_max_shared;

  float local_sum = 0.0f;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    float e = expf(row_logits[c] - row_max);
    row_grads[c] = e;
    local_sum += e;
  }
  float row_sum = LossBlockReduce(sum_storage).Sum(local_sum);
  if (tid == 0) {
    row_sum_shared = row_sum;
  }
  __syncthreads();
  row_sum = row_sum_shared;

  float inv_sum = 1.0f / (row_sum + 1e-20f);
  float inv_n = 1.0f / static_cast<float>(n);
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    float p = row_grads[c] * inv_sum;
    row_grads[c] = (p - (c == static_cast<int64_t>(target) ? 1.0f : 0.0f)) * inv_n;
  }
}

__global__ void CategoricalLogitsMetricsKernel(const int32_t *target_ids, const float *logits,
                                               float *loss_sum, float *correct_sum, int64_t n,
                                               int64_t classes) {
  using BestClassBlockReduce = cub::BlockReduce<BestClass, kLossThreads>;

  __shared__ typename BestClassBlockReduce::TempStorage best_storage;
  __shared__ typename LossBlockReduce::TempStorage sum_storage;
  __shared__ BestClass best_shared;
  __shared__ float row_sum_shared;

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
  BestClass local = {local_max, local_best};
  BestClass block_best = BestClassBlockReduce(best_storage).Reduce(local, BestClassReduce{});
  if (tid == 0) {
    best_shared = block_best;
  }
  __syncthreads();
  block_best = best_shared;
  float row_max = block_best.value;
  int best = block_best.index;

  float local_sum = 0.0f;
  for (int64_t c = tid; c < classes; c += blockDim.x) {
    local_sum += expf(row_logits[c] - row_max);
  }
  float row_sum = LossBlockReduce(sum_storage).Sum(local_sum);
  if (tid == 0) {
    row_sum_shared = row_sum;
  }
  __syncthreads();
  row_sum = row_sum_shared;

  if (tid == 0) {
    float target_logit = row_logits[target];
    atomicAdd(loss_sum, logf(row_sum + 1e-20f) + row_max - target_logit);
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
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(prediction_grads, predictions.shape(), DType::kFloat32, ctx.stream()));

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

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(logit_grads, logits.shape(), DType::kFloat32, ctx.stream()));

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

} // namespace dlcuda
