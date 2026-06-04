#pragma once

#include "char_model.cuh"

namespace dlcuda {
namespace {

__global__ void FillTrainingWindowKernel(const int32_t *encoded_corpus, int32_t *input_ids,
                                         int32_t *target_ids, int64_t seq_len, int64_t offset) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < seq_len) {
    input_ids[idx] = encoded_corpus[offset + idx];
    target_ids[idx] = encoded_corpus[offset + idx + 1];
  }
}

inline Status FillTrainingWindow(RuntimeContext &ctx, const Tensor &encoded_corpus_device,
                                 Tensor *input_ids, Tensor *target_ids, int seq_len,
                                 int64_t offset) {
  if (input_ids == nullptr || target_ids == nullptr) {
    return Status::InvalidArgument("FillTrainingWindow received null tensor pointer");
  }
  auto window_blocks = detail::BlocksForElements(seq_len, kExampleThreads);
  if (!window_blocks.ok()) {
    return window_blocks.status();
  }
  if (window_blocks.value() > 0) {
    FillTrainingWindowKernel<<<window_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
        encoded_corpus_device.data_as<int32_t>(), input_ids->data_as<int32_t>(),
        target_ids->data_as<int32_t>(), seq_len, offset);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("FillTrainingWindowKernel"));
  }
  return Status::Ok();
}

inline Status RunCharTrainBody(RuntimeContext &ctx, Sequential &model, AdamOptimizer &optimizer,
                               const std::vector<ParameterRef> &params, const Tensor &input_ids,
                               const Tensor &target_ids, Tensor *logits, Tensor *loss_grad,
                               float grad_clip, ClassificationMetrics *metrics, float *grad_norm) {
  DLCUDA_RETURN_IF_ERROR(optimizer.ZeroGrad(ctx, params));

  DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, input_ids, logits));
  if (metrics != nullptr) {
    auto metrics_result = CategoricalCrossEntropyMetricsFromLogits(ctx, target_ids, *logits);
    if (!metrics_result.ok()) {
      return metrics_result.status();
    }
    *metrics = metrics_result.value();
  }

  DLCUDA_RETURN_IF_ERROR(
      CategoricalCrossEntropyBackwardFromLogits(ctx, target_ids, *logits, loss_grad));
  DLCUDA_RETURN_IF_ERROR(model.Backward(ctx, *loss_grad, nullptr));
  DLCUDA_RETURN_IF_ERROR(ClipGradNorm(ctx, params, grad_clip, grad_norm));
  return Status::Ok();
}

inline int64_t EvaluationOffset(const WindowSplit &split, int64_t index, int64_t windows) {
  if (windows <= 1 || split.count <= 1) {
    return split.begin + split.count / 2;
  }
  int64_t last = split.count - 1;
  return split.begin + (index * last) / (windows - 1);
}

inline Result<EvaluationSummary> EvaluateCharSplit(RuntimeContext &ctx, Sequential &model,
                                                   const Tensor &encoded_corpus_device,
                                                   Tensor *input_ids, Tensor *target_ids,
                                                   int seq_len, const WindowSplit &split,
                                                   int requested_windows) {
  if (input_ids == nullptr || target_ids == nullptr) {
    return Status::InvalidArgument("EvaluateCharSplit received null tensor pointer");
  }
  if (split.count <= 0) {
    return Status::InvalidArgument("EvaluateCharSplit requires a non-empty split");
  }
  if (requested_windows <= 0) {
    return Status::InvalidArgument("requested evaluation windows must be > 0");
  }

  int64_t windows = std::min<int64_t>(split.count, requested_windows);
  double loss_sum = 0.0;
  double accuracy_sum = 0.0;
  Tensor logits;
  for (int64_t i = 0; i < windows; ++i) {
    int64_t offset = EvaluationOffset(split, i, windows);
    DLCUDA_RETURN_IF_ERROR(
        FillTrainingWindow(ctx, encoded_corpus_device, input_ids, target_ids, seq_len, offset));
    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, *input_ids, &logits));
    auto metrics_result = CategoricalCrossEntropyMetricsFromLogits(ctx, *target_ids, logits);
    if (!metrics_result.ok()) {
      return metrics_result.status();
    }
    loss_sum += static_cast<double>(metrics_result.value().loss);
    accuracy_sum += static_cast<double>(metrics_result.value().accuracy);
  }

  EvaluationSummary summary;
  summary.windows = windows;
  summary.metrics.loss = static_cast<float>(loss_sum / static_cast<double>(windows));
  summary.metrics.accuracy = static_cast<float>(accuracy_sum / static_cast<double>(windows));
  return summary;
}

} // namespace
} // namespace dlcuda
