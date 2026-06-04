#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

namespace dlcuda {

struct ClassificationMetrics {
  float loss = 0.0f;
  float accuracy = 0.0f;
};

Status BinaryCrossEntropyBackward(RuntimeContext &ctx, const Tensor &targets,
                                  const Tensor &predictions, Tensor *prediction_grads);

Result<float> BinaryCrossEntropyLoss(RuntimeContext &ctx, const Tensor &targets,
                                     const Tensor &predictions);

Status CategoricalCrossEntropyBackwardFromLogits(RuntimeContext &ctx, const Tensor &target_ids,
                                                 const Tensor &logits, Tensor *logit_grads);

Result<ClassificationMetrics> CategoricalCrossEntropyMetricsFromLogits(RuntimeContext &ctx,
                                                                       const Tensor &target_ids,
                                                                       const Tensor &logits);

} // namespace dlcuda
