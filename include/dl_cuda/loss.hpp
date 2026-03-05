#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

namespace dlcuda {

struct ClassificationMetrics {
  float loss = 0.0f;
  float accuracy = 0.0f;
};

Result<float> BinaryCrossEntropy(RuntimeContext &ctx, const Tensor &targets,
                                 const Tensor &predictions,
                                 Tensor *prediction_grads);

Result<ClassificationMetrics>
CategoricalCrossEntropyFromIds(RuntimeContext &ctx, const Tensor &target_ids,
                               const Tensor &probabilities,
                               Tensor *probability_grads);

} // namespace dlcuda
