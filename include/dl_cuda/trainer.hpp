#pragma once

#include "dl_cuda/nn.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"

#include <vector>

namespace dlcuda {

struct TrainStepStats {
  float loss = 0.0f;
  float accuracy = 0.0f;
  float learning_rate = 0.0f;
};

Result<float> ClipGradNorm(RuntimeContext &ctx,
                           const std::vector<ParameterRef> &params,
                           float max_norm);

} // namespace dlcuda
