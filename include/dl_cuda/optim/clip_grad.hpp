#pragma once

#include "dl_cuda/optim/optimizer.hpp"

namespace dlcuda {

Status ClipGradNorm(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float max_norm,
                    float *total_norm = nullptr);

} // namespace dlcuda
