#pragma once

#include "dl_cuda/nn.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"

#include <string>
#include <vector>

namespace dlcuda {

struct CheckpointMetadata {
  std::string model_name;
  int32_t format_version = 2;
};

Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata,
                      const std::vector<ParameterRef> &params);

Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params);

} // namespace dlcuda
