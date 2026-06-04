#pragma once

#include "dl_cuda/nn.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dlcuda {

class Optimizer;

struct CheckpointKeyValue {
  std::string key;
  std::string value;
};

struct CheckpointMetadata {
  std::string model_name;
  int32_t format_version = 3;
  int64_t epoch = 0;
  int64_t step = 0;
  std::vector<CheckpointKeyValue> training_config;
  std::vector<CheckpointKeyValue> corpus_metadata;
  std::vector<CheckpointKeyValue> vocab_metadata;
  std::vector<CheckpointKeyValue> scheduler_state;
  std::vector<CheckpointKeyValue> extra_metadata;
  std::vector<CheckpointKeyValue> rng_states;
};

Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata, const std::vector<ParameterRef> &params);
Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata, const std::vector<ParameterRef> &params,
                      Optimizer *optimizer);

Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params,
                      CheckpointMetadata *metadata = nullptr);
Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params, Optimizer *optimizer,
                      CheckpointMetadata *metadata = nullptr);

} // namespace dlcuda
