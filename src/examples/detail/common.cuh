#pragma once

#include "dl_cuda/examples.hpp"

#include "dl_cuda/checkpoint.hpp"
#include "dl_cuda/data.hpp"
#include "dl_cuda/loss.hpp"
#include "dl_cuda/nn.hpp"
#include "dl_cuda/optim.hpp"
#include "dl_cuda/runtime.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <cstdio>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dlcuda {
namespace {

[[maybe_unused]] constexpr int kExampleThreads = 256;

} // namespace
} // namespace dlcuda

namespace dlcuda::examples_detail {

RuntimeOptions OptionsFromXorConfig(const TrainXorConfig &cfg);
RuntimeOptions OptionsFromCharConfig(bool use_cublas, bool tf32, uint64_t seed);
Status ValidatePositiveFinite(float value, const char *name);
Status ValidateTopP(float value);
Status ValidateFraction(float value, const char *name);
Status ValidateXorConfig(const TrainXorConfig &cfg);
Status ValidateCharConfig(const TrainCharConfig &cfg);
Status ValidateSampleCharConfig(const SampleCharConfig &cfg);
Status ValidateCorpusWindow(size_t corpus_size, int seq_len, const char *context);
std::string BoolString(bool value);
std::string IntString(int64_t value);
std::string UIntString(uint64_t value);
std::string FloatString(float value);
uint64_t HashBytes(const std::string &bytes);
std::string HashString(const std::string &bytes);
std::string VocabBytes(const CharVocab &vocab);
std::string SerializeMt19937(const std::mt19937 &rng);
const std::string *FindCheckpointValue(const std::vector<CheckpointKeyValue> &values,
                                       const std::string &key);
Status RequireCheckpointValue(const std::vector<CheckpointKeyValue> &values, const std::string &key,
                              const std::string &expected, const char *context);
Result<int64_t> ComputeEndEpoch(int64_t start_epoch, int epochs_to_run);
Status ValidateOptimizerStepMetadata(const CheckpointMetadata &metadata,
                                     const Optimizer &optimizer);
Status RestoreMt19937State(const CheckpointMetadata &metadata, const std::string &name,
                           std::mt19937 *rng);
std::vector<CheckpointKeyValue> ConstantSchedulerState(float lr, int64_t step);
std::vector<CheckpointKeyValue> BuildXorTrainingConfig(const TrainXorConfig &cfg);
CheckpointMetadata BuildXorCheckpointMetadata(const TrainXorConfig &cfg, int64_t completed_epoch,
                                              int64_t step);
Status ValidateXorCheckpointMetadata(const CheckpointMetadata &metadata, const TrainXorConfig &cfg);

} // namespace dlcuda::examples_detail

namespace dlcuda {

using examples_detail::BoolString;
using examples_detail::BuildXorCheckpointMetadata;
using examples_detail::BuildXorTrainingConfig;
using examples_detail::ComputeEndEpoch;
using examples_detail::ConstantSchedulerState;
using examples_detail::FindCheckpointValue;
using examples_detail::FloatString;
using examples_detail::HashBytes;
using examples_detail::HashString;
using examples_detail::IntString;
using examples_detail::OptionsFromCharConfig;
using examples_detail::OptionsFromXorConfig;
using examples_detail::RequireCheckpointValue;
using examples_detail::RestoreMt19937State;
using examples_detail::SerializeMt19937;
using examples_detail::UIntString;
using examples_detail::ValidateCharConfig;
using examples_detail::ValidateCorpusWindow;
using examples_detail::ValidateFraction;
using examples_detail::ValidateOptimizerStepMetadata;
using examples_detail::ValidatePositiveFinite;
using examples_detail::ValidateSampleCharConfig;
using examples_detail::ValidateTopP;
using examples_detail::ValidateXorCheckpointMetadata;
using examples_detail::ValidateXorConfig;
using examples_detail::VocabBytes;

} // namespace dlcuda
