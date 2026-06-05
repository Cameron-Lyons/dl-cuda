#include "common.cuh"

#include "dl_cuda/detail/value_validation.hpp"

namespace dlcuda::examples_detail {

RuntimeOptions OptionsFromXorConfig(const TrainXorConfig &cfg) {
  RuntimeOptions opts;
  opts.use_cublas = cfg.use_cublas;
  opts.tf32 = cfg.tf32;
  opts.seed = cfg.seed;
  opts.stream = 0;
  return opts;
}

RuntimeOptions OptionsFromCharConfig(bool use_cublas, bool tf32, uint64_t seed) {
  RuntimeOptions opts;
  opts.use_cublas = use_cublas;
  opts.tf32 = tf32;
  opts.seed = seed;
  opts.stream = 0;
  return opts;
}

Status ValidatePositiveFinite(float value, const char *name) {
  return ::dlcuda::detail::ValidatePositiveFinite(value, name);
}

Status ValidateTopP(float value) {
  if (!std::isfinite(value) || !(value > 0.0f && value <= 1.0f)) {
    return Status::InvalidArgument("top_p must be finite and in (0, 1]");
  }
  return Status::Ok();
}

Status ValidateFraction(float value, const char *name) {
  return ::dlcuda::detail::ValidateRate(value, name);
}

Status ValidateXorConfig(const TrainXorConfig &cfg) {
  if (cfg.epochs < 0) {
    return Status::InvalidArgument("epochs must be >= 0");
  }
  if (cfg.print_every <= 0) {
    return Status::InvalidArgument("print_every must be > 0");
  }
  if (cfg.hidden_size <= 0) {
    return Status::InvalidArgument("hidden_size must be > 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.lr, "lr"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.grad_clip, "grad_clip"));
  return Status::Ok();
}

Status ValidateCharConfig(const TrainCharConfig &cfg) {
  if (cfg.seq_len <= 1) {
    return Status::InvalidArgument("seq_len must be > 1");
  }
  if (cfg.d_model <= 0) {
    return Status::InvalidArgument("d_model must be > 0");
  }
  if (cfg.epochs < 0) {
    return Status::InvalidArgument("epochs must be >= 0");
  }
  if (cfg.print_every <= 0) {
    return Status::InvalidArgument("print_every must be > 0");
  }
  if (cfg.val_every <= 0) {
    return Status::InvalidArgument("val_every must be > 0");
  }
  if (cfg.val_windows <= 0) {
    return Status::InvalidArgument("val_windows must be > 0");
  }
  if (cfg.test_windows <= 0) {
    return Status::InvalidArgument("test_windows must be > 0");
  }
  if (cfg.early_stop_patience < 0) {
    return Status::InvalidArgument("early_stop_patience must be >= 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.lr, "lr"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.grad_clip, "grad_clip"));
  DLCUDA_RETURN_IF_ERROR(ValidateFraction(cfg.val_fraction, "val_fraction"));
  DLCUDA_RETURN_IF_ERROR(ValidateFraction(cfg.test_fraction, "test_fraction"));
  if (cfg.val_fraction + cfg.test_fraction >= 1.0f) {
    return Status::InvalidArgument("val_fraction + test_fraction must be < 1");
  }
  if (cfg.early_stop_patience > 0 && cfg.val_fraction <= 0.0f) {
    return Status::InvalidArgument("early stopping requires val_fraction > 0");
  }
  if (!std::isfinite(cfg.min_delta) || cfg.min_delta < 0.0f) {
    return Status::InvalidArgument("min_delta must be finite and >= 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.temperature, "temperature"));
  DLCUDA_RETURN_IF_ERROR(ValidateTopP(cfg.top_p));
  if (cfg.gen_len < 0) {
    return Status::InvalidArgument("gen_len must be >= 0");
  }
  return Status::Ok();
}

Status ValidateSampleCharConfig(const SampleCharConfig &cfg) {
  if (cfg.seq_len <= 1) {
    return Status::InvalidArgument("seq_len must be > 1");
  }
  if (cfg.d_model <= 0) {
    return Status::InvalidArgument("d_model must be > 0");
  }
  if (cfg.gen_len < 0) {
    return Status::InvalidArgument("gen_len must be >= 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.temperature, "temperature"));
  DLCUDA_RETURN_IF_ERROR(ValidateTopP(cfg.top_p));
  return Status::Ok();
}

Status ValidateCorpusWindow(size_t corpus_size, int seq_len, const char *context) {
  if (seq_len <= 0) {
    return Status::InvalidArgument(std::string(context) + " seq_len must be > 0");
  }
  if (static_cast<size_t>(seq_len) + 1 > corpus_size) {
    return Status::InvalidArgument(std::string(context) + " corpus is too short for seq_len");
  }
  return Status::Ok();
}

std::string BoolString(bool value) {
  return value ? "true" : "false";
}

std::string IntString(int64_t value) {
  return std::to_string(value);
}

std::string UIntString(uint64_t value) {
  return std::to_string(value);
}

std::string FloatString(float value) {
  std::ostringstream out;
  out.precision(std::numeric_limits<float>::max_digits10);
  out << value;
  return out.str();
}

uint64_t HashBytes(const std::string &bytes) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  for (unsigned char byte : bytes) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

std::string HashString(const std::string &bytes) {
  return UIntString(HashBytes(bytes));
}

std::string VocabBytes(const CharVocab &vocab) {
  std::string bytes;
  bytes.reserve(static_cast<size_t>(vocab.size()));
  for (int i = 0; i < vocab.size(); ++i) {
    bytes.push_back(vocab.Decode(i));
  }
  return bytes;
}

std::string SerializeMt19937(const std::mt19937 &rng) {
  std::ostringstream out;
  out << rng;
  return out.str();
}

const std::string *FindCheckpointValue(const std::vector<CheckpointKeyValue> &values,
                                       const std::string &key) {
  for (const auto &value : values) {
    if (value.key == key) {
      return &value.value;
    }
  }
  return nullptr;
}

Status RequireCheckpointValue(const std::vector<CheckpointKeyValue> &values, const std::string &key,
                              const std::string &expected, const char *context) {
  const std::string *actual = FindCheckpointValue(values, key);
  if (actual == nullptr) {
    return Status::InvalidArgument(std::string("Checkpoint missing ") + context +
                                   " metadata: " + key);
  }
  if (*actual != expected) {
    return Status::InvalidArgument(std::string("Checkpoint ") + context + " mismatch for " + key +
                                   ": expected " + expected + " got " + *actual);
  }
  return Status::Ok();
}

Result<int64_t> ComputeEndEpoch(int64_t start_epoch, int epochs_to_run) {
  if (start_epoch < 0) {
    return Status::InvalidArgument("Checkpoint epoch must be non-negative");
  }
  if (start_epoch > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(epochs_to_run)) {
    return Status::InvalidArgument("Checkpoint epoch plus requested epochs overflows");
  }
  return start_epoch + static_cast<int64_t>(epochs_to_run);
}

Status ValidateOptimizerStepMetadata(const CheckpointMetadata &metadata,
                                     const Optimizer &optimizer) {
  if (metadata.format_version >= 3 && metadata.step != optimizer.step_count()) {
    return Status::InvalidArgument("Checkpoint optimizer step does not match metadata step");
  }
  return Status::Ok();
}

Status RestoreMt19937State(const CheckpointMetadata &metadata, const std::string &name,
                           std::mt19937 *rng) {
  if (rng == nullptr) {
    return Status::InvalidArgument("RNG restore destination is null");
  }
  const std::string *state = FindCheckpointValue(metadata.rng_states, name);
  if (state == nullptr) {
    return Status::NotFound("Checkpoint missing RNG state: " + name);
  }
  std::istringstream in(*state);
  in >> *rng;
  if (!in) {
    return Status::InvalidArgument("Checkpoint RNG state is invalid: " + name);
  }
  return Status::Ok();
}

std::vector<CheckpointKeyValue> ConstantSchedulerState(float lr, int64_t step) {
  return {{"type", "constant"}, {"base_lr", FloatString(lr)}, {"step", IntString(step)}};
}

std::vector<CheckpointKeyValue> BuildXorTrainingConfig(const TrainXorConfig &cfg) {
  return {{"command", "train-xor"},
          {"hidden_size", IntString(cfg.hidden_size)},
          {"epochs", IntString(cfg.epochs)},
          {"print_every", IntString(cfg.print_every)},
          {"lr", FloatString(cfg.lr)},
          {"grad_clip", FloatString(cfg.grad_clip)},
          {"use_cublas", BoolString(cfg.use_cublas)},
          {"tf32", BoolString(cfg.tf32)},
          {"seed", UIntString(cfg.seed)}};
}

CheckpointMetadata BuildXorCheckpointMetadata(const TrainXorConfig &cfg, int64_t completed_epoch,
                                              int64_t step) {
  CheckpointMetadata metadata;
  metadata.model_name = "xor-mlp";
  metadata.format_version = 3;
  metadata.epoch = completed_epoch;
  metadata.step = step;
  metadata.training_config = BuildXorTrainingConfig(cfg);
  metadata.scheduler_state = ConstantSchedulerState(cfg.lr, step);
  metadata.extra_metadata = {{"dataset", "xor"}, {"samples", "4"}};
  return metadata;
}

Status ValidateXorCheckpointMetadata(const CheckpointMetadata &metadata,
                                     const TrainXorConfig &cfg) {
  if (metadata.format_version < 3) {
    return Status::Ok();
  }
  return RequireCheckpointValue(metadata.training_config, "hidden_size", IntString(cfg.hidden_size),
                                "training_config");
}

} // namespace dlcuda::examples_detail
