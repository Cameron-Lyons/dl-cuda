#include "dl_cuda/checkpoint.hpp"

#include "dl_cuda/detail/checkpoint_io.hpp"
#include "dl_cuda/optim.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dlcuda {
namespace {

static constexpr const char kMagic[] = "DLCUDA2";
static constexpr uint32_t kLegacyFormatVersion = 2;
static constexpr uint32_t kCurrentFormatVersion = 3;

using detail::CloseFile;
using detail::CopyDeviceToHost;
using detail::FilePtr;
using detail::ReadExact;
using detail::WriteExact;

struct TensorRecord {
  DType dtype = DType::kFloat32;
  std::vector<int64_t> shape;
  std::vector<char> bytes;
};

Result<uint64_t> TensorByteSizeForShape(DType dtype, const std::vector<int64_t> &shape) {
  return detail::TensorByteSizeForShape(dtype, shape, "checkpoint tensor");
}

Status WriteString(FILE *file, const std::string &text) {
  return detail::WriteString(file, text, "checkpoint");
}

Status ReadString(FILE *file, std::string *text) {
  return detail::ReadString(file, text, "checkpoint");
}

Status CopyHostToDevice(RuntimeContext &ctx, const std::vector<char> &src, Tensor *dst) {
  return detail::CopyHostToDevice(ctx, src, dst, "Checkpoint tensor byte size mismatch");
}

Status ValidateCheckpointParameters(const std::vector<ParameterRef> &params) {
  if (params.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument("Too many parameters to checkpoint");
  }
  std::unordered_set<std::string> seen_names;
  seen_names.reserve(params.size());
  for (const auto &param : params) {
    if (param.name.empty()) {
      return Status::InvalidArgument("Checkpoint parameter names must be non-empty");
    }
    if (!seen_names.insert(param.name).second) {
      return Status::InvalidArgument("Duplicate checkpoint parameter name: " + param.name);
    }
    if (param.value == nullptr || !param.value->defined()) {
      return Status::InvalidArgument("Undefined parameter tensor: " + param.name);
    }
    auto expected_bytes = TensorByteSizeForShape(param.value->dtype(), param.value->shape());
    if (!expected_bytes.ok()) {
      return expected_bytes.status();
    }
    if (expected_bytes.value() != static_cast<uint64_t>(param.value->bytes())) {
      return Status::InvalidArgument("Parameter tensor byte size mismatch: " + param.name);
    }
  }
  return Status::Ok();
}

Status ValidateKeyValues(const std::vector<CheckpointKeyValue> &values, const char *field_name) {
  if (values.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(std::string(field_name) + " has too many checkpoint entries");
  }
  std::unordered_set<std::string> seen_keys;
  seen_keys.reserve(values.size());
  for (const auto &value : values) {
    if (value.key.empty()) {
      return Status::InvalidArgument(std::string(field_name) +
                                     " checkpoint keys must be non-empty");
    }
    if (!seen_keys.insert(value.key).second) {
      return Status::InvalidArgument(std::string("Duplicate checkpoint key in ") + field_name +
                                     ": " + value.key);
    }
    if (value.key.size() > detail::kMaxCheckpointStringBytes ||
        value.value.size() > detail::kMaxCheckpointStringBytes) {
      return Status::InvalidArgument(std::string(field_name) + " checkpoint string is too large");
    }
  }
  return Status::Ok();
}

bool HasV3MetadataFields(const CheckpointMetadata &metadata) {
  return metadata.epoch != 0 || metadata.step != 0 || !metadata.training_config.empty() ||
         !metadata.corpus_metadata.empty() || !metadata.vocab_metadata.empty() ||
         !metadata.scheduler_state.empty() || !metadata.extra_metadata.empty() ||
         !metadata.rng_states.empty();
}

Status ValidateCheckpointMetadata(const CheckpointMetadata &metadata, const Optimizer *optimizer) {
  if (metadata.model_name.empty()) {
    return Status::InvalidArgument("Checkpoint metadata.model_name is required");
  }
  if (metadata.format_version != static_cast<int32_t>(kLegacyFormatVersion) &&
      metadata.format_version != static_cast<int32_t>(kCurrentFormatVersion)) {
    return Status::InvalidArgument("Unsupported checkpoint format version");
  }
  if (metadata.epoch < 0) {
    return Status::InvalidArgument("Checkpoint metadata.epoch must be non-negative");
  }
  if (metadata.step < 0) {
    return Status::InvalidArgument("Checkpoint metadata.step must be non-negative");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateKeyValues(metadata.training_config, "training_config"));
  DLCUDA_RETURN_IF_ERROR(ValidateKeyValues(metadata.corpus_metadata, "corpus_metadata"));
  DLCUDA_RETURN_IF_ERROR(ValidateKeyValues(metadata.vocab_metadata, "vocab_metadata"));
  DLCUDA_RETURN_IF_ERROR(ValidateKeyValues(metadata.scheduler_state, "scheduler_state"));
  DLCUDA_RETURN_IF_ERROR(ValidateKeyValues(metadata.extra_metadata, "extra_metadata"));
  DLCUDA_RETURN_IF_ERROR(ValidateKeyValues(metadata.rng_states, "rng_states"));
  if (metadata.format_version == static_cast<int32_t>(kLegacyFormatVersion) &&
      (optimizer != nullptr || HasV3MetadataFields(metadata))) {
    return Status::InvalidArgument("Checkpoint format version 2 cannot store training state");
  }
  return Status::Ok();
}

Status WriteKeyValues(FILE *f, const std::vector<CheckpointKeyValue> &values,
                      const char *field_name) {
  if (values.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(std::string(field_name) + " has too many checkpoint entries");
  }
  uint32_t count = static_cast<uint32_t>(values.size());
  if (!WriteExact(f, &count, sizeof(count))) {
    return Status::IoError(std::string("Failed to write checkpoint ") + field_name + " count");
  }
  for (const auto &value : values) {
    DLCUDA_RETURN_IF_ERROR(WriteString(f, value.key));
    DLCUDA_RETURN_IF_ERROR(WriteString(f, value.value));
  }
  return Status::Ok();
}

Status ReadKeyValues(FILE *f, std::vector<CheckpointKeyValue> *values, const char *field_name) {
  if (values == nullptr) {
    return Status::InvalidArgument(std::string("ReadKeyValues destination is null: ") + field_name);
  }
  uint32_t count = 0;
  if (!ReadExact(f, &count, sizeof(count))) {
    return Status::IoError(std::string("Failed to read checkpoint ") + field_name + " count");
  }
  values->clear();
  values->resize(count);
  for (auto &value : *values) {
    DLCUDA_RETURN_IF_ERROR(ReadString(f, &value.key));
    DLCUDA_RETURN_IF_ERROR(ReadString(f, &value.value));
  }
  return ValidateKeyValues(*values, field_name);
}

Status WriteCheckpointMetadataV3(FILE *f, const CheckpointMetadata &metadata) {
  if (!WriteExact(f, &metadata.epoch, sizeof(metadata.epoch)) ||
      !WriteExact(f, &metadata.step, sizeof(metadata.step))) {
    return Status::IoError("Failed to write checkpoint training progress");
  }
  DLCUDA_RETURN_IF_ERROR(WriteKeyValues(f, metadata.training_config, "training_config"));
  DLCUDA_RETURN_IF_ERROR(WriteKeyValues(f, metadata.corpus_metadata, "corpus_metadata"));
  DLCUDA_RETURN_IF_ERROR(WriteKeyValues(f, metadata.vocab_metadata, "vocab_metadata"));
  DLCUDA_RETURN_IF_ERROR(WriteKeyValues(f, metadata.scheduler_state, "scheduler_state"));
  DLCUDA_RETURN_IF_ERROR(WriteKeyValues(f, metadata.extra_metadata, "extra_metadata"));
  return WriteKeyValues(f, metadata.rng_states, "rng_states");
}

Status ReadCheckpointMetadataV3(FILE *f, CheckpointMetadata *metadata) {
  if (metadata == nullptr) {
    return Status::InvalidArgument("ReadCheckpointMetadataV3 destination is null");
  }
  if (!ReadExact(f, &metadata->epoch, sizeof(metadata->epoch)) ||
      !ReadExact(f, &metadata->step, sizeof(metadata->step))) {
    return Status::IoError("Failed to read checkpoint training progress");
  }
  if (metadata->epoch < 0 || metadata->step < 0) {
    return Status::InvalidArgument("Checkpoint training progress must be non-negative");
  }
  DLCUDA_RETURN_IF_ERROR(ReadKeyValues(f, &metadata->training_config, "training_config"));
  DLCUDA_RETURN_IF_ERROR(ReadKeyValues(f, &metadata->corpus_metadata, "corpus_metadata"));
  DLCUDA_RETURN_IF_ERROR(ReadKeyValues(f, &metadata->vocab_metadata, "vocab_metadata"));
  DLCUDA_RETURN_IF_ERROR(ReadKeyValues(f, &metadata->scheduler_state, "scheduler_state"));
  DLCUDA_RETURN_IF_ERROR(ReadKeyValues(f, &metadata->extra_metadata, "extra_metadata"));
  return ReadKeyValues(f, &metadata->rng_states, "rng_states");
}

} // namespace

Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata, const std::vector<ParameterRef> &params) {
  return SaveCheckpoint(ctx, path, metadata, params, nullptr);
}

Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata, const std::vector<ParameterRef> &params,
                      Optimizer *optimizer) {
  if (path.empty()) {
    return Status::InvalidArgument("Checkpoint path must be non-empty");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateCheckpointMetadata(metadata, optimizer));
  DLCUDA_RETURN_IF_ERROR(ValidateCheckpointParameters(params));

  FilePtr file(std::fopen(path.c_str(), "wb"));
  if (!file) {
    return Status::IoError("Failed to open checkpoint for writing: " + path);
  }
  FILE *f = file.get();

  if (!WriteExact(f, kMagic, sizeof(kMagic))) {
    return Status::IoError("Failed to write checkpoint magic");
  }

  uint32_t version = static_cast<uint32_t>(metadata.format_version);
  if (!WriteExact(f, &version, sizeof(version))) {
    return Status::IoError("Failed to write checkpoint version");
  }

  Status write_model_name_status = WriteString(f, metadata.model_name);
  if (!write_model_name_status.ok()) {
    return write_model_name_status;
  }

  uint32_t param_count = static_cast<uint32_t>(params.size());
  if (!WriteExact(f, &param_count, sizeof(param_count))) {
    return Status::IoError("Failed to write parameter count");
  }

  for (const auto &param : params) {
    Status write_param_name_status = WriteString(f, param.name);
    if (!write_param_name_status.ok()) {
      return write_param_name_status;
    }

    uint32_t dtype = static_cast<uint32_t>(param.value->dtype());
    if (!WriteExact(f, &dtype, sizeof(dtype))) {
      return Status::IoError("Failed to write parameter dtype");
    }

    if (param.value->shape().size() > detail::kMaxCheckpointTensorRank) {
      return Status::InvalidArgument("Parameter rank is too large to checkpoint: " + param.name);
    }
    uint32_t rank = static_cast<uint32_t>(param.value->shape().size());
    if (!WriteExact(f, &rank, sizeof(rank))) {
      return Status::IoError("Failed to write parameter rank");
    }
    if (rank > 0 && !WriteExact(f, param.value->shape().data(), rank * sizeof(int64_t))) {
      return Status::IoError("Failed to write parameter shape");
    }

    uint64_t bytes = static_cast<uint64_t>(param.value->bytes());
    if (!WriteExact(f, &bytes, sizeof(bytes))) {
      return Status::IoError("Failed to write parameter byte size");
    }

    std::vector<char> host_data;
    Status copy_status = CopyDeviceToHost(ctx, *param.value, &host_data);
    if (!copy_status.ok()) {
      return copy_status;
    }

    if (bytes > 0 && !WriteExact(f, host_data.data(), static_cast<size_t>(bytes))) {
      return Status::IoError("Failed to write parameter bytes");
    }
  }

  if (version == kCurrentFormatVersion) {
    DLCUDA_RETURN_IF_ERROR(WriteCheckpointMetadataV3(f, metadata));
    uint8_t has_optimizer = optimizer == nullptr ? 0U : 1U;
    if (!WriteExact(f, &has_optimizer, sizeof(has_optimizer))) {
      return Status::IoError("Failed to write checkpoint optimizer state flag");
    }
    if (optimizer != nullptr) {
      DLCUDA_RETURN_IF_ERROR(optimizer->SaveCheckpoint(ctx, f, params));
    }
  }

  if (std::fflush(f) != 0) {
    return Status::IoError("Failed to flush checkpoint file");
  }
  return CloseFile(&file, "checkpoint file");
}

Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params, CheckpointMetadata *metadata) {
  return LoadCheckpoint(ctx, path, expected_model_name, params, static_cast<Optimizer *>(nullptr),
                        metadata);
}

Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params, Optimizer *optimizer,
                      CheckpointMetadata *metadata) {
  if (expected_model_name.empty()) {
    return Status::InvalidArgument("expected_model_name must be set");
  }
  if (path.empty()) {
    return Status::InvalidArgument("Checkpoint path must be non-empty");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateCheckpointParameters(params));

  FilePtr file(std::fopen(path.c_str(), "rb"));
  if (!file) {
    return Status::IoError("Failed to open checkpoint for reading: " + path);
  }
  FILE *f = file.get();

  char magic[sizeof(kMagic)] = {0};
  if (!ReadExact(f, magic, sizeof(magic))) {
    return Status::IoError("Failed to read checkpoint magic");
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return Status::InvalidArgument("Checkpoint magic mismatch");
  }

  uint32_t version = 0;
  if (!ReadExact(f, &version, sizeof(version))) {
    return Status::IoError("Failed to read checkpoint version");
  }
  if (version != kLegacyFormatVersion && version != kCurrentFormatVersion) {
    return Status::InvalidArgument("Unsupported checkpoint version");
  }

  std::string model_name;
  Status read_model_name_status = ReadString(f, &model_name);
  if (!read_model_name_status.ok()) {
    return read_model_name_status;
  }
  if (model_name != expected_model_name) {
    return Status::InvalidArgument("Checkpoint model mismatch: expected " + expected_model_name +
                                   " got " + model_name);
  }
  CheckpointMetadata loaded_metadata;
  loaded_metadata.model_name = model_name;
  loaded_metadata.format_version = static_cast<int32_t>(version);

  uint32_t param_count = 0;
  if (!ReadExact(f, &param_count, sizeof(param_count))) {
    return Status::IoError("Failed to read checkpoint parameter count");
  }
  if (param_count != params.size()) {
    return Status::InvalidArgument("Checkpoint parameter count does not match current model");
  }

  std::unordered_map<std::string, TensorRecord> records;
  records.reserve(param_count);
  std::unordered_set<std::string> checkpoint_names;
  checkpoint_names.reserve(param_count);

  for (uint32_t i = 0; i < param_count; ++i) {
    std::string name;
    Status read_name_status = ReadString(f, &name);
    if (!read_name_status.ok()) {
      return read_name_status;
    }
    if (!checkpoint_names.insert(name).second) {
      return Status::InvalidArgument("Duplicate parameter in checkpoint: " + name);
    }

    uint32_t dtype_u32 = 0;
    if (!ReadExact(f, &dtype_u32, sizeof(dtype_u32))) {
      return Status::IoError("Failed to read parameter dtype");
    }
    DType dtype = static_cast<DType>(dtype_u32);
    if (DTypeSize(dtype) == 0) {
      return Status::InvalidArgument("Unsupported checkpoint parameter dtype");
    }

    uint32_t rank = 0;
    if (!ReadExact(f, &rank, sizeof(rank))) {
      return Status::IoError("Failed to read parameter rank");
    }
    if (rank > detail::kMaxCheckpointTensorRank) {
      return Status::InvalidArgument("Checkpoint tensor rank is too large");
    }

    TensorRecord record;
    record.dtype = dtype;
    record.shape.resize(rank);
    if (rank > 0 && !ReadExact(f, record.shape.data(), rank * sizeof(int64_t))) {
      return Status::IoError("Failed to read parameter shape");
    }

    uint64_t bytes = 0;
    if (!ReadExact(f, &bytes, sizeof(bytes))) {
      return Status::IoError("Failed to read parameter byte size");
    }
    auto expected_bytes = TensorByteSizeForShape(dtype, record.shape);
    if (!expected_bytes.ok()) {
      return expected_bytes.status();
    }
    if (bytes != expected_bytes.value()) {
      return Status::InvalidArgument("Checkpoint tensor byte size does not match dtype and shape");
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return Status::InvalidArgument("Checkpoint tensor byte size is too large");
    }

    record.bytes.resize(static_cast<size_t>(bytes));
    if (bytes > 0 && !ReadExact(f, record.bytes.data(), static_cast<size_t>(bytes))) {
      return Status::IoError("Failed to read parameter bytes");
    }

    records.emplace(name, std::move(record));
  }

  for (const auto &param : params) {
    auto it = records.find(param.name);
    if (it == records.end()) {
      return Status::NotFound("Missing parameter in checkpoint: " + param.name);
    }

    const TensorRecord &record = it->second;
    if (record.dtype != param.value->dtype()) {
      return Status::InvalidArgument("DType mismatch for parameter: " + param.name + " expected " +
                                     DTypeName(param.value->dtype()) + " got " +
                                     DTypeName(record.dtype));
    }
    if (record.shape != param.value->shape()) {
      return Status::InvalidArgument("Shape mismatch for parameter: " + param.name);
    }
    if (record.bytes.size() != param.value->bytes()) {
      return Status::InvalidArgument("Byte size mismatch for parameter: " + param.name);
    }

    DLCUDA_RETURN_IF_ERROR(CopyHostToDevice(ctx, record.bytes, param.value));
  }

  if (version == kCurrentFormatVersion) {
    DLCUDA_RETURN_IF_ERROR(ReadCheckpointMetadataV3(f, &loaded_metadata));
    uint8_t has_optimizer = 0;
    if (!ReadExact(f, &has_optimizer, sizeof(has_optimizer))) {
      return Status::IoError("Failed to read checkpoint optimizer state flag");
    }
    if (has_optimizer > 1U) {
      return Status::InvalidArgument("Checkpoint optimizer state flag is invalid");
    }
    if (metadata != nullptr) {
      *metadata = loaded_metadata;
    }
    if (optimizer != nullptr) {
      if (has_optimizer == 0U) {
        return Status::NotFound("Checkpoint does not contain optimizer state");
      }
      DLCUDA_RETURN_IF_ERROR(optimizer->LoadCheckpoint(ctx, f, params));
    }
    return Status::Ok();
  }

  if (metadata != nullptr) {
    *metadata = loaded_metadata;
  }
  if (optimizer != nullptr) {
    return Status::NotFound("Checkpoint format version 2 does not contain optimizer state");
  }
  return Status::Ok();
}

} // namespace dlcuda
