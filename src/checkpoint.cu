#include "dl_cuda/checkpoint.hpp"

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
static constexpr uint32_t kFormatVersion = 2;
static constexpr uint32_t kMaxTensorRank = 16;
static constexpr uint32_t kMaxStringBytes = 1U << 20;

struct TensorRecord {
  DType dtype = DType::kFloat32;
  std::vector<int64_t> shape;
  std::vector<char> bytes;
};

Result<uint64_t> TensorByteSizeForShape(DType dtype, const std::vector<int64_t> &shape) {
  size_t dtype_size = DTypeSize(dtype);
  if (dtype_size == 0) {
    return Status::InvalidArgument("Unsupported checkpoint parameter dtype");
  }
  auto numel_result = ShapeNumel(shape);
  if (!numel_result.ok()) {
    return numel_result.status();
  }
  int64_t numel = numel_result.value();
  if (static_cast<uint64_t>(numel) >
      std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dtype_size)) {
    return Status::InvalidArgument("Checkpoint tensor byte size overflow");
  }
  return static_cast<uint64_t>(numel) * static_cast<uint64_t>(dtype_size);
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

struct FileCloser {
  void operator()(FILE *file) const {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;

Status CloseFile(FilePtr *file, const char *context) {
  if (file == nullptr || !*file) {
    return Status::Ok();
  }
  FILE *raw = file->release();
  if (std::fclose(raw) != 0) {
    return Status::IoError(std::string("Failed to close ") + context);
  }
  return Status::Ok();
}

bool WriteExact(FILE *f, const void *data, size_t size) {
  return std::fwrite(data, 1, size, f) == size;
}

bool ReadExact(FILE *f, void *data, size_t size) {
  return std::fread(data, 1, size, f) == size;
}

Status WriteString(FILE *f, const std::string &text) {
  if (text.size() > kMaxStringBytes) {
    return Status::InvalidArgument("String is too large to write");
  }
  uint32_t len = static_cast<uint32_t>(text.size());
  if (!WriteExact(f, &len, sizeof(len))) {
    return Status::IoError("Failed to write string length");
  }
  if (len > 0 && !WriteExact(f, text.data(), len)) {
    return Status::IoError("Failed to write string bytes");
  }
  return Status::Ok();
}

Status ReadString(FILE *f, std::string *text) {
  if (text == nullptr) {
    return Status::InvalidArgument("ReadString destination is null");
  }
  uint32_t len = 0;
  if (!ReadExact(f, &len, sizeof(len))) {
    return Status::IoError("Failed to read string length");
  }
  if (len > kMaxStringBytes) {
    return Status::InvalidArgument("Checkpoint string is too large");
  }
  text->assign(len, '\0');
  if (len > 0 && !ReadExact(f, text->data(), len)) {
    return Status::IoError("Failed to read string bytes");
  }
  return Status::Ok();
}

Status CopyDeviceToHost(RuntimeContext &ctx, const Tensor &src, std::vector<char> *dst) {
  if (dst == nullptr) {
    return Status::InvalidArgument("CopyDeviceToHost destination is null");
  }
  dst->resize(src.bytes());
  if (dst->empty()) {
    return Status::Ok();
  }
  DLCUDA_RETURN_IF_ERROR(src.CopyToHost(dst->data(), dst->size(), ctx.stream()));
  return ctx.Synchronize();
}

Status CopyHostToDevice(RuntimeContext &ctx, const std::vector<char> &src, Tensor *dst) {
  if (dst == nullptr || !dst->defined()) {
    return Status::InvalidArgument("CopyHostToDevice destination is undefined");
  }
  if (src.empty()) {
    return Status::Ok();
  }
  if (src.size() != dst->bytes()) {
    return Status::InvalidArgument("Checkpoint tensor byte size mismatch");
  }
  DLCUDA_RETURN_IF_ERROR(dst->CopyFromHost(src.data(), src.size(), ctx.stream()));
  return ctx.Synchronize();
}

} // namespace

Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata, const std::vector<ParameterRef> &params) {
  if (metadata.model_name.empty()) {
    return Status::InvalidArgument("Checkpoint metadata.model_name is required");
  }
  if (metadata.format_version != static_cast<int32_t>(kFormatVersion)) {
    return Status::InvalidArgument("Unsupported checkpoint format version");
  }
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

    if (param.value->shape().size() > kMaxTensorRank) {
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
  if (std::fflush(f) != 0) {
    return Status::IoError("Failed to flush checkpoint file");
  }
  return CloseFile(&file, "checkpoint file");
}

Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params) {
  if (expected_model_name.empty()) {
    return Status::InvalidArgument("expected_model_name must be set");
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
  if (version != kFormatVersion) {
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
    if (rank > kMaxTensorRank) {
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

  return Status::Ok();
}

} // namespace dlcuda
