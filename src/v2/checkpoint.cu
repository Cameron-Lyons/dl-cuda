#include "dl_cuda/checkpoint.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace dlcuda {
namespace {

static constexpr const char kMagic[] = "DLCUDA2";
static constexpr uint32_t kFormatVersion = 2;

struct TensorRecord {
  DType dtype = DType::kFloat32;
  std::vector<int64_t> shape;
  std::vector<char> bytes;
};

bool WriteExact(FILE *f, const void *data, size_t size) {
  return std::fwrite(data, 1, size, f) == size;
}

bool ReadExact(FILE *f, void *data, size_t size) {
  return std::fread(data, 1, size, f) == size;
}

Status WriteString(FILE *f, const std::string &text) {
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
  uint32_t len = 0;
  if (!ReadExact(f, &len, sizeof(len))) {
    return Status::IoError("Failed to read string length");
  }
  text->assign(len, '\0');
  if (len > 0 && !ReadExact(f, text->data(), len)) {
    return Status::IoError("Failed to read string bytes");
  }
  return Status::Ok();
}

Status CopyDeviceToHost(RuntimeContext &ctx, const Tensor &src,
                        std::vector<char> *dst) {
  dst->resize(src.bytes());
  if (dst->empty()) {
    return Status::Ok();
  }
  cudaError_t err = cudaMemcpyAsync(dst->data(), src.data(), src.bytes(),
                                    cudaMemcpyDeviceToHost, ctx.stream());
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMemcpyAsync D2H failed: ") +
                                cudaGetErrorString(err));
  }
  err = cudaStreamSynchronize(ctx.stream());
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(err));
  }
  return Status::Ok();
}

Status CopyHostToDevice(RuntimeContext &ctx, const std::vector<char> &src,
                        Tensor *dst) {
  if (src.empty()) {
    return Status::Ok();
  }
  if (src.size() != dst->bytes()) {
    return Status::InvalidArgument("Checkpoint tensor byte size mismatch");
  }
  cudaError_t err = cudaMemcpyAsync(dst->data(), src.data(), src.size(),
                                    cudaMemcpyHostToDevice, ctx.stream());
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMemcpyAsync H2D failed: ") +
                                cudaGetErrorString(err));
  }
  err = cudaStreamSynchronize(ctx.stream());
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(err));
  }
  return Status::Ok();
}

} // namespace

Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const CheckpointMetadata &metadata,
                      const std::vector<ParameterRef> &params) {
  if (metadata.model_name.empty()) {
    return Status::InvalidArgument("Checkpoint metadata.model_name is required");
  }
  if (metadata.format_version != static_cast<int32_t>(kFormatVersion)) {
    return Status::InvalidArgument("Unsupported checkpoint format version");
  }

  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) {
    return Status::IoError("Failed to open checkpoint for writing: " + path);
  }

  auto close_file = [&]() { std::fclose(f); };

  if (!WriteExact(f, kMagic, sizeof(kMagic))) {
    close_file();
    return Status::IoError("Failed to write checkpoint magic");
  }

  uint32_t version = static_cast<uint32_t>(metadata.format_version);
  if (!WriteExact(f, &version, sizeof(version))) {
    close_file();
    return Status::IoError("Failed to write checkpoint version");
  }

  Status write_model_name_status = WriteString(f, metadata.model_name);
  if (!write_model_name_status.ok()) {
    close_file();
    return write_model_name_status;
  }

  uint32_t param_count = static_cast<uint32_t>(params.size());
  if (!WriteExact(f, &param_count, sizeof(param_count))) {
    close_file();
    return Status::IoError("Failed to write parameter count");
  }

  for (const auto &param : params) {
    if (param.value == nullptr || !param.value->defined()) {
      close_file();
      return Status::InvalidArgument("Undefined parameter tensor: " + param.name);
    }

    Status write_param_name_status = WriteString(f, param.name);
    if (!write_param_name_status.ok()) {
      close_file();
      return write_param_name_status;
    }

    uint32_t dtype = static_cast<uint32_t>(param.value->dtype());
    if (!WriteExact(f, &dtype, sizeof(dtype))) {
      close_file();
      return Status::IoError("Failed to write parameter dtype");
    }

    uint32_t rank = static_cast<uint32_t>(param.value->shape().size());
    if (!WriteExact(f, &rank, sizeof(rank))) {
      close_file();
      return Status::IoError("Failed to write parameter rank");
    }
    if (rank > 0 &&
        !WriteExact(f, param.value->shape().data(), rank * sizeof(int64_t))) {
      close_file();
      return Status::IoError("Failed to write parameter shape");
    }

    uint64_t bytes = static_cast<uint64_t>(param.value->bytes());
    if (!WriteExact(f, &bytes, sizeof(bytes))) {
      close_file();
      return Status::IoError("Failed to write parameter byte size");
    }

    std::vector<char> host_data;
    Status copy_status = CopyDeviceToHost(ctx, *param.value, &host_data);
    if (!copy_status.ok()) {
      close_file();
      return copy_status;
    }

    if (bytes > 0 && !WriteExact(f, host_data.data(), static_cast<size_t>(bytes))) {
      close_file();
      return Status::IoError("Failed to write parameter bytes");
    }
  }

  close_file();
  return Status::Ok();
}

Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                      const std::string &expected_model_name,
                      const std::vector<ParameterRef> &params) {
  if (expected_model_name.empty()) {
    return Status::InvalidArgument("expected_model_name must be set");
  }

  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return Status::IoError("Failed to open checkpoint for reading: " + path);
  }

  auto close_file = [&]() { std::fclose(f); };

  char magic[sizeof(kMagic)] = {0};
  if (!ReadExact(f, magic, sizeof(magic))) {
    close_file();
    return Status::IoError("Failed to read checkpoint magic");
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    close_file();
    return Status::InvalidArgument("Checkpoint magic mismatch");
  }

  uint32_t version = 0;
  if (!ReadExact(f, &version, sizeof(version))) {
    close_file();
    return Status::IoError("Failed to read checkpoint version");
  }
  if (version != kFormatVersion) {
    close_file();
    return Status::InvalidArgument("Unsupported checkpoint version");
  }

  std::string model_name;
  Status read_model_name_status = ReadString(f, &model_name);
  if (!read_model_name_status.ok()) {
    close_file();
    return read_model_name_status;
  }
  if (model_name != expected_model_name) {
    close_file();
    return Status::InvalidArgument("Checkpoint model mismatch: expected " +
                                   expected_model_name + " got " + model_name);
  }

  uint32_t param_count = 0;
  if (!ReadExact(f, &param_count, sizeof(param_count))) {
    close_file();
    return Status::IoError("Failed to read checkpoint parameter count");
  }

  std::unordered_map<std::string, TensorRecord> records;
  records.reserve(param_count);

  for (uint32_t i = 0; i < param_count; ++i) {
    std::string name;
    Status read_name_status = ReadString(f, &name);
    if (!read_name_status.ok()) {
      close_file();
      return read_name_status;
    }

    uint32_t dtype_u32 = 0;
    if (!ReadExact(f, &dtype_u32, sizeof(dtype_u32))) {
      close_file();
      return Status::IoError("Failed to read parameter dtype");
    }
    DType dtype = static_cast<DType>(dtype_u32);

    uint32_t rank = 0;
    if (!ReadExact(f, &rank, sizeof(rank))) {
      close_file();
      return Status::IoError("Failed to read parameter rank");
    }

    TensorRecord record;
    record.dtype = dtype;
    record.shape.resize(rank);
    if (rank > 0 && !ReadExact(f, record.shape.data(), rank * sizeof(int64_t))) {
      close_file();
      return Status::IoError("Failed to read parameter shape");
    }

    uint64_t bytes = 0;
    if (!ReadExact(f, &bytes, sizeof(bytes))) {
      close_file();
      return Status::IoError("Failed to read parameter byte size");
    }

    record.bytes.resize(static_cast<size_t>(bytes));
    if (bytes > 0 && !ReadExact(f, record.bytes.data(), static_cast<size_t>(bytes))) {
      close_file();
      return Status::IoError("Failed to read parameter bytes");
    }

    records.emplace(name, std::move(record));
  }

  close_file();

  if (records.size() != params.size()) {
    return Status::InvalidArgument(
        "Checkpoint parameter count does not match current model");
  }

  for (const auto &param : params) {
    if (param.value == nullptr || !param.value->defined()) {
      return Status::InvalidArgument("Undefined parameter tensor: " + param.name);
    }
    auto it = records.find(param.name);
    if (it == records.end()) {
      return Status::NotFound("Missing parameter in checkpoint: " + param.name);
    }

    const TensorRecord &record = it->second;
    if (record.dtype != param.value->dtype()) {
      return Status::InvalidArgument("DType mismatch for parameter: " + param.name);
    }
    if (record.shape != param.value->shape()) {
      return Status::InvalidArgument("Shape mismatch for parameter: " + param.name);
    }
    if (record.bytes.size() != param.value->bytes()) {
      return Status::InvalidArgument("Byte size mismatch for parameter: " +
                                     param.name);
    }

    DLCUDA_RETURN_IF_ERROR(CopyHostToDevice(ctx, record.bytes, param.value));
  }

  return Status::Ok();
}

} // namespace dlcuda
