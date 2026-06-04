#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace dlcuda::detail {

inline constexpr uint32_t kMaxCheckpointTensorRank = 16;
inline constexpr uint32_t kMaxCheckpointStringBytes = 1U << 20;

struct FileCloser {
  void operator()(FILE *file) const {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;

inline Status CloseFile(FilePtr *file, const char *context) {
  if (file == nullptr || !*file) {
    return Status::Ok();
  }
  FILE *raw = file->release();
  if (std::fclose(raw) != 0) {
    return Status::IoError(std::string("Failed to close ") + context);
  }
  return Status::Ok();
}

inline bool WriteExact(FILE *file, const void *data, size_t size) {
  return std::fwrite(data, 1, size, file) == size;
}

inline bool ReadExact(FILE *file, void *data, size_t size) {
  return std::fread(data, 1, size, file) == size;
}

inline Status WriteString(FILE *file, const std::string &text, const char *context) {
  if (text.size() > kMaxCheckpointStringBytes) {
    return Status::InvalidArgument(std::string(context) + " string is too large");
  }
  uint32_t len = static_cast<uint32_t>(text.size());
  if (!WriteExact(file, &len, sizeof(len))) {
    return Status::IoError(std::string("Failed to write ") + context + " string length");
  }
  if (len > 0 && !WriteExact(file, text.data(), len)) {
    return Status::IoError(std::string("Failed to write ") + context + " string bytes");
  }
  return Status::Ok();
}

inline Status ReadString(FILE *file, std::string *text, const char *context) {
  if (text == nullptr) {
    return Status::InvalidArgument("ReadString destination is null");
  }
  uint32_t len = 0;
  if (!ReadExact(file, &len, sizeof(len))) {
    return Status::IoError(std::string("Failed to read ") + context + " string length");
  }
  if (len > kMaxCheckpointStringBytes) {
    return Status::InvalidArgument(std::string(context) + " string is too large");
  }
  text->assign(len, '\0');
  if (len > 0 && !ReadExact(file, text->data(), len)) {
    return Status::IoError(std::string("Failed to read ") + context + " string bytes");
  }
  return Status::Ok();
}

inline Result<uint64_t> TensorByteSizeForShape(DType dtype, const std::vector<int64_t> &shape,
                                               const char *context) {
  size_t dtype_size = DTypeSize(dtype);
  if (dtype_size == 0) {
    return Status::InvalidArgument(std::string("Unsupported ") + context + " dtype");
  }
  auto numel_result = ShapeNumel(shape);
  if (!numel_result.ok()) {
    return numel_result.status();
  }
  int64_t numel = numel_result.value();
  if (static_cast<uint64_t>(numel) >
      std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dtype_size)) {
    return Status::InvalidArgument(std::string(context) + " byte size overflow");
  }
  return static_cast<uint64_t>(numel) * static_cast<uint64_t>(dtype_size);
}

inline Status CopyDeviceToHost(RuntimeContext &ctx, const Tensor &src, std::vector<char> *dst) {
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

inline Status CopyHostToDevice(RuntimeContext &ctx, const std::vector<char> &src, Tensor *dst,
                               const char *size_mismatch_message) {
  if (dst == nullptr || !dst->defined()) {
    return Status::InvalidArgument("CopyHostToDevice destination is undefined");
  }
  if (src.empty()) {
    return Status::Ok();
  }
  if (src.size() != dst->bytes()) {
    return Status::InvalidArgument(size_mismatch_message);
  }
  DLCUDA_RETURN_IF_ERROR(dst->CopyFromHost(src.data(), src.size(), ctx.stream()));
  return ctx.Synchronize();
}

} // namespace dlcuda::detail
