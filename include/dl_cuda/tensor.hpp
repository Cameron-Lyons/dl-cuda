#pragma once

#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/status.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace dlcuda {

enum class DType {
  kFloat32 = 0,
  kInt32 = 1,
};

enum class DeviceType {
  kCuda = 0,
};

[[nodiscard]] inline size_t DTypeSize(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return sizeof(float);
  case DType::kInt32:
    return sizeof(int32_t);
  }
  return 0;
}

[[nodiscard]] inline const char *DTypeName(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return "float32";
  case DType::kInt32:
    return "int32";
  }
  return "unknown";
}

[[nodiscard]] inline Result<int64_t> ShapeNumel(const std::vector<int64_t> &shape) {
  int64_t numel = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return Status::InvalidArgument("Tensor shape must be non-negative");
    }
    if (dim != 0 && numel > std::numeric_limits<int64_t>::max() / dim) {
      return Status::InvalidArgument("Tensor shape element count overflow");
    }
    numel *= dim;
  }
  return numel;
}

class Tensor {
public:
  Tensor() = default;

  static Result<Tensor> Allocate(const std::vector<int64_t> &shape, DType dtype,
                                 DeviceType device = DeviceType::kCuda) {
    if (device != DeviceType::kCuda) {
      return Status::Unsupported("Only CUDA tensors are supported");
    }
    size_t dtype_size = DTypeSize(dtype);
    if (dtype_size == 0) {
      return Status::InvalidArgument("Unsupported tensor dtype");
    }

    auto numel_result = ShapeNumel(shape);
    if (!numel_result.ok()) {
      return numel_result.status();
    }
    int64_t numel = numel_result.value();
    if (static_cast<uint64_t>(numel) > std::numeric_limits<size_t>::max() / dtype_size) {
      return Status::InvalidArgument("Tensor byte size overflow");
    }
    size_t bytes = static_cast<size_t>(numel) * dtype_size;

    void *ptr = nullptr;
    if (bytes > 0) {
      cudaError_t err = cudaMalloc(&ptr, bytes);
      DLCUDA_RETURN_IF_ERROR(detail::CudaStatus(err, "cudaMalloc"));
    }

    Tensor out;
    out.storage_ = std::make_shared<Storage>(ptr, bytes, device);
    out.shape_ = shape;
    out.numel_ = numel;
    out.dtype_ = dtype;
    return out;
  }

  [[nodiscard]] bool defined() const {
    return storage_ != nullptr;
  }

  [[nodiscard]] void *data() const {
    return storage_ ? storage_->ptr : nullptr;
  }

  template <typename T> [[nodiscard]] T *data_as() const {
    return reinterpret_cast<T *>(data());
  }

  [[nodiscard]] const std::vector<int64_t> &shape() const {
    return shape_;
  }

  [[nodiscard]] int64_t rank() const {
    return static_cast<int64_t>(shape_.size());
  }

  [[nodiscard]] int64_t dim(int index) const {
    return shape_.at(static_cast<size_t>(index));
  }

  [[nodiscard]] int64_t numel() const {
    if (!defined()) {
      return 0;
    }
    return numel_;
  }

  [[nodiscard]] size_t bytes() const {
    if (!defined()) {
      return 0;
    }
    return storage_->bytes;
  }

  [[nodiscard]] DType dtype() const {
    return dtype_;
  }

  [[nodiscard]] DeviceType device() const {
    return storage_ ? storage_->device : DeviceType::kCuda;
  }

  Status FillZero(cudaStream_t stream = 0) {
    if (!defined()) {
      return Status::InvalidArgument("Tensor is undefined");
    }
    if (bytes() == 0) {
      return Status::Ok();
    }
    cudaError_t err = cudaMemsetAsync(data(), 0, bytes(), stream);
    return detail::CudaStatus(err, "cudaMemsetAsync");
  }

  Status CopyFromHost(const void *src, size_t bytes, cudaStream_t stream = 0) {
    if (!defined()) {
      return Status::InvalidArgument("Tensor is undefined");
    }
    if (src == nullptr && bytes > 0) {
      return Status::InvalidArgument("Host copy source is null");
    }
    if (bytes > this->bytes()) {
      return Status::InvalidArgument("Host copy exceeds tensor size");
    }
    if (bytes == 0) {
      return Status::Ok();
    }
    cudaError_t err = cudaMemcpyAsync(data(), src, bytes, cudaMemcpyHostToDevice, stream);
    return detail::CudaStatus(err, "cudaMemcpyAsync(H2D)");
  }

  Status CopyToHost(void *dst, size_t bytes, cudaStream_t stream = 0) const {
    if (!defined()) {
      return Status::InvalidArgument("Tensor is undefined");
    }
    if (dst == nullptr && bytes > 0) {
      return Status::InvalidArgument("Host copy destination is null");
    }
    if (bytes > this->bytes()) {
      return Status::InvalidArgument("Host copy exceeds tensor size");
    }
    if (bytes == 0) {
      return Status::Ok();
    }
    cudaError_t err = cudaMemcpyAsync(dst, data(), bytes, cudaMemcpyDeviceToHost, stream);
    return detail::CudaStatus(err, "cudaMemcpyAsync(D2H)");
  }

  Status CopyRangeToHost(void *dst, size_t offset_bytes, size_t bytes,
                         cudaStream_t stream = 0) const {
    if (!defined()) {
      return Status::InvalidArgument("Tensor is undefined");
    }
    if (dst == nullptr && bytes > 0) {
      return Status::InvalidArgument("Host copy destination is null");
    }
    if (offset_bytes > this->bytes() || bytes > this->bytes() - offset_bytes) {
      return Status::InvalidArgument("Host range copy exceeds tensor size");
    }
    if (bytes == 0) {
      return Status::Ok();
    }
    const char *src = static_cast<const char *>(data()) + offset_bytes;
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    return detail::CudaStatus(err, "cudaMemcpyAsync(D2H range)");
  }

private:
  struct Storage {
    Storage(void *ptr_in, size_t bytes_in, DeviceType device_in)
        : ptr(ptr_in), bytes(bytes_in), device(device_in) {}

    ~Storage() {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    }

    void *ptr = nullptr;
    size_t bytes = 0;
    DeviceType device = DeviceType::kCuda;
  };

  std::shared_ptr<Storage> storage_;
  std::vector<int64_t> shape_;
  int64_t numel_ = 1;
  DType dtype_ = DType::kFloat32;
};

inline Status CopyTensor(const Tensor &src, Tensor *dst, cudaStream_t stream = 0) {
  if (!src.defined() || !dst || !dst->defined()) {
    return Status::InvalidArgument("CopyTensor requires defined source and destination");
  }
  if (src.dtype() != dst->dtype() || src.shape() != dst->shape()) {
    return Status::InvalidArgument("CopyTensor requires matching dtype and shape");
  }
  if (src.bytes() == 0) {
    return Status::Ok();
  }
  cudaError_t err =
      cudaMemcpyAsync(dst->data(), src.data(), src.bytes(), cudaMemcpyDeviceToDevice, stream);
  return detail::CudaStatus(err, "cudaMemcpyAsync(D2D)");
}

inline Result<Tensor> CloneLike(const Tensor &src) {
  if (!src.defined()) {
    return Status::InvalidArgument("CloneLike requires a defined source tensor");
  }
  auto out = Tensor::Allocate(src.shape(), src.dtype(), src.device());
  if (!out.ok()) {
    return out;
  }
  return out;
}

inline Status EnsureTensor(Tensor *tensor, const std::vector<int64_t> &shape, DType dtype,
                           DeviceType device = DeviceType::kCuda) {
  if (tensor == nullptr) {
    return Status::InvalidArgument("EnsureTensor received null tensor");
  }
  if (tensor->defined() && tensor->shape() == shape && tensor->dtype() == dtype &&
      tensor->device() == device) {
    return Status::Ok();
  }
  auto allocated = Tensor::Allocate(shape, dtype, device);
  if (!allocated.ok()) {
    return allocated.status();
  }
  *tensor = allocated.value();
  return Status::Ok();
}

[[nodiscard]] inline std::string ShapeString(const Tensor &tensor) {
  std::string out = "[";
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(tensor.shape()[i]);
  }
  out += "]";
  return out;
}

} // namespace dlcuda
