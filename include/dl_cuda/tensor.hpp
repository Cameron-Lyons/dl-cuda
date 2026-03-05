#pragma once

#include "dl_cuda/status.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
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

inline size_t DTypeSize(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return sizeof(float);
  case DType::kInt32:
    return sizeof(int32_t);
  }
  return 0;
}

inline const char *DTypeName(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return "float32";
  case DType::kInt32:
    return "int32";
  }
  return "unknown";
}

class Tensor {
public:
  Tensor() = default;

  static Result<Tensor> Allocate(const std::vector<int64_t> &shape,
                                 DType dtype,
                                 DeviceType device = DeviceType::kCuda) {
    if (device != DeviceType::kCuda) {
      return Status::Unsupported("Only CUDA tensors are supported");
    }
    for (int64_t dim : shape) {
      if (dim < 0) {
        return Status::InvalidArgument("Tensor shape must be non-negative");
      }
    }

    int64_t numel = 1;
    for (int64_t dim : shape) {
      numel *= dim;
    }
    size_t bytes = static_cast<size_t>(numel) * DTypeSize(dtype);

    void *ptr = nullptr;
    if (bytes > 0) {
      cudaError_t err = cudaMalloc(&ptr, bytes);
      if (err != cudaSuccess) {
        return Status::RuntimeError(std::string("cudaMalloc failed: ") +
                                    cudaGetErrorString(err));
      }
    }

    Tensor out;
    out.storage_ = std::make_shared<Storage>(ptr, bytes, device);
    out.shape_ = shape;
    out.dtype_ = dtype;
    return out;
  }

  bool defined() const { return storage_ != nullptr; }

  void *data() const { return storage_ ? storage_->ptr : nullptr; }

  template <typename T> T *data_as() const {
    return reinterpret_cast<T *>(data());
  }

  const std::vector<int64_t> &shape() const { return shape_; }

  int64_t rank() const { return static_cast<int64_t>(shape_.size()); }

  int64_t dim(int index) const { return shape_.at(static_cast<size_t>(index)); }

  int64_t numel() const {
    if (shape_.empty()) {
      return 1;
    }
    int64_t out = 1;
    for (int64_t dim : shape_) {
      out *= dim;
    }
    return out;
  }

  size_t bytes() const {
    if (!defined()) {
      return 0;
    }
    return storage_->bytes;
  }

  DType dtype() const { return dtype_; }

  DeviceType device() const {
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
    if (err != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaMemsetAsync failed: ") +
                                  cudaGetErrorString(err));
    }
    return Status::Ok();
  }

  Status CopyFromHost(const void *src, size_t bytes, cudaStream_t stream = 0) {
    if (!defined()) {
      return Status::InvalidArgument("Tensor is undefined");
    }
    if (bytes > this->bytes()) {
      return Status::InvalidArgument("Host copy exceeds tensor size");
    }
    if (bytes == 0) {
      return Status::Ok();
    }
    cudaError_t err = cudaMemcpyAsync(data(), src, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaMemcpyAsync(H2D) failed: ") +
                                  cudaGetErrorString(err));
    }
    return Status::Ok();
  }

  Status CopyToHost(void *dst, size_t bytes, cudaStream_t stream = 0) const {
    if (!defined()) {
      return Status::InvalidArgument("Tensor is undefined");
    }
    if (bytes > this->bytes()) {
      return Status::InvalidArgument("Host copy exceeds tensor size");
    }
    if (bytes == 0) {
      return Status::Ok();
    }
    cudaError_t err = cudaMemcpyAsync(dst, data(), bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaMemcpyAsync(D2H) failed: ") +
                                  cudaGetErrorString(err));
    }
    return Status::Ok();
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
  cudaError_t err = cudaMemcpyAsync(dst->data(), src.data(), src.bytes(), cudaMemcpyDeviceToDevice,
                                    stream);
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaMemcpyAsync(D2D) failed: ") +
                                cudaGetErrorString(err));
  }
  return Status::Ok();
}

inline Result<Tensor> CloneLike(const Tensor &src) {
  auto out = Tensor::Allocate(src.shape(), src.dtype(), src.device());
  if (!out.ok()) {
    return out;
  }
  return out;
}

inline Status EnsureTensor(Tensor *tensor, const std::vector<int64_t> &shape,
                           DType dtype,
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

inline std::string ShapeString(const Tensor &tensor) {
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
