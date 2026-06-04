#pragma once

#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/dtype.hpp"
#include "dl_cuda/status.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace dlcuda {

enum class DeviceType {
  kCuda = 0,
};

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
    return AllocateImpl(shape, dtype, device, 0, false);
  }

  static Result<Tensor> AllocateAsync(const std::vector<int64_t> &shape, DType dtype,
                                      cudaStream_t stream = 0,
                                      DeviceType device = DeviceType::kCuda) {
    return AllocateImpl(shape, dtype, device, stream, true);
  }

private:
  static Result<Tensor> AllocateImpl(const std::vector<int64_t> &shape, DType dtype,
                                     DeviceType device, cudaStream_t stream, bool stream_ordered) {
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
    bool async_free = false;
    if (bytes > 0) {
      cudaError_t err = cudaSuccess;
      const char *alloc_context = "cudaMalloc";
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
      if (stream_ordered) {
        alloc_context = "cudaMallocAsync";
        err = cudaMallocAsync(&ptr, bytes, stream);
        if (err == cudaSuccess) {
          async_free = true;
        } else if (err == cudaErrorNotSupported || err == cudaErrorInsufficientDriver) {
          (void)cudaGetLastError();
          alloc_context = "cudaMalloc";
          err = cudaMalloc(&ptr, bytes);
        }
      } else
#else
      (void)stream;
      (void)stream_ordered;
#endif
      {
        err = cudaMalloc(&ptr, bytes);
      }
      DLCUDA_RETURN_IF_ERROR(detail::CudaStatus(err, alloc_context));
    }

    Tensor out;
    out.storage_ = std::make_shared<Storage>(ptr, bytes, device, stream, async_free);
    out.shape_ = shape;
    out.numel_ = numel;
    out.dtype_ = dtype;
    return out;
  }

public:
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
    Storage(void *ptr_in, size_t bytes_in, DeviceType device_in, cudaStream_t stream_in,
            bool async_free_in)
        : ptr(ptr_in), bytes(bytes_in), device(device_in), stream(stream_in),
          async_free(async_free_in) {}

    ~Storage() {
      if (ptr != nullptr) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        if (async_free) {
          cudaError_t err = cudaFreeAsync(ptr, stream);
          if (err == cudaSuccess) {
            return;
          }
          (void)cudaGetLastError();
        }
#endif
        cudaFree(ptr);
      }
    }

    void *ptr = nullptr;
    size_t bytes = 0;
    DeviceType device = DeviceType::kCuda;
    cudaStream_t stream = 0;
    bool async_free = false;
  };

  std::shared_ptr<Storage> storage_;
  std::vector<int64_t> shape_;
  int64_t numel_ = 1;
  DType dtype_ = DType::kFloat32;
};

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

inline Status EnsureTensorAsync(Tensor *tensor, const std::vector<int64_t> &shape, DType dtype,
                                cudaStream_t stream = 0, DeviceType device = DeviceType::kCuda) {
  if (tensor == nullptr) {
    return Status::InvalidArgument("EnsureTensorAsync received null tensor");
  }
  if (tensor->defined() && tensor->shape() == shape && tensor->dtype() == dtype &&
      tensor->device() == device) {
    return Status::Ok();
  }
  auto allocated = Tensor::AllocateAsync(shape, dtype, stream, device);
  if (!allocated.ok()) {
    return allocated.status();
  }
  *tensor = allocated.value();
  return Status::Ok();
}

} // namespace dlcuda
