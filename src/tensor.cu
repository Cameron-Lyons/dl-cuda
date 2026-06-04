#include "dl_cuda/tensor.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>

namespace dlcuda {

struct Tensor::Storage {
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

Tensor::~Tensor() = default;

Result<Tensor> Tensor::Allocate(const std::vector<int64_t> &shape, DType dtype, DeviceType device) {
  return AllocateImpl(shape, dtype, device, 0, false);
}

Result<Tensor> Tensor::AllocateAsync(const std::vector<int64_t> &shape, DType dtype,
                                     cudaStream_t stream, DeviceType device) {
  return AllocateImpl(shape, dtype, device, stream, true);
}

Result<Tensor> Tensor::AllocateImpl(const std::vector<int64_t> &shape, DType dtype,
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

void *Tensor::data() const {
  return storage_ ? storage_->ptr : nullptr;
}

size_t Tensor::bytes() const {
  if (!defined()) {
    return 0;
  }
  return storage_->bytes;
}

DeviceType Tensor::device() const {
  return storage_ ? storage_->device : DeviceType::kCuda;
}

Status Tensor::FillZero(cudaStream_t stream) {
  if (!defined()) {
    return Status::InvalidArgument("Tensor is undefined");
  }
  if (bytes() == 0) {
    return Status::Ok();
  }
  cudaError_t err = cudaMemsetAsync(data(), 0, bytes(), stream);
  return detail::CudaStatus(err, "cudaMemsetAsync");
}

Status Tensor::CopyFromHost(const void *src, size_t bytes, cudaStream_t stream) {
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

Status Tensor::CopyToHost(void *dst, size_t bytes, cudaStream_t stream) const {
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

Status Tensor::CopyRangeToHost(void *dst, size_t offset_bytes, size_t bytes,
                               cudaStream_t stream) const {
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

} // namespace dlcuda
