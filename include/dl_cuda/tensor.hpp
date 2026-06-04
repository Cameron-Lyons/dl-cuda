#pragma once

#include "dl_cuda/detail/cuda_forward.hpp"
#include "dl_cuda/dtype.hpp"
#include "dl_cuda/status.hpp"

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
  ~Tensor();

  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor &operator=(Tensor &&) noexcept = default;

  static Result<Tensor> Allocate(const std::vector<int64_t> &shape, DType dtype,
                                 DeviceType device = DeviceType::kCuda);
  static Result<Tensor> AllocateAsync(const std::vector<int64_t> &shape, DType dtype,
                                      cudaStream_t stream = 0,
                                      DeviceType device = DeviceType::kCuda);

  [[nodiscard]] bool defined() const {
    return storage_ != nullptr;
  }

  [[nodiscard]] void *data() const;

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

  [[nodiscard]] size_t bytes() const;

  [[nodiscard]] DType dtype() const {
    return dtype_;
  }

  [[nodiscard]] DeviceType device() const;

  [[nodiscard]] Result<Tensor> Reshape(const std::vector<int64_t> &shape) const {
    if (!defined()) {
      return Status::InvalidArgument("Cannot reshape an undefined tensor");
    }
    auto numel_result = ShapeNumel(shape);
    if (!numel_result.ok()) {
      return numel_result.status();
    }
    if (numel_result.value() != numel_) {
      return Status::InvalidArgument("Reshape element count must match tensor element count");
    }

    Tensor out;
    out.storage_ = storage_;
    out.shape_ = shape;
    out.numel_ = numel_result.value();
    out.dtype_ = dtype_;
    return out;
  }

  Status FillZero(cudaStream_t stream = 0);
  Status CopyFromHost(const void *src, size_t bytes, cudaStream_t stream = 0);
  Status CopyToHost(void *dst, size_t bytes, cudaStream_t stream = 0) const;
  Status CopyRangeToHost(void *dst, size_t offset_bytes, size_t bytes,
                         cudaStream_t stream = 0) const;

private:
  struct Storage;

  static Result<Tensor> AllocateImpl(const std::vector<int64_t> &shape, DType dtype,
                                     DeviceType device, cudaStream_t stream, bool stream_ordered);

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
