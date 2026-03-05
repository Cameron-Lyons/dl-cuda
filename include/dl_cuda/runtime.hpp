#pragma once

#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace dlcuda {

struct RuntimeOptions {
  bool use_cublas = true;
  bool tf32 = true;
  uint64_t seed = 12345ULL;
  cudaStream_t stream = 0;
};

class RuntimeContext {
public:
  explicit RuntimeContext(const RuntimeOptions &options = RuntimeOptions())
      : options_(options), host_rng_(static_cast<uint32_t>(options.seed)) {}

  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext &operator=(const RuntimeContext &) = delete;

  ~RuntimeContext() {
    if (cublas_handle_ != nullptr) {
      cublasDestroy(cublas_handle_);
      cublas_handle_ = nullptr;
    }
  }

  Status Initialize() {
    if (!options_.use_cublas) {
      return Status::Ok();
    }
    return EnsureCublas();
  }

  Status EnsureCublas() {
    if (cublas_handle_ != nullptr) {
      return Status::Ok();
    }
    cublasStatus_t create_status = cublasCreate(&cublas_handle_);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
      return Status::RuntimeError("cublasCreate failed");
    }
    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, options_.stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
      return Status::RuntimeError("cublasSetStream failed");
    }
    return ApplyMathMode();
  }

  bool use_cublas() const { return options_.use_cublas; }

  Status SetUseCuBLAS(bool enabled) {
    options_.use_cublas = enabled;
    if (enabled) {
      return EnsureCublas();
    }
    return Status::Ok();
  }

  bool tf32_enabled() const { return options_.tf32; }

  Status SetTF32(bool enabled) {
    options_.tf32 = enabled;
    if (!options_.use_cublas || cublas_handle_ == nullptr) {
      return Status::Ok();
    }
    return ApplyMathMode();
  }

  cudaStream_t stream() const { return options_.stream; }

  Status SetStream(cudaStream_t stream) {
    options_.stream = stream;
    if (cublas_handle_ != nullptr) {
      cublasStatus_t stream_status = cublasSetStream(cublas_handle_, options_.stream);
      if (stream_status != CUBLAS_STATUS_SUCCESS) {
        return Status::RuntimeError("cublasSetStream failed");
      }
    }
    return Status::Ok();
  }

  Status Synchronize() {
    cudaError_t status = cudaStreamSynchronize(options_.stream);
    if (status != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                  cudaGetErrorString(status));
    }
    return Status::Ok();
  }

  cublasHandle_t cublas_handle() const { return cublas_handle_; }

  uint64_t seed() const { return options_.seed; }

  uint64_t NextInitSeed() {
    ++seed_counter_;
    return options_.seed + 9973ULL * seed_counter_;
  }

  Result<Tensor> ScratchTensor(const std::string &key,
                               const std::vector<int64_t> &shape, DType dtype,
                               DeviceType device = DeviceType::kCuda) {
    auto it = scratch_tensors_.find(key);
    if (it == scratch_tensors_.end() || it->second.shape() != shape ||
        it->second.dtype() != dtype || it->second.device() != device) {
      auto tensor = Tensor::Allocate(shape, dtype, device);
      if (!tensor.ok()) {
        return tensor.status();
      }
      it = scratch_tensors_.insert_or_assign(key, tensor.value()).first;
    }
    return it->second;
  }

  std::mt19937 &host_rng() { return host_rng_; }

private:
  Status ApplyMathMode() {
    if (cublas_handle_ == nullptr) {
      return Status::RuntimeError("ApplyMathMode called before cublasCreate");
    }
#if defined(CUBLAS_TF32_TENSOR_OP_MATH)
    cublasMath_t mode = options_.tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
#else
    cublasMath_t mode = CUBLAS_DEFAULT_MATH;
#endif
    cublasStatus_t status = cublasSetMathMode(cublas_handle_, mode);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return Status::RuntimeError("cublasSetMathMode failed");
    }
    return Status::Ok();
  }

  RuntimeOptions options_;
  cublasHandle_t cublas_handle_ = nullptr;
  uint64_t seed_counter_ = 0ULL;
  std::mt19937 host_rng_;
  std::unordered_map<std::string, Tensor> scratch_tensors_;
};

} // namespace dlcuda
