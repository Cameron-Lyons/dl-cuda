#pragma once

#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <cublas_v2.h>
#if defined(DLCUDA_HAS_CUBLASLT)
#include <cublasLt.h>
#endif
#include <cuda_runtime.h>

#include <cstdint>
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
  explicit RuntimeContext(const RuntimeOptions &options = RuntimeOptions()) : options_(options) {}

  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext &operator=(const RuntimeContext &) = delete;

  ~RuntimeContext() {
    scratch_tensors_.clear();
    if (cublas_handle_ != nullptr) {
      cublasDestroy(cublas_handle_);
      cublas_handle_ = nullptr;
    }
#if defined(DLCUDA_HAS_CUBLASLT)
    if (cublaslt_handle_ != nullptr) {
      cublasLtDestroy(cublaslt_handle_);
      cublaslt_handle_ = nullptr;
    }
#endif
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
    DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(create_status, "cublasCreate"));
    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, options_.stream);
    Status stream_result = detail::CublasStatus(stream_status, "cublasSetStream");
    if (!stream_result.ok()) {
      (void)ReleaseCublas();
      return stream_result;
    }
    Status math_result = ApplyMathMode();
    if (!math_result.ok()) {
      (void)ReleaseCublas();
      return math_result;
    }
    return Status::Ok();
  }

#if defined(DLCUDA_HAS_CUBLASLT)
  Status EnsureCublasLt() {
    if (cublaslt_handle_ != nullptr) {
      return Status::Ok();
    }
    cublasStatus_t create_status = cublasLtCreate(&cublaslt_handle_);
    return detail::CublasStatus(create_status, "cublasLtCreate");
  }
#endif

  [[nodiscard]] bool use_cublas() const {
    return options_.use_cublas;
  }

  [[nodiscard]] bool tf32() const {
    return options_.tf32;
  }

  [[nodiscard]] cudaStream_t stream() const {
    return options_.stream;
  }

  Status Synchronize() {
    cudaError_t status = cudaStreamSynchronize(options_.stream);
    return detail::CudaStatus(status, "cudaStreamSynchronize");
  }

  [[nodiscard]] cublasHandle_t cublas_handle() const {
    return cublas_handle_;
  }

#if defined(DLCUDA_HAS_CUBLASLT)
  [[nodiscard]] cublasLtHandle_t cublaslt_handle() const {
    return cublaslt_handle_;
  }
#endif

  [[nodiscard]] uint64_t NextInitSeed() {
    ++seed_counter_;
    return options_.seed + 9973ULL * seed_counter_;
  }

  Result<Tensor> ScratchTensor(const std::string &key, const std::vector<int64_t> &shape,
                               DType dtype, DeviceType device = DeviceType::kCuda) {
    auto it = scratch_tensors_.find(key);
    if (it == scratch_tensors_.end() || it->second.shape() != shape ||
        it->second.dtype() != dtype || it->second.device() != device) {
      auto tensor = Tensor::AllocateAsync(shape, dtype, options_.stream, device);
      if (!tensor.ok()) {
        return tensor.status();
      }
      it = scratch_tensors_.insert_or_assign(key, tensor.value()).first;
    }
    return it->second;
  }

private:
  Status ReleaseCublas() {
    if (cublas_handle_ == nullptr) {
      return Status::Ok();
    }
    cublasHandle_t handle = cublas_handle_;
    cublas_handle_ = nullptr;
    cublasStatus_t status = cublasDestroy(handle);
    return detail::CublasStatus(status, "cublasDestroy");
  }

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
    return detail::CublasStatus(status, "cublasSetMathMode");
  }

  RuntimeOptions options_;
  cublasHandle_t cublas_handle_ = nullptr;
#if defined(DLCUDA_HAS_CUBLASLT)
  cublasLtHandle_t cublaslt_handle_ = nullptr;
#endif
  uint64_t seed_counter_ = 0ULL;
  std::unordered_map<std::string, Tensor> scratch_tensors_;
};

} // namespace dlcuda
