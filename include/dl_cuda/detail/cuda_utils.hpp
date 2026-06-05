#pragma once

#include "dl_cuda/dtype.hpp"
#include "dl_cuda/status.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace dlcuda::detail {

inline Status CudaStatus(cudaError_t err, const std::string &context) {
  if (err == cudaSuccess) {
    return Status::Ok();
  }
  return Status::RuntimeError(context + ": " + cudaGetErrorString(err));
}

inline Status CublasStatus(cublasStatus_t status, const std::string &context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  std::ostringstream oss;
  oss << context << " failed with cuBLAS status code " << static_cast<int>(status);
  return Status::RuntimeError(oss.str());
}

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000

inline Result<cudaDataType_t> CublasCudaDataType(DType dtype, const std::string &context) {
  switch (dtype) {
  case DType::kFloat32:
    return CUDA_R_32F;
  case DType::kFloat16:
    return CUDA_R_16F;
  case DType::kBFloat16:
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
    return CUDA_R_16BF;
#else
    return Status::Unsupported("bfloat16 " + context + " requires CUDA 11 or newer");
#endif
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument(context + " does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

inline cublasComputeType_t CublasComputeType(bool use_tf32, DType dtype) {
#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
  if (dtype == DType::kFloat32 && use_tf32) {
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  }
#else
  (void)use_tf32;
#endif
  (void)dtype;
  return CUBLAS_COMPUTE_32F;
}

#endif

inline Status CheckKernelLaunch(const std::string &context) {
  return CudaStatus(cudaGetLastError(), context);
}

inline Result<int> CheckedInt(int64_t value, const char *name) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << name << " is outside int range: " << value;
    return Status::InvalidArgument(oss.str());
  }
  return static_cast<int>(value);
}

inline Result<int> RowsForGrid(int64_t rows, const char *name) {
  if (rows < 0 || rows > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << name << " row count is outside CUDA grid range: " << rows;
    return Status::InvalidArgument(oss.str());
  }
  return static_cast<int>(rows);
}

inline Result<int> BlocksForElements(int64_t elements, int threads = 256) {
  if (elements < 0) {
    return Status::InvalidArgument("element count must be non-negative");
  }
  if (threads <= 0) {
    return Status::InvalidArgument("thread count must be positive");
  }
  int64_t blocks = elements == 0 ? 0 : 1 + ((elements - 1) / threads);
  if (blocks > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "block count is outside CUDA grid range: " << blocks;
    return Status::InvalidArgument(oss.str());
  }
  return static_cast<int>(blocks);
}

inline Result<int> CappedBlocksForElements(int64_t elements, int threads, int max_blocks) {
  if (elements < 0) {
    return Status::InvalidArgument("element count must be non-negative");
  }
  if (threads <= 0) {
    return Status::InvalidArgument("thread count must be positive");
  }
  if (max_blocks <= 0) {
    return Status::InvalidArgument("max block count must be positive");
  }
  int64_t blocks = elements == 0 ? 0 : 1 + ((elements - 1) / threads);
  if (blocks > static_cast<int64_t>(max_blocks)) {
    blocks = max_blocks;
  }
  return static_cast<int>(blocks);
}

} // namespace dlcuda::detail
