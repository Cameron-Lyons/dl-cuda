#pragma once

#include "dl_cuda/detail/cuda_forward.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

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
  explicit RuntimeContext(const RuntimeOptions &options = RuntimeOptions());

  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext &operator=(const RuntimeContext &) = delete;

  ~RuntimeContext();

  Status Initialize();
  Status EnsureCublas();

#if defined(DLCUDA_HAS_CUBLASLT)
  Status EnsureCublasLt();
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

  Status Synchronize();

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
                               DType dtype, DeviceType device = DeviceType::kCuda);

private:
  Status ReleaseCublas();
  Status ApplyMathMode();

  RuntimeOptions options_;
  cublasHandle_t cublas_handle_ = nullptr;
#if defined(DLCUDA_HAS_CUBLASLT)
  cublasLtHandle_t cublaslt_handle_ = nullptr;
#endif
  uint64_t seed_counter_ = 0ULL;
  std::unordered_map<std::string, Tensor> scratch_tensors_;
};

} // namespace dlcuda
