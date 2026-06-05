#pragma once

#include "dl_cuda/optim.hpp"

#include "dl_cuda/detail/checkpoint_io.hpp"
#include "dl_cuda/detail/cuda_dtype.cuh"
#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dlcuda {
namespace {

[[maybe_unused]] constexpr int kOptimizerThreads = 256;
[[maybe_unused]] constexpr int kNormReductionMaxBlocks = 4096;
[[maybe_unused]] constexpr double kPi = 3.14159265358979323846;
[[maybe_unused]] constexpr const char kOptimizerMagic[] = "DLCUDAOPT1";
[[maybe_unused]] constexpr uint32_t kOptimizerCheckpointVersion = 1;

using OptimizerBlockReduce = cub::BlockReduce<float, kOptimizerThreads>;

} // namespace

using detail::CloseFile;
using detail::CopyDeviceToHost;
using detail::FilePtr;
using detail::ReadExact;
using detail::WriteExact;

struct HostTensorRecord {
  DType dtype = DType::kFloat32;
  std::vector<int64_t> shape;
  std::vector<char> bytes;
};

Status ValidatePositiveFinite(float value, const char *name);
Status ValidateNonNegativeFinite(float value, const char *name);
Status ValidateRate(float value, const char *name);
Status ValidateParameterOnly(const ParameterRef &param, const char *op_name);
Status ValidateGradient(const ParameterRef &param, const char *op_name);
Status ValidateParameterAndGradient(const ParameterRef &param, const char *op_name);
Status ValidateOptimizerParamGroups(const std::vector<OptimizerParamGroup> &groups);
Status ValidateCheckpointParameters(const std::vector<ParameterRef> &params);
Result<uint64_t> TensorByteSizeForShape(DType dtype, const std::vector<int64_t> &shape);
Status ValidateStateTensorRefs(const std::vector<Optimizer::StateTensorRef> &states);
Status ValidateHyperparameterRefs(const std::vector<Optimizer::Hyperparameter> &hyperparameters);
Status ValidateLoadedHyperparameters(
    const std::vector<Optimizer::Hyperparameter> &expected_hyperparameters,
    const std::vector<Optimizer::Hyperparameter> &loaded_hyperparameters);
Status WriteString(FILE *file, const std::string &text);
Status ReadString(FILE *file, std::string *text);
Status CopyHostToDevice(RuntimeContext &ctx, const std::vector<char> &src, Tensor *dst);
Status WriteParamGroups(FILE *file, const std::vector<OptimizerParamGroup> &groups);
Status ReadParamGroups(FILE *file, std::vector<OptimizerParamGroup> *groups);
Status WriteHyperparameters(FILE *file,
                            const std::vector<Optimizer::Hyperparameter> &hyperparameters);
Status ReadHyperparameters(FILE *file, std::vector<Optimizer::Hyperparameter> *hyperparameters);
Status WriteStateTensors(RuntimeContext &ctx, FILE *file,
                         const std::vector<Optimizer::StateTensorRef> &states);
Status ReadStateTensors(FILE *file, std::unordered_map<std::string, HostTensorRecord> *records);
Status RestoreStateTensors(RuntimeContext &ctx,
                           const std::unordered_map<std::string, HostTensorRecord> &records,
                           const std::vector<Optimizer::StateTensorRef> &states);
Result<Tensor> ScratchTensorForBytes(RuntimeContext &ctx, const std::string &key, size_t bytes);
Status EnsureStateMap(RuntimeContext &ctx, const std::vector<ParameterRef> &params,
                      std::unordered_map<const Tensor *, Tensor> *state);
void ClearStateMap(std::unordered_map<const Tensor *, Tensor> *state);
std::string StateName(const ParameterRef &param, const char *suffix);

template <typename ParamCodec, typename Launcher>
Status DispatchOptimizerGradDType(const ParameterRef &param, const char *op_name,
                                  Launcher &&launcher) {
  switch (param.grad->dtype()) {
  case DType::kFloat32:
    return std::forward<Launcher>(launcher).template operator()<ParamCodec, detail::Float32Codec>();
  case DType::kFloat16:
    return std::forward<Launcher>(launcher).template operator()<ParamCodec, detail::Float16Codec>();
  case DType::kBFloat16:
    return std::forward<Launcher>(launcher)
        .template operator()<ParamCodec, detail::BFloat16Codec>();
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument(std::string(op_name) + " does not support grad dtype " +
                                 std::string(DTypeName(param.grad->dtype())));
}

template <typename Launcher>
Status DispatchOptimizerParamGradDTypes(const ParameterRef &param, const char *op_name,
                                        Launcher &&launcher) {
  switch (param.value->dtype()) {
  case DType::kFloat32:
    return DispatchOptimizerGradDType<detail::Float32Codec>(param, op_name,
                                                            std::forward<Launcher>(launcher));
  case DType::kFloat16:
    return DispatchOptimizerGradDType<detail::Float16Codec>(param, op_name,
                                                            std::forward<Launcher>(launcher));
  case DType::kBFloat16:
    return DispatchOptimizerGradDType<detail::BFloat16Codec>(param, op_name,
                                                             std::forward<Launcher>(launcher));
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument(std::string(op_name) + " does not support parameter dtype " +
                                 std::string(DTypeName(param.value->dtype())));
}

} // namespace dlcuda
