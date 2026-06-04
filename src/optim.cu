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

constexpr int kOptimizerThreads = 256;
constexpr int kNormReductionMaxBlocks = 4096;
constexpr double kPi = 3.14159265358979323846;
constexpr const char kOptimizerMagic[] = "DLCUDAOPT1";
constexpr uint32_t kOptimizerCheckpointVersion = 1;

using OptimizerBlockReduce = cub::BlockReduce<float, kOptimizerThreads>;
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

Status ValidatePositiveFinite(float value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0f)) {
    return Status::InvalidArgument(std::string(name) + " must be finite and > 0");
  }
  return Status::Ok();
}

Status ValidateNonNegativeFinite(float value, const char *name) {
  if (!std::isfinite(value) || value < 0.0f) {
    return Status::InvalidArgument(std::string(name) + " must be finite and >= 0");
  }
  return Status::Ok();
}

Status ValidateRate(float value, const char *name) {
  if (!std::isfinite(value) || value < 0.0f || value >= 1.0f) {
    return Status::InvalidArgument(std::string(name) + " must be finite and in [0, 1)");
  }
  return Status::Ok();
}

Status ValidateParameterOnly(const ParameterRef &param, const char *op_name) {
  if (param.value == nullptr || !param.value->defined()) {
    return Status::InvalidArgument(std::string(op_name) + ": undefined parameter for " +
                                   param.name);
  }
  if (!IsFloatingPointDType(param.value->dtype())) {
    return Status::InvalidArgument(std::string(op_name) +
                                   " only supports floating-point parameters");
  }
  return Status::Ok();
}

Status ValidateGradient(const ParameterRef &param, const char *op_name) {
  if (param.grad == nullptr || !param.grad->defined()) {
    return Status::InvalidArgument(std::string(op_name) + ": undefined grad tensor for " +
                                   param.name);
  }
  if (!IsFloatingPointDType(param.grad->dtype())) {
    return Status::InvalidArgument(std::string(op_name) + " only supports floating-point grads");
  }
  return Status::Ok();
}

Status ValidateParameterAndGradient(const ParameterRef &param, const char *op_name) {
  DLCUDA_RETURN_IF_ERROR(ValidateParameterOnly(param, op_name));
  DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, op_name));
  if (param.value->shape() != param.grad->shape()) {
    return Status::InvalidArgument(std::string(op_name) + " shape mismatch for " + param.name);
  }
  return Status::Ok();
}

Status ValidateOptimizerParamGroups(const std::vector<OptimizerParamGroup> &groups) {
  if (groups.empty()) {
    return Status::InvalidArgument("Optimizer requires at least one parameter group");
  }

  bool has_default_group = false;
  std::unordered_set<std::string> seen_names;
  for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
    const auto &group = groups[group_index];
    DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(group.lr, "Optimizer parameter group lr"));
    DLCUDA_RETURN_IF_ERROR(
        ValidateNonNegativeFinite(group.weight_decay, "Optimizer parameter group weight_decay"));
    if (group.parameter_names.empty()) {
      if (has_default_group) {
        return Status::InvalidArgument("Only one catch-all optimizer parameter group is allowed");
      }
      has_default_group = true;
      continue;
    }

    for (const auto &name : group.parameter_names) {
      if (name.empty()) {
        return Status::InvalidArgument("Optimizer parameter group names must be non-empty");
      }
      if (!seen_names.insert(name).second) {
        return Status::InvalidArgument("Duplicate optimizer parameter group entry: " + name);
      }
    }
  }

  return Status::Ok();
}

Status ValidateCheckpointParameters(const std::vector<ParameterRef> &params) {
  std::unordered_set<std::string> seen_names;
  seen_names.reserve(params.size());
  for (const auto &param : params) {
    if (param.name.empty()) {
      return Status::InvalidArgument("Optimizer checkpoint parameter names must be non-empty");
    }
    if (!seen_names.insert(param.name).second) {
      return Status::InvalidArgument("Duplicate optimizer checkpoint parameter: " + param.name);
    }
    DLCUDA_RETURN_IF_ERROR(ValidateParameterOnly(param, "Optimizer checkpoint"));
  }
  return Status::Ok();
}

Result<uint64_t> TensorByteSizeForShape(DType dtype, const std::vector<int64_t> &shape) {
  return detail::TensorByteSizeForShape(dtype, shape, "optimizer checkpoint tensor");
}

Status ValidateStateTensorRefs(const std::vector<Optimizer::StateTensorRef> &states) {
  std::unordered_set<std::string> seen_names;
  seen_names.reserve(states.size());
  for (const auto &state : states) {
    if (state.name.empty()) {
      return Status::InvalidArgument("Optimizer state tensor names must be non-empty");
    }
    if (!seen_names.insert(state.name).second) {
      return Status::InvalidArgument("Duplicate optimizer state tensor: " + state.name);
    }
    if (state.tensor == nullptr || !state.tensor->defined()) {
      return Status::InvalidArgument("Undefined optimizer state tensor: " + state.name);
    }
    auto expected_bytes = TensorByteSizeForShape(state.tensor->dtype(), state.tensor->shape());
    if (!expected_bytes.ok()) {
      return expected_bytes.status();
    }
    if (expected_bytes.value() != static_cast<uint64_t>(state.tensor->bytes())) {
      return Status::InvalidArgument("Optimizer state tensor byte size mismatch: " + state.name);
    }
  }
  return Status::Ok();
}

Status ValidateHyperparameterRefs(const std::vector<Optimizer::Hyperparameter> &hyperparameters) {
  std::unordered_set<std::string> seen_names;
  seen_names.reserve(hyperparameters.size());
  for (const auto &hyperparameter : hyperparameters) {
    if (hyperparameter.name.empty()) {
      return Status::InvalidArgument("Optimizer hyperparameter names must be non-empty");
    }
    if (!seen_names.insert(hyperparameter.name).second) {
      return Status::InvalidArgument("Duplicate optimizer hyperparameter: " + hyperparameter.name);
    }
    if (!std::isfinite(hyperparameter.value)) {
      return Status::InvalidArgument("Optimizer hyperparameter must be finite: " +
                                     hyperparameter.name);
    }
  }
  return Status::Ok();
}

Status ValidateLoadedHyperparameters(
    const std::vector<Optimizer::Hyperparameter> &expected_hyperparameters,
    const std::vector<Optimizer::Hyperparameter> &loaded_hyperparameters) {
  if (expected_hyperparameters.size() != loaded_hyperparameters.size()) {
    return Status::InvalidArgument("Optimizer checkpoint hyperparameter count mismatch");
  }
  std::unordered_map<std::string, float> loaded;
  loaded.reserve(loaded_hyperparameters.size());
  for (const auto &hyperparameter : loaded_hyperparameters) {
    loaded.emplace(hyperparameter.name, hyperparameter.value);
  }
  for (const auto &expected : expected_hyperparameters) {
    auto it = loaded.find(expected.name);
    if (it == loaded.end()) {
      return Status::NotFound("Missing optimizer checkpoint hyperparameter: " + expected.name);
    }
    if (it->second != expected.value) {
      return Status::InvalidArgument("Optimizer checkpoint hyperparameter mismatch: " +
                                     expected.name);
    }
  }
  return Status::Ok();
}

Status WriteString(FILE *file, const std::string &text) {
  return detail::WriteString(file, text, "optimizer checkpoint");
}

Status ReadString(FILE *file, std::string *text) {
  return detail::ReadString(file, text, "optimizer checkpoint");
}

Status CopyHostToDevice(RuntimeContext &ctx, const std::vector<char> &src, Tensor *dst) {
  return detail::CopyHostToDevice(ctx, src, dst, "Optimizer checkpoint tensor byte size mismatch");
}

Status WriteParamGroups(FILE *file, const std::vector<OptimizerParamGroup> &groups) {
  if (groups.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument("Too many optimizer parameter groups to checkpoint");
  }
  uint32_t group_count = static_cast<uint32_t>(groups.size());
  if (!WriteExact(file, &group_count, sizeof(group_count))) {
    return Status::IoError("Failed to write optimizer parameter group count");
  }
  for (const auto &group : groups) {
    if (!WriteExact(file, &group.lr, sizeof(group.lr)) ||
        !WriteExact(file, &group.weight_decay, sizeof(group.weight_decay))) {
      return Status::IoError("Failed to write optimizer parameter group scalars");
    }
    if (group.parameter_names.size() > std::numeric_limits<uint32_t>::max()) {
      return Status::InvalidArgument("Too many names in optimizer parameter group");
    }
    uint32_t name_count = static_cast<uint32_t>(group.parameter_names.size());
    if (!WriteExact(file, &name_count, sizeof(name_count))) {
      return Status::IoError("Failed to write optimizer parameter group name count");
    }
    for (const auto &name : group.parameter_names) {
      DLCUDA_RETURN_IF_ERROR(WriteString(file, name));
    }
  }
  return Status::Ok();
}

Status ReadParamGroups(FILE *file, std::vector<OptimizerParamGroup> *groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("ReadParamGroups destination is null");
  }
  uint32_t group_count = 0;
  if (!ReadExact(file, &group_count, sizeof(group_count))) {
    return Status::IoError("Failed to read optimizer parameter group count");
  }
  groups->clear();
  groups->resize(group_count);
  for (auto &group : *groups) {
    if (!ReadExact(file, &group.lr, sizeof(group.lr)) ||
        !ReadExact(file, &group.weight_decay, sizeof(group.weight_decay))) {
      return Status::IoError("Failed to read optimizer parameter group scalars");
    }
    uint32_t name_count = 0;
    if (!ReadExact(file, &name_count, sizeof(name_count))) {
      return Status::IoError("Failed to read optimizer parameter group name count");
    }
    group.parameter_names.resize(name_count);
    for (auto &name : group.parameter_names) {
      DLCUDA_RETURN_IF_ERROR(ReadString(file, &name));
    }
  }
  return ValidateOptimizerParamGroups(*groups);
}

Status WriteHyperparameters(FILE *file,
                            const std::vector<Optimizer::Hyperparameter> &hyperparameters) {
  if (hyperparameters.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument("Too many optimizer hyperparameters to checkpoint");
  }
  uint32_t hyperparameter_count = static_cast<uint32_t>(hyperparameters.size());
  if (!WriteExact(file, &hyperparameter_count, sizeof(hyperparameter_count))) {
    return Status::IoError("Failed to write optimizer hyperparameter count");
  }
  for (const auto &hyperparameter : hyperparameters) {
    DLCUDA_RETURN_IF_ERROR(WriteString(file, hyperparameter.name));
    if (!WriteExact(file, &hyperparameter.value, sizeof(hyperparameter.value))) {
      return Status::IoError("Failed to write optimizer hyperparameter value");
    }
  }
  return Status::Ok();
}

Status ReadHyperparameters(FILE *file, std::vector<Optimizer::Hyperparameter> *hyperparameters) {
  if (hyperparameters == nullptr) {
    return Status::InvalidArgument("ReadHyperparameters destination is null");
  }
  uint32_t hyperparameter_count = 0;
  if (!ReadExact(file, &hyperparameter_count, sizeof(hyperparameter_count))) {
    return Status::IoError("Failed to read optimizer hyperparameter count");
  }
  hyperparameters->clear();
  hyperparameters->resize(hyperparameter_count);
  for (auto &hyperparameter : *hyperparameters) {
    DLCUDA_RETURN_IF_ERROR(ReadString(file, &hyperparameter.name));
    if (!ReadExact(file, &hyperparameter.value, sizeof(hyperparameter.value))) {
      return Status::IoError("Failed to read optimizer hyperparameter value");
    }
  }
  return ValidateHyperparameterRefs(*hyperparameters);
}

Status WriteStateTensors(RuntimeContext &ctx, FILE *file,
                         const std::vector<Optimizer::StateTensorRef> &states) {
  if (states.size() > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument("Too many optimizer state tensors to checkpoint");
  }
  uint32_t state_count = static_cast<uint32_t>(states.size());
  if (!WriteExact(file, &state_count, sizeof(state_count))) {
    return Status::IoError("Failed to write optimizer state tensor count");
  }
  for (const auto &state : states) {
    DLCUDA_RETURN_IF_ERROR(WriteString(file, state.name));
    uint32_t dtype = static_cast<uint32_t>(state.tensor->dtype());
    if (!WriteExact(file, &dtype, sizeof(dtype))) {
      return Status::IoError("Failed to write optimizer state dtype");
    }
    if (state.tensor->shape().size() > detail::kMaxCheckpointTensorRank) {
      return Status::InvalidArgument("Optimizer state rank is too large: " + state.name);
    }
    uint32_t rank = static_cast<uint32_t>(state.tensor->shape().size());
    if (!WriteExact(file, &rank, sizeof(rank))) {
      return Status::IoError("Failed to write optimizer state rank");
    }
    if (rank > 0 && !WriteExact(file, state.tensor->shape().data(), rank * sizeof(int64_t))) {
      return Status::IoError("Failed to write optimizer state shape");
    }
    uint64_t bytes = static_cast<uint64_t>(state.tensor->bytes());
    if (!WriteExact(file, &bytes, sizeof(bytes))) {
      return Status::IoError("Failed to write optimizer state byte size");
    }

    std::vector<char> host_data;
    DLCUDA_RETURN_IF_ERROR(CopyDeviceToHost(ctx, *state.tensor, &host_data));
    if (bytes > 0 && !WriteExact(file, host_data.data(), static_cast<size_t>(bytes))) {
      return Status::IoError("Failed to write optimizer state bytes");
    }
  }
  return Status::Ok();
}

Status ReadStateTensors(FILE *file, std::unordered_map<std::string, HostTensorRecord> *records) {
  if (records == nullptr) {
    return Status::InvalidArgument("ReadStateTensors destination is null");
  }
  uint32_t state_count = 0;
  if (!ReadExact(file, &state_count, sizeof(state_count))) {
    return Status::IoError("Failed to read optimizer state tensor count");
  }
  records->clear();
  records->reserve(state_count);
  for (uint32_t i = 0; i < state_count; ++i) {
    std::string name;
    DLCUDA_RETURN_IF_ERROR(ReadString(file, &name));
    if (name.empty()) {
      return Status::InvalidArgument("Optimizer checkpoint state tensor name is empty");
    }
    if (records->find(name) != records->end()) {
      return Status::InvalidArgument("Duplicate optimizer checkpoint state tensor: " + name);
    }

    uint32_t dtype_u32 = 0;
    if (!ReadExact(file, &dtype_u32, sizeof(dtype_u32))) {
      return Status::IoError("Failed to read optimizer state dtype");
    }
    DType dtype = static_cast<DType>(dtype_u32);
    if (DTypeSize(dtype) == 0) {
      return Status::InvalidArgument("Unsupported optimizer checkpoint state dtype");
    }

    uint32_t rank = 0;
    if (!ReadExact(file, &rank, sizeof(rank))) {
      return Status::IoError("Failed to read optimizer state rank");
    }
    if (rank > detail::kMaxCheckpointTensorRank) {
      return Status::InvalidArgument("Optimizer checkpoint state rank is too large");
    }

    HostTensorRecord record;
    record.dtype = dtype;
    record.shape.resize(rank);
    if (rank > 0 && !ReadExact(file, record.shape.data(), rank * sizeof(int64_t))) {
      return Status::IoError("Failed to read optimizer state shape");
    }

    uint64_t bytes = 0;
    if (!ReadExact(file, &bytes, sizeof(bytes))) {
      return Status::IoError("Failed to read optimizer state byte size");
    }
    auto expected_bytes = TensorByteSizeForShape(dtype, record.shape);
    if (!expected_bytes.ok()) {
      return expected_bytes.status();
    }
    if (bytes != expected_bytes.value()) {
      return Status::InvalidArgument("Optimizer checkpoint tensor byte size mismatch");
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return Status::InvalidArgument("Optimizer checkpoint tensor byte size is too large");
    }

    record.bytes.resize(static_cast<size_t>(bytes));
    if (bytes > 0 && !ReadExact(file, record.bytes.data(), static_cast<size_t>(bytes))) {
      return Status::IoError("Failed to read optimizer state bytes");
    }
    records->emplace(name, std::move(record));
  }
  return Status::Ok();
}

Status RestoreStateTensors(RuntimeContext &ctx,
                           const std::unordered_map<std::string, HostTensorRecord> &records,
                           const std::vector<Optimizer::StateTensorRef> &states) {
  if (records.size() != states.size()) {
    return Status::InvalidArgument("Optimizer checkpoint state tensor count mismatch");
  }
  for (const auto &state : states) {
    auto it = records.find(state.name);
    if (it == records.end()) {
      return Status::NotFound("Missing optimizer checkpoint state tensor: " + state.name);
    }
    const HostTensorRecord &record = it->second;
    if (record.dtype != state.tensor->dtype()) {
      return Status::InvalidArgument("Optimizer checkpoint state dtype mismatch: " + state.name);
    }
    if (record.shape != state.tensor->shape()) {
      return Status::InvalidArgument("Optimizer checkpoint state shape mismatch: " + state.name);
    }
    if (record.bytes.size() != state.tensor->bytes()) {
      return Status::InvalidArgument("Optimizer checkpoint state byte size mismatch: " +
                                     state.name);
    }
    DLCUDA_RETURN_IF_ERROR(CopyHostToDevice(ctx, record.bytes, state.tensor));
  }
  return Status::Ok();
}

Status EnsureStateMap(RuntimeContext &ctx, const std::vector<ParameterRef> &params,
                      std::unordered_map<const Tensor *, Tensor> *state) {
  if (state == nullptr) {
    return Status::InvalidArgument("Optimizer state map is null");
  }

  std::unordered_set<const Tensor *> active_params;
  active_params.reserve(params.size());

  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateParameterOnly(param, "Optimizer state"));
    active_params.insert(param.value);

    auto it = state->find(param.value);
    bool needs_init = (it == state->end());
    if (!needs_init) {
      needs_init =
          (it->second.shape() != param.value->shape() || it->second.dtype() != DType::kFloat32);
    }

    if (needs_init) {
      auto tensor = Tensor::AllocateAsync(param.value->shape(), DType::kFloat32, ctx.stream());
      if (!tensor.ok()) {
        return tensor.status();
      }
      it = state->insert_or_assign(param.value, tensor.value()).first;
      DLCUDA_RETURN_IF_ERROR(it->second.FillZero(ctx.stream()));
    }
  }

  for (auto it = state->begin(); it != state->end();) {
    if (active_params.find(it->first) == active_params.end()) {
      it = state->erase(it);
    } else {
      ++it;
    }
  }

  return Status::Ok();
}

void ClearStateMap(std::unordered_map<const Tensor *, Tensor> *state) {
  if (state != nullptr) {
    state->clear();
  }
}

std::string StateName(const ParameterRef &param, const char *suffix) {
  return param.name + "." + suffix;
}

template <typename ParamCodec, typename GradCodec>
__global__ void SGDUpdateKernel(typename ParamCodec::Storage *params,
                                const typename GradCodec::Storage *grads, float *momentum_buffer,
                                bool has_momentum, float lr, float momentum, float weight_decay,
                                float dampening, bool nesterov, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = ParamCodec::Load(params, idx);
    float g = GradCodec::Load(grads, idx);
    if (weight_decay != 0.0f) {
      g += weight_decay * p;
    }

    float update = g;
    if (has_momentum) {
      float buffer = momentum * momentum_buffer[idx] + (1.0f - dampening) * g;
      momentum_buffer[idx] = buffer;
      update = nesterov ? g + momentum * buffer : buffer;
    }

    ParamCodec::Store(params, idx, p - lr * update);
  }
}

template <typename ParamCodec, typename GradCodec>
__global__ void AdamUpdateKernel(typename ParamCodec::Storage *params,
                                 const typename GradCodec::Storage *grads, float *m, float *v,
                                 float lr, float beta1, float beta2, float epsilon,
                                 float inv_bias_correction1, float inv_bias_correction2,
                                 float weight_decay, bool decoupled_weight_decay, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = ParamCodec::Load(params, idx);
    float g = GradCodec::Load(grads, idx);
    if (weight_decay != 0.0f && !decoupled_weight_decay) {
      g += weight_decay * p;
    }

    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    float m_hat = m_new * inv_bias_correction1;
    float v_hat = v_new * inv_bias_correction2;
    float updated = p - lr * (m_hat / (sqrtf(v_hat) + epsilon));
    if (weight_decay != 0.0f && decoupled_weight_decay) {
      updated -= lr * weight_decay * p;
    }
    ParamCodec::Store(params, idx, updated);
  }
}

template <typename ParamCodec, typename GradCodec>
__global__ void RMSPropUpdateKernel(typename ParamCodec::Storage *params,
                                    const typename GradCodec::Storage *grads, float *square_avg,
                                    float *momentum_buffer, float *grad_avg, bool has_momentum,
                                    bool centered, float lr, float alpha, float epsilon,
                                    float momentum, float weight_decay, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = ParamCodec::Load(params, idx);
    float g = GradCodec::Load(grads, idx);
    if (weight_decay != 0.0f) {
      g += weight_decay * p;
    }

    float square = alpha * square_avg[idx] + (1.0f - alpha) * g * g;
    square_avg[idx] = square;

    float avg = square;
    if (centered) {
      float mean = alpha * grad_avg[idx] + (1.0f - alpha) * g;
      grad_avg[idx] = mean;
      avg -= mean * mean;
    }
    float update = g / sqrtf(fmaxf(avg, 0.0f) + epsilon);
    if (has_momentum) {
      float buffer = momentum * momentum_buffer[idx] + update;
      momentum_buffer[idx] = buffer;
      update = buffer;
    }

    ParamCodec::Store(params, idx, p - lr * update);
  }
}

template <typename Codec>
__global__ void AccumulateNormSqKernel(const typename Codec::Storage *grads, int64_t n,
                                       float *total_norm_sq) {
  __shared__ typename OptimizerBlockReduce::TempStorage reduce_storage;
  int tid = threadIdx.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

  float local = 0.0f;
  for (int64_t i = idx; i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float v = Codec::Load(grads, i);
    local += v * v;
  }
  float block_sum = OptimizerBlockReduce(reduce_storage).Sum(local);

  if (tid == 0) {
    atomicAdd(total_norm_sq, block_sum);
  }
}

__global__ void ComputeClipScaleKernel(const float *total_norm_sq, float max_norm,
                                       float *clip_scale) {
  float total_norm = sqrtf(total_norm_sq[0]);
  clip_scale[0] = total_norm > max_norm ? max_norm / (total_norm + 1e-6f) : 1.0f;
}

template <typename Codec>
__global__ void ScaleByFactorKernel(typename Codec::Storage *data, const float *clip_scale,
                                    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    Codec::Store(data, idx, Codec::Load(data, idx) * clip_scale[0]);
  }
}

template <typename ParamCodec, typename GradCodec>
Status LaunchSGDUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *momentum_buffer,
                       bool has_momentum, float lr, float momentum, float weight_decay,
                       float dampening, bool nesterov, int blocks) {
  SGDUpdateKernel<ParamCodec, GradCodec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      param.value->data_as<typename ParamCodec::Storage>(),
      param.grad->data_as<typename GradCodec::Storage>(),
      has_momentum ? momentum_buffer->data_as<float>() : nullptr, has_momentum, lr, momentum,
      weight_decay, dampening, nesterov, param.value->numel());
  return detail::CheckKernelLaunch("SGD update kernel");
}

template <typename ParamCodec>
Status LaunchSGDUpdateForParam(RuntimeContext &ctx, const ParameterRef &param,
                               Tensor *momentum_buffer, bool has_momentum, float lr, float momentum,
                               float weight_decay, float dampening, bool nesterov, int blocks) {
  switch (param.grad->dtype()) {
  case DType::kFloat32:
    return LaunchSGDUpdate<ParamCodec, detail::Float32Codec>(
        ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay, dampening, nesterov,
        blocks);
  case DType::kFloat16:
    return LaunchSGDUpdate<ParamCodec, detail::Float16Codec>(
        ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay, dampening, nesterov,
        blocks);
  case DType::kBFloat16:
    return LaunchSGDUpdate<ParamCodec, detail::BFloat16Codec>(
        ctx, param, momentum_buffer, has_momentum, lr, momentum, weight_decay, dampening, nesterov,
        blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("SGD does not support grad dtype " +
                                 std::string(DTypeName(param.grad->dtype())));
}

Status LaunchSGDUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *momentum_buffer,
                       bool has_momentum, float lr, float momentum, float weight_decay,
                       float dampening, bool nesterov, int blocks) {
  switch (param.value->dtype()) {
  case DType::kFloat32:
    return LaunchSGDUpdateForParam<detail::Float32Codec>(ctx, param, momentum_buffer, has_momentum,
                                                         lr, momentum, weight_decay, dampening,
                                                         nesterov, blocks);
  case DType::kFloat16:
    return LaunchSGDUpdateForParam<detail::Float16Codec>(ctx, param, momentum_buffer, has_momentum,
                                                         lr, momentum, weight_decay, dampening,
                                                         nesterov, blocks);
  case DType::kBFloat16:
    return LaunchSGDUpdateForParam<detail::BFloat16Codec>(ctx, param, momentum_buffer, has_momentum,
                                                          lr, momentum, weight_decay, dampening,
                                                          nesterov, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("SGD does not support parameter dtype " +
                                 std::string(DTypeName(param.value->dtype())));
}

template <typename ParamCodec, typename GradCodec>
Status LaunchAdamUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *m, Tensor *v,
                        float lr, float beta1, float beta2, float epsilon,
                        float inv_bias_correction1, float inv_bias_correction2, float weight_decay,
                        bool decoupled_weight_decay, int blocks) {
  AdamUpdateKernel<ParamCodec, GradCodec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      param.value->data_as<typename ParamCodec::Storage>(),
      param.grad->data_as<typename GradCodec::Storage>(), m->data_as<float>(), v->data_as<float>(),
      lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2, weight_decay,
      decoupled_weight_decay, param.value->numel());
  return detail::CheckKernelLaunch("Adam update kernel");
}

template <typename ParamCodec>
Status LaunchAdamUpdateForParam(RuntimeContext &ctx, const ParameterRef &param, Tensor *m,
                                Tensor *v, float lr, float beta1, float beta2, float epsilon,
                                float inv_bias_correction1, float inv_bias_correction2,
                                float weight_decay, bool decoupled_weight_decay, int blocks) {
  switch (param.grad->dtype()) {
  case DType::kFloat32:
    return LaunchAdamUpdate<ParamCodec, detail::Float32Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kFloat16:
    return LaunchAdamUpdate<ParamCodec, detail::Float16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kBFloat16:
    return LaunchAdamUpdate<ParamCodec, detail::BFloat16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Adam does not support grad dtype " +
                                 std::string(DTypeName(param.grad->dtype())));
}

Status LaunchAdamUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *m, Tensor *v,
                        float lr, float beta1, float beta2, float epsilon,
                        float inv_bias_correction1, float inv_bias_correction2, float weight_decay,
                        bool decoupled_weight_decay, int blocks) {
  switch (param.value->dtype()) {
  case DType::kFloat32:
    return LaunchAdamUpdateForParam<detail::Float32Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kFloat16:
    return LaunchAdamUpdateForParam<detail::Float16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kBFloat16:
    return LaunchAdamUpdateForParam<detail::BFloat16Codec>(
        ctx, param, m, v, lr, beta1, beta2, epsilon, inv_bias_correction1, inv_bias_correction2,
        weight_decay, decoupled_weight_decay, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Adam does not support parameter dtype " +
                                 std::string(DTypeName(param.value->dtype())));
}

template <typename ParamCodec, typename GradCodec>
Status LaunchRMSPropUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *square_avg,
                           Tensor *momentum_buffer, Tensor *grad_avg, bool has_momentum,
                           bool centered, float lr, float alpha, float epsilon, float momentum,
                           float weight_decay, int blocks) {
  RMSPropUpdateKernel<ParamCodec, GradCodec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      param.value->data_as<typename ParamCodec::Storage>(),
      param.grad->data_as<typename GradCodec::Storage>(), square_avg->data_as<float>(),
      has_momentum ? momentum_buffer->data_as<float>() : nullptr,
      centered ? grad_avg->data_as<float>() : nullptr, has_momentum, centered, lr, alpha, epsilon,
      momentum, weight_decay, param.value->numel());
  return detail::CheckKernelLaunch("RMSProp update kernel");
}

template <typename ParamCodec>
Status LaunchRMSPropUpdateForParam(RuntimeContext &ctx, const ParameterRef &param,
                                   Tensor *square_avg, Tensor *momentum_buffer, Tensor *grad_avg,
                                   bool has_momentum, bool centered, float lr, float alpha,
                                   float epsilon, float momentum, float weight_decay, int blocks) {
  switch (param.grad->dtype()) {
  case DType::kFloat32:
    return LaunchRMSPropUpdate<ParamCodec, detail::Float32Codec>(
        ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum, centered, lr, alpha,
        epsilon, momentum, weight_decay, blocks);
  case DType::kFloat16:
    return LaunchRMSPropUpdate<ParamCodec, detail::Float16Codec>(
        ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum, centered, lr, alpha,
        epsilon, momentum, weight_decay, blocks);
  case DType::kBFloat16:
    return LaunchRMSPropUpdate<ParamCodec, detail::BFloat16Codec>(
        ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum, centered, lr, alpha,
        epsilon, momentum, weight_decay, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("RMSProp does not support grad dtype " +
                                 std::string(DTypeName(param.grad->dtype())));
}

Status LaunchRMSPropUpdate(RuntimeContext &ctx, const ParameterRef &param, Tensor *square_avg,
                           Tensor *momentum_buffer, Tensor *grad_avg, bool has_momentum,
                           bool centered, float lr, float alpha, float epsilon, float momentum,
                           float weight_decay, int blocks) {
  switch (param.value->dtype()) {
  case DType::kFloat32:
    return LaunchRMSPropUpdateForParam<detail::Float32Codec>(
        ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum, centered, lr, alpha,
        epsilon, momentum, weight_decay, blocks);
  case DType::kFloat16:
    return LaunchRMSPropUpdateForParam<detail::Float16Codec>(
        ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum, centered, lr, alpha,
        epsilon, momentum, weight_decay, blocks);
  case DType::kBFloat16:
    return LaunchRMSPropUpdateForParam<detail::BFloat16Codec>(
        ctx, param, square_avg, momentum_buffer, grad_avg, has_momentum, centered, lr, alpha,
        epsilon, momentum, weight_decay, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("RMSProp does not support parameter dtype " +
                                 std::string(DTypeName(param.value->dtype())));
}

template <typename Codec>
Status LaunchAccumulateNormSq(RuntimeContext &ctx, const Tensor &grad, Tensor *total_norm_sq_buffer,
                              int blocks) {
  AccumulateNormSqKernel<Codec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      grad.data_as<typename Codec::Storage>(), grad.numel(),
      total_norm_sq_buffer->data_as<float>());
  return detail::CheckKernelLaunch("AccumulateNormSqKernel");
}

Status LaunchAccumulateNormSq(RuntimeContext &ctx, const Tensor &grad, Tensor *total_norm_sq_buffer,
                              int blocks) {
  switch (grad.dtype()) {
  case DType::kFloat32:
    return LaunchAccumulateNormSq<detail::Float32Codec>(ctx, grad, total_norm_sq_buffer, blocks);
  case DType::kFloat16:
    return LaunchAccumulateNormSq<detail::Float16Codec>(ctx, grad, total_norm_sq_buffer, blocks);
  case DType::kBFloat16:
    return LaunchAccumulateNormSq<detail::BFloat16Codec>(ctx, grad, total_norm_sq_buffer, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ClipGradNorm does not support dtype " +
                                 std::string(DTypeName(grad.dtype())));
}

template <typename Codec>
Status LaunchScaleByFactor(RuntimeContext &ctx, Tensor *grad, Tensor *clip_scale_buffer,
                           int blocks) {
  ScaleByFactorKernel<Codec><<<blocks, kOptimizerThreads, 0, ctx.stream()>>>(
      grad->data_as<typename Codec::Storage>(), clip_scale_buffer->data_as<float>(), grad->numel());
  return detail::CheckKernelLaunch("ScaleByFactorKernel");
}

Status LaunchScaleByFactor(RuntimeContext &ctx, Tensor *grad, Tensor *clip_scale_buffer,
                           int blocks) {
  switch (grad->dtype()) {
  case DType::kFloat32:
    return LaunchScaleByFactor<detail::Float32Codec>(ctx, grad, clip_scale_buffer, blocks);
  case DType::kFloat16:
    return LaunchScaleByFactor<detail::Float16Codec>(ctx, grad, clip_scale_buffer, blocks);
  case DType::kBFloat16:
    return LaunchScaleByFactor<detail::BFloat16Codec>(ctx, grad, clip_scale_buffer, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ClipGradNorm does not support dtype " +
                                 std::string(DTypeName(grad->dtype())));
}

} // namespace

Result<float> ConstantLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  return base_lr;
}

Result<float> StepLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  if (step_size_ <= 0) {
    return Status::InvalidArgument("StepLRScheduler step_size must be > 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(gamma_, "StepLRScheduler gamma"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  int64_t intervals = step_index / step_size_;
  return base_lr * std::pow(gamma_, static_cast<float>(intervals));
}

Result<float> ExponentialLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(gamma_, "ExponentialLRScheduler gamma"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  return base_lr * std::pow(gamma_, static_cast<float>(step_index));
}

Result<float> CosineAnnealingLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  if (max_steps_ <= 0) {
    return Status::InvalidArgument("CosineAnnealingLRScheduler max_steps must be > 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(min_lr_, "CosineAnnealingLRScheduler min_lr"));
  int64_t clamped_step = std::min(step_index, max_steps_);
  double cosine =
      std::cos(kPi * static_cast<double>(clamped_step) / static_cast<double>(max_steps_));
  return static_cast<float>(min_lr_ +
                            0.5 * (static_cast<double>(base_lr) - min_lr_) * (1.0 + cosine));
}

Optimizer::Optimizer(float lr, float weight_decay) {
  param_groups_.push_back(OptimizerParamGroup{{}, lr, weight_decay});
}

Optimizer::Optimizer(std::vector<OptimizerParamGroup> param_groups)
    : param_groups_(std::move(param_groups)) {
  if (param_groups_.empty()) {
    param_groups_.push_back(OptimizerParamGroup{});
  }
}

Status Optimizer::ZeroGrad(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, "ZeroGrad"));
    DLCUDA_RETURN_IF_ERROR(param.grad->FillZero(ctx.stream()));
  }
  return Status::Ok();
}

Result<std::vector<ResolvedOptimizerParam>>
Optimizer::ResolveParameterGroups(const std::vector<ParameterRef> &params, const float *lr_override,
                                  const LearningRateScheduler *scheduler) const {
  DLCUDA_RETURN_IF_ERROR(ValidateOptimizerParamGroups(param_groups_));

  std::optional<size_t> default_group_index;
  std::unordered_map<std::string, size_t> named_groups;
  for (size_t group_index = 0; group_index < param_groups_.size(); ++group_index) {
    const auto &group = param_groups_[group_index];
    if (group.parameter_names.empty()) {
      default_group_index = group_index;
      continue;
    }
    for (const auto &name : group.parameter_names) {
      named_groups.emplace(name, group_index);
    }
  }

  std::vector<ResolvedOptimizerParam> resolved;
  resolved.reserve(params.size());
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateParameterAndGradient(param, Name()));

    std::optional<size_t> group_index;
    auto named_it = named_groups.find(param.name);
    if (named_it != named_groups.end()) {
      group_index = named_it->second;
    } else if (default_group_index.has_value()) {
      group_index = default_group_index.value();
    }
    if (!group_index.has_value()) {
      return Status::InvalidArgument("No optimizer parameter group for " + param.name);
    }

    const OptimizerParamGroup &group = param_groups_[group_index.value()];
    float lr = group.lr;
    if (lr_override != nullptr) {
      lr = *lr_override;
    } else if (scheduler != nullptr) {
      auto scheduled_lr = scheduler->LearningRate(step_count_, group.lr);
      if (!scheduled_lr.ok()) {
        return scheduled_lr.status();
      }
      lr = scheduled_lr.value();
    }
    DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(lr, "Optimizer lr"));

    resolved.push_back(ResolvedOptimizerParam{&param, lr, group.weight_decay});
  }
  return resolved;
}

Status Optimizer::Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  if (params.empty()) {
    return Status::Ok();
  }
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameters());
  auto resolved = ResolveParameterGroups(params, nullptr, nullptr);
  if (!resolved.ok()) {
    return resolved.status();
  }
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));
  DLCUDA_RETURN_IF_ERROR(StepImpl(ctx, resolved.value(), step_count_ + 1));
  ++step_count_;
  return Status::Ok();
}

Status Optimizer::Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float lr) {
  if (params.empty()) {
    return Status::Ok();
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(lr, "Optimizer lr"));
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameters());
  auto resolved = ResolveParameterGroups(params, &lr, nullptr);
  if (!resolved.ok()) {
    return resolved.status();
  }
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));
  DLCUDA_RETURN_IF_ERROR(StepImpl(ctx, resolved.value(), step_count_ + 1));
  ++step_count_;
  return Status::Ok();
}

Status Optimizer::Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params,
                       const LearningRateScheduler &scheduler) {
  if (params.empty()) {
    return Status::Ok();
  }
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameters());
  auto resolved = ResolveParameterGroups(params, nullptr, &scheduler);
  if (!resolved.ok()) {
    return resolved.status();
  }
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));
  DLCUDA_RETURN_IF_ERROR(StepImpl(ctx, resolved.value(), step_count_ + 1));
  ++step_count_;
  return Status::Ok();
}

Status Optimizer::SetParameterGroups(std::vector<OptimizerParamGroup> param_groups) {
  DLCUDA_RETURN_IF_ERROR(ValidateOptimizerParamGroups(param_groups));
  param_groups_ = std::move(param_groups);
  return Status::Ok();
}

Status Optimizer::SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                                 const std::vector<ParameterRef> &params) {
  if (path.empty()) {
    return Status::InvalidArgument("Optimizer checkpoint path must be non-empty");
  }
  FilePtr file(std::fopen(path.c_str(), "wb"));
  if (!file) {
    return Status::IoError("Failed to open optimizer checkpoint for writing: " + path);
  }
  DLCUDA_RETURN_IF_ERROR(SaveCheckpoint(ctx, file.get(), params));
  if (std::fflush(file.get()) != 0) {
    return Status::IoError("Failed to flush optimizer checkpoint file");
  }
  return CloseFile(&file, "optimizer checkpoint file");
}

Status Optimizer::SaveCheckpoint(RuntimeContext &ctx, FILE *file,
                                 const std::vector<ParameterRef> &params) {
  if (file == nullptr) {
    return Status::InvalidArgument("Optimizer checkpoint file must be non-null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameters());
  DLCUDA_RETURN_IF_ERROR(ValidateOptimizerParamGroups(param_groups_));
  DLCUDA_RETURN_IF_ERROR(ValidateCheckpointParameters(params));
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));

  std::vector<StateTensorRef> states;
  DLCUDA_RETURN_IF_ERROR(CollectStateTensors(params, &states));
  DLCUDA_RETURN_IF_ERROR(ValidateStateTensorRefs(states));
  std::vector<Hyperparameter> hyperparameters;
  CollectHyperparameters(&hyperparameters);
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameterRefs(hyperparameters));

  if (!WriteExact(file, kOptimizerMagic, sizeof(kOptimizerMagic))) {
    return Status::IoError("Failed to write optimizer checkpoint magic");
  }
  uint32_t version = kOptimizerCheckpointVersion;
  if (!WriteExact(file, &version, sizeof(version))) {
    return Status::IoError("Failed to write optimizer checkpoint version");
  }
  DLCUDA_RETURN_IF_ERROR(WriteString(file, Name()));
  if (!WriteExact(file, &step_count_, sizeof(step_count_))) {
    return Status::IoError("Failed to write optimizer step count");
  }
  DLCUDA_RETURN_IF_ERROR(WriteHyperparameters(file, hyperparameters));
  DLCUDA_RETURN_IF_ERROR(WriteParamGroups(file, param_groups_));
  return WriteStateTensors(ctx, file, states);
}

Status Optimizer::LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                                 const std::vector<ParameterRef> &params) {
  if (path.empty()) {
    return Status::InvalidArgument("Optimizer checkpoint path must be non-empty");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameters());
  DLCUDA_RETURN_IF_ERROR(ValidateCheckpointParameters(params));

  FilePtr file(std::fopen(path.c_str(), "rb"));
  if (!file) {
    return Status::IoError("Failed to open optimizer checkpoint for reading: " + path);
  }
  return LoadCheckpoint(ctx, file.get(), params);
}

Status Optimizer::LoadCheckpoint(RuntimeContext &ctx, FILE *file,
                                 const std::vector<ParameterRef> &params) {
  if (file == nullptr) {
    return Status::InvalidArgument("Optimizer checkpoint file must be non-null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameters());
  DLCUDA_RETURN_IF_ERROR(ValidateCheckpointParameters(params));

  char magic[sizeof(kOptimizerMagic)] = {0};
  if (!ReadExact(file, magic, sizeof(magic))) {
    return Status::IoError("Failed to read optimizer checkpoint magic");
  }
  if (std::memcmp(magic, kOptimizerMagic, sizeof(kOptimizerMagic)) != 0) {
    return Status::InvalidArgument("Optimizer checkpoint magic mismatch");
  }

  uint32_t version = 0;
  if (!ReadExact(file, &version, sizeof(version))) {
    return Status::IoError("Failed to read optimizer checkpoint version");
  }
  if (version != kOptimizerCheckpointVersion) {
    return Status::InvalidArgument("Unsupported optimizer checkpoint version");
  }

  std::string optimizer_name;
  DLCUDA_RETURN_IF_ERROR(ReadString(file, &optimizer_name));
  if (optimizer_name != Name()) {
    return Status::InvalidArgument("Optimizer checkpoint mismatch: expected " +
                                   std::string(Name()) + " got " + optimizer_name);
  }

  int64_t loaded_step_count = 0;
  if (!ReadExact(file, &loaded_step_count, sizeof(loaded_step_count))) {
    return Status::IoError("Failed to read optimizer step count");
  }
  if (loaded_step_count < 0) {
    return Status::InvalidArgument("Optimizer checkpoint step count must be non-negative");
  }

  std::vector<Hyperparameter> loaded_hyperparameters;
  DLCUDA_RETURN_IF_ERROR(ReadHyperparameters(file, &loaded_hyperparameters));
  std::vector<Hyperparameter> expected_hyperparameters;
  CollectHyperparameters(&expected_hyperparameters);
  DLCUDA_RETURN_IF_ERROR(ValidateHyperparameterRefs(expected_hyperparameters));
  DLCUDA_RETURN_IF_ERROR(
      ValidateLoadedHyperparameters(expected_hyperparameters, loaded_hyperparameters));

  std::vector<OptimizerParamGroup> loaded_groups;
  DLCUDA_RETURN_IF_ERROR(ReadParamGroups(file, &loaded_groups));

  std::unordered_map<std::string, HostTensorRecord> records;
  DLCUDA_RETURN_IF_ERROR(ReadStateTensors(file, &records));

  DLCUDA_RETURN_IF_ERROR(SetParameterGroups(std::move(loaded_groups)));
  DLCUDA_RETURN_IF_ERROR(EnsureState(ctx, params));
  std::vector<StateTensorRef> states;
  DLCUDA_RETURN_IF_ERROR(CollectStateTensors(params, &states));
  DLCUDA_RETURN_IF_ERROR(ValidateStateTensorRefs(states));
  DLCUDA_RETURN_IF_ERROR(RestoreStateTensors(ctx, records, states));

  step_count_ = loaded_step_count;
  return Status::Ok();
}

SGDOptimizer::SGDOptimizer(float lr, float momentum, float weight_decay, float dampening,
                           bool nesterov)
    : Optimizer(lr, weight_decay), momentum_(momentum), dampening_(dampening), nesterov_(nesterov) {
}

SGDOptimizer::SGDOptimizer(std::vector<OptimizerParamGroup> param_groups, float momentum,
                           float dampening, bool nesterov)
    : Optimizer(std::move(param_groups)), momentum_(momentum), dampening_(dampening),
      nesterov_(nesterov) {}

Status SGDOptimizer::ValidateHyperparameters() const {
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(momentum_, "SGD momentum"));
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(dampening_, "SGD dampening"));
  if (nesterov_ && (momentum_ <= 0.0f || dampening_ != 0.0f)) {
    return Status::InvalidArgument("SGD nesterov requires momentum > 0 and dampening == 0");
  }
  return Status::Ok();
}

void SGDOptimizer::CollectHyperparameters(std::vector<Hyperparameter> *out) const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->push_back(Hyperparameter{"momentum", momentum_});
  out->push_back(Hyperparameter{"dampening", dampening_});
  out->push_back(Hyperparameter{"nesterov", nesterov_ ? 1.0f : 0.0f});
}

Status SGDOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  if (momentum_ == 0.0f) {
    ClearStateMap(&momentum_state_);
    for (const auto &param : params) {
      DLCUDA_RETURN_IF_ERROR(ValidateParameterOnly(param, "SGD"));
    }
    return Status::Ok();
  }
  return EnsureStateMap(ctx, params, &momentum_state_);
}

Status SGDOptimizer::StepImpl(RuntimeContext &ctx,
                              const std::vector<ResolvedOptimizerParam> &params,
                              int64_t step_index) {
  (void)step_index;
  bool has_momentum = momentum_ != 0.0f;
  for (const auto &resolved : params) {
    const ParameterRef &param = *resolved.param;
    Tensor *momentum_buffer = has_momentum ? &momentum_state_.at(param.value) : nullptr;
    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchSGDUpdate(ctx, param, momentum_buffer, has_momentum, resolved.lr,
                                             momentum_, resolved.weight_decay, dampening_,
                                             nesterov_, blocks.value()));
    }
  }
  return Status::Ok();
}

Status SGDOptimizer::CollectStateTensors(const std::vector<ParameterRef> &params,
                                         std::vector<StateTensorRef> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("SGD state destination is null");
  }
  out->clear();
  if (momentum_ == 0.0f) {
    return Status::Ok();
  }
  out->reserve(params.size());
  for (const auto &param : params) {
    out->push_back(StateTensorRef{StateName(param, "momentum"), &momentum_state_.at(param.value)});
  }
  return Status::Ok();
}

AdamOptimizer::AdamOptimizer(float beta1, float beta2, float epsilon)
    : Optimizer(), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

AdamOptimizer::AdamOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1,
                             float beta2, float epsilon)
    : AdamOptimizer(std::move(param_groups), beta1, beta2, epsilon, false) {}

AdamOptimizer::AdamOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1,
                             float beta2, float epsilon, bool decoupled_weight_decay)
    : Optimizer(std::move(param_groups)), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      decoupled_weight_decay_(decoupled_weight_decay) {}

Status AdamOptimizer::ValidateHyperparameters() const {
  DLCUDA_RETURN_IF_ERROR(ValidateRate(beta1_, "Adam beta1"));
  DLCUDA_RETURN_IF_ERROR(ValidateRate(beta2_, "Adam beta2"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(epsilon_, "Adam epsilon"));
  return Status::Ok();
}

void AdamOptimizer::CollectHyperparameters(std::vector<Hyperparameter> *out) const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->push_back(Hyperparameter{"beta1", beta1_});
  out->push_back(Hyperparameter{"beta2", beta2_});
  out->push_back(Hyperparameter{"epsilon", epsilon_});
}

Status AdamOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &m_state_));
  DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &v_state_));
  return Status::Ok();
}

Status AdamOptimizer::StepImpl(RuntimeContext &ctx,
                               const std::vector<ResolvedOptimizerParam> &params,
                               int64_t step_index) {
  float beta1_power = std::pow(beta1_, static_cast<float>(step_index));
  float beta2_power = std::pow(beta2_, static_cast<float>(step_index));
  float inv_bias_correction1 = 1.0f / (1.0f - beta1_power);
  float inv_bias_correction2 = 1.0f / (1.0f - beta2_power);

  for (const auto &resolved : params) {
    const ParameterRef &param = *resolved.param;
    Tensor &m = m_state_.at(param.value);
    Tensor &v = v_state_.at(param.value);

    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchAdamUpdate(
          ctx, param, &m, &v, resolved.lr, beta1_, beta2_, epsilon_, inv_bias_correction1,
          inv_bias_correction2, resolved.weight_decay, decoupled_weight_decay_, blocks.value()));
    }
  }

  return Status::Ok();
}

Status AdamOptimizer::CollectStateTensors(const std::vector<ParameterRef> &params,
                                          std::vector<StateTensorRef> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("Adam state destination is null");
  }
  out->clear();
  out->reserve(params.size() * 2);
  for (const auto &param : params) {
    out->push_back(StateTensorRef{StateName(param, "m"), &m_state_.at(param.value)});
    out->push_back(StateTensorRef{StateName(param, "v"), &v_state_.at(param.value)});
  }
  return Status::Ok();
}

AdamWOptimizer::AdamWOptimizer(float lr, float weight_decay, float beta1, float beta2,
                               float epsilon)
    : AdamOptimizer(std::vector<OptimizerParamGroup>{{{}, lr, weight_decay}}, beta1, beta2, epsilon,
                    true) {}

AdamWOptimizer::AdamWOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1,
                               float beta2, float epsilon)
    : AdamOptimizer(std::move(param_groups), beta1, beta2, epsilon, true) {}

RMSPropOptimizer::RMSPropOptimizer(float lr, float alpha, float epsilon, float momentum,
                                   float weight_decay, bool centered)
    : Optimizer(lr, weight_decay), alpha_(alpha), epsilon_(epsilon), momentum_(momentum),
      centered_(centered) {}

RMSPropOptimizer::RMSPropOptimizer(std::vector<OptimizerParamGroup> param_groups, float alpha,
                                   float epsilon, float momentum, bool centered)
    : Optimizer(std::move(param_groups)), alpha_(alpha), epsilon_(epsilon), momentum_(momentum),
      centered_(centered) {}

Status RMSPropOptimizer::ValidateHyperparameters() const {
  DLCUDA_RETURN_IF_ERROR(ValidateRate(alpha_, "RMSProp alpha"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(epsilon_, "RMSProp epsilon"));
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(momentum_, "RMSProp momentum"));
  return Status::Ok();
}

void RMSPropOptimizer::CollectHyperparameters(std::vector<Hyperparameter> *out) const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->push_back(Hyperparameter{"alpha", alpha_});
  out->push_back(Hyperparameter{"epsilon", epsilon_});
  out->push_back(Hyperparameter{"momentum", momentum_});
  out->push_back(Hyperparameter{"centered", centered_ ? 1.0f : 0.0f});
}

Status RMSPropOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &square_avg_state_));
  if (momentum_ != 0.0f) {
    DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &momentum_state_));
  } else {
    ClearStateMap(&momentum_state_);
  }
  if (centered_) {
    DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &grad_avg_state_));
  } else {
    ClearStateMap(&grad_avg_state_);
  }
  return Status::Ok();
}

Status RMSPropOptimizer::StepImpl(RuntimeContext &ctx,
                                  const std::vector<ResolvedOptimizerParam> &params,
                                  int64_t step_index) {
  (void)step_index;
  bool has_momentum = momentum_ != 0.0f;
  for (const auto &resolved : params) {
    const ParameterRef &param = *resolved.param;
    Tensor &square_avg = square_avg_state_.at(param.value);
    Tensor *momentum_buffer = has_momentum ? &momentum_state_.at(param.value) : nullptr;
    Tensor *grad_avg = centered_ ? &grad_avg_state_.at(param.value) : nullptr;

    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchRMSPropUpdate(
          ctx, param, &square_avg, momentum_buffer, grad_avg, has_momentum, centered_, resolved.lr,
          alpha_, epsilon_, momentum_, resolved.weight_decay, blocks.value()));
    }
  }
  return Status::Ok();
}

Status RMSPropOptimizer::CollectStateTensors(const std::vector<ParameterRef> &params,
                                             std::vector<StateTensorRef> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("RMSProp state destination is null");
  }
  out->clear();
  size_t states_per_param = 1 + (momentum_ != 0.0f ? 1 : 0) + (centered_ ? 1 : 0);
  out->reserve(params.size() * states_per_param);
  for (const auto &param : params) {
    out->push_back(
        StateTensorRef{StateName(param, "square_avg"), &square_avg_state_.at(param.value)});
    if (momentum_ != 0.0f) {
      out->push_back(
          StateTensorRef{StateName(param, "momentum"), &momentum_state_.at(param.value)});
    }
    if (centered_) {
      out->push_back(
          StateTensorRef{StateName(param, "grad_avg"), &grad_avg_state_.at(param.value)});
    }
  }
  return Status::Ok();
}

Status ClipGradNorm(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float max_norm,
                    float *total_norm) {
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(max_norm, "max_norm"));

  bool has_grad_elements = false;
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, "ClipGradNorm"));
    if (param.grad->numel() > 0) {
      has_grad_elements = true;
    }
  }
  if (!has_grad_elements) {
    if (total_norm != nullptr) {
      *total_norm = 0.0f;
    }
    return Status::Ok();
  }

  auto total_norm_sq_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.total_norm_sq", {1}, DType::kFloat32);
  if (!total_norm_sq_tensor.ok()) {
    return total_norm_sq_tensor.status();
  }
  Tensor total_norm_sq_buffer = total_norm_sq_tensor.value();
  DLCUDA_RETURN_IF_ERROR(total_norm_sq_buffer.FillZero(ctx.stream()));

  auto clip_scale_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.clip_scale", {1}, DType::kFloat32);
  if (!clip_scale_tensor.ok()) {
    return clip_scale_tensor.status();
  }
  Tensor clip_scale_buffer = clip_scale_tensor.value();

  for (const auto &param : params) {
    int64_t n = param.grad->numel();
    auto blocks = detail::CappedBlocksForElements(n, kOptimizerThreads, kNormReductionMaxBlocks);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() <= 0) {
      continue;
    }
    DLCUDA_RETURN_IF_ERROR(
        LaunchAccumulateNormSq(ctx, *param.grad, &total_norm_sq_buffer, blocks.value()));
  }

  ComputeClipScaleKernel<<<1, 1, 0, ctx.stream()>>>(total_norm_sq_buffer.data_as<float>(), max_norm,
                                                    clip_scale_buffer.data_as<float>());
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ComputeClipScaleKernel"));

  for (const auto &param : params) {
    auto blocks = detail::BlocksForElements(param.grad->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() <= 0) {
      continue;
    }
    DLCUDA_RETURN_IF_ERROR(
        LaunchScaleByFactor(ctx, param.grad, &clip_scale_buffer, blocks.value()));
  }

  if (total_norm != nullptr) {
    float total_norm_sq = 0.0f;
    DLCUDA_RETURN_IF_ERROR(
        total_norm_sq_buffer.CopyToHost(&total_norm_sq, sizeof(total_norm_sq), ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());
    *total_norm = std::sqrt(total_norm_sq);
  }

  return Status::Ok();
}

} // namespace dlcuda
