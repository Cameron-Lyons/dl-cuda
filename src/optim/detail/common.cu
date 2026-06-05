#include "common.cuh"

#include "dl_cuda/detail/value_validation.hpp"

namespace dlcuda {

Status ValidatePositiveFinite(float value, const char *name) {
  return detail::ValidatePositiveFinite(value, name);
}

Status ValidateNonNegativeFinite(float value, const char *name) {
  return detail::ValidateNonNegativeFinite(value, name);
}

Status ValidateRate(float value, const char *name) {
  return detail::ValidateRate(value, name);
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

Result<Tensor> ScratchTensorForBytes(RuntimeContext &ctx, const std::string &key, size_t bytes) {
  int64_t words = 0;
  if (bytes > 0) {
    size_t word_count = (bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
    if (word_count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return Status::InvalidArgument("Scratch byte request is too large");
    }
    words = static_cast<int64_t>(word_count);
  }
  return ctx.ScratchTensor(key, {words}, DType::kInt32);
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

} // namespace dlcuda
