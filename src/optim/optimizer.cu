#include "detail/common.cuh"

namespace dlcuda {

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

} // namespace dlcuda
