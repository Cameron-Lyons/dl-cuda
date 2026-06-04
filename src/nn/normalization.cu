#include "detail/normalization_kernels.cuh"

namespace dlcuda {

LayerNorm::LayerNorm(int64_t normalized_size, RuntimeContext &ctx, float eps, DType dtype)
    : normalized_size_(normalized_size), eps_(eps), dtype_(dtype) {
  if (normalized_size_ <= 0 || eps_ <= 0.0f) {
    init_status_ = Status::InvalidArgument("LayerNorm normalized_size and eps must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "LayerNorm");
  if (!init_status_.ok()) {
    return;
  }

  auto gamma = Tensor::AllocateAsync({normalized_size_}, dtype_, ctx.stream());
  auto beta = Tensor::AllocateAsync({normalized_size_}, dtype_, ctx.stream());
  auto grad_gamma = Tensor::AllocateAsync({normalized_size_}, DType::kFloat32, ctx.stream());
  auto grad_beta = Tensor::AllocateAsync({normalized_size_}, DType::kFloat32, ctx.stream());
  if (!gamma.ok() || !beta.ok() || !grad_gamma.ok() || !grad_beta.ok()) {
    init_status_ = !gamma.ok()        ? gamma.status()
                   : !beta.ok()       ? beta.status()
                   : !grad_gamma.ok() ? grad_gamma.status()
                                      : grad_beta.status();
    return;
  }
  gamma_ = gamma.value();
  beta_ = beta.value();
  grad_gamma_ = grad_gamma.value();
  grad_beta_ = grad_beta.value();

  std::vector<float> host_gamma(static_cast<size_t>(normalized_size_), 1.0f);
  init_status_ = CopyHostFloatsToTensor(&gamma_, host_gamma, ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = beta_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_gamma_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_beta_.FillZero(ctx.stream());
}

Status LayerNorm::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("LayerNorm::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "LayerNorm input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "LayerNorm input"));
  if (input.rank() < 1 || input.dim(static_cast<int>(input.rank() - 1)) != normalized_size_) {
    return Status::InvalidArgument("LayerNorm input last dimension mismatch");
  }

  last_rows_ = input.numel() / normalized_size_;
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&cached_x_hat_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&inv_std_, {last_rows_}, DType::kFloat32, ctx.stream()));

  auto rows = detail::RowsForGrid(last_rows_, "LayerNorm");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchLayerNormForwardKernel(ctx, dtype_, input, gamma_, beta_,
                                                        &forward_output_, &cached_x_hat_, &inv_std_,
                                                        rows.value(), normalized_size_, eps_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status LayerNorm::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "LayerNorm grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "LayerNorm grad_output"));
  if (!cached_x_hat_.defined()) {
    return Status::RuntimeError("LayerNorm backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_x_hat_, "grad_output", "cached_x_hat"));
  DLCUDA_RETURN_IF_ERROR(grad_gamma_.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(grad_beta_.FillZero(ctx.stream()));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), dtype_, ctx.stream()));
  auto rows = detail::RowsForGrid(last_rows_, "LayerNorm");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchLayerNormBackwardKernel(
        ctx, dtype_, grad_output, cached_x_hat_, gamma_, &backward_output_, &grad_gamma_,
        &grad_beta_, inv_std_, rows.value(), normalized_size_));
  }

  if (grad_input != nullptr) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void LayerNorm::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "gamma"), &gamma_, &grad_gamma_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "beta"), &beta_, &grad_beta_});
}

BatchNorm1d::BatchNorm1d(int64_t features, RuntimeContext &ctx, float eps, float momentum,
                         DType dtype)
    : features_(features), eps_(eps), momentum_(momentum), dtype_(dtype) {
  if (features_ <= 0 || eps_ <= 0.0f || momentum_ < 0.0f || momentum_ > 1.0f) {
    init_status_ = Status::InvalidArgument("BatchNorm1d features/eps/momentum are invalid");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "BatchNorm1d");
  if (!init_status_.ok()) {
    return;
  }

  auto gamma = Tensor::AllocateAsync({features_}, dtype_, ctx.stream());
  auto beta = Tensor::AllocateAsync({features_}, dtype_, ctx.stream());
  auto grad_gamma = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  auto grad_beta = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  auto running_mean = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  auto running_var = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  if (!gamma.ok() || !beta.ok() || !grad_gamma.ok() || !grad_beta.ok() || !running_mean.ok() ||
      !running_var.ok()) {
    init_status_ = !gamma.ok()          ? gamma.status()
                   : !beta.ok()         ? beta.status()
                   : !grad_gamma.ok()   ? grad_gamma.status()
                   : !grad_beta.ok()    ? grad_beta.status()
                   : !running_mean.ok() ? running_mean.status()
                                        : running_var.status();
    return;
  }
  gamma_ = gamma.value();
  beta_ = beta.value();
  grad_gamma_ = grad_gamma.value();
  grad_beta_ = grad_beta.value();
  running_mean_ = running_mean.value();
  running_var_ = running_var.value();

  std::vector<float> host_ones(static_cast<size_t>(features_), 1.0f);
  init_status_ = CopyHostFloatsToTensor(&gamma_, host_ones, ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ =
      running_var_.CopyFromHost(host_ones.data(), host_ones.size() * sizeof(float), ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = beta_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = running_mean_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_gamma_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_beta_.FillZero(ctx.stream());
}

void BatchNorm1d::SetTraining(bool training) {
  training_ = training;
}

Status BatchNorm1d::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("BatchNorm1d::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "BatchNorm1d input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "BatchNorm1d input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "BatchNorm1d input"));
  if (input.dim(1) != features_) {
    return Status::InvalidArgument("BatchNorm1d feature dimension mismatch");
  }
  if (training_ && input.dim(0) <= 0) {
    return Status::InvalidArgument("BatchNorm1d training batch size must be positive");
  }

  last_batch_ = input.dim(0);
  last_training_ = training_;
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&cached_x_hat_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&inv_std_, {features_}, DType::kFloat32, ctx.stream()));

  if (last_batch_ == 0) {
    *output = forward_output_;
    return Status::Ok();
  }

  if (training_) {
    auto rows = detail::RowsForGrid(features_, "BatchNorm1d");
    if (!rows.ok()) {
      return rows.status();
    }
    DLCUDA_RETURN_IF_ERROR(LaunchBatchNorm1dForwardTrainingKernel(
        ctx, dtype_, input, gamma_, beta_, &forward_output_, &cached_x_hat_, &inv_std_,
        &running_mean_, &running_var_, rows.value(), last_batch_, features_, eps_, momentum_));
  } else {
    auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchBatchNorm1dForwardEvalKernel(
          ctx, dtype_, input, gamma_, beta_, &forward_output_, &cached_x_hat_, &inv_std_,
          running_mean_, running_var_, blocks.value(), last_batch_, features_, eps_));
    }
  }

  *output = forward_output_;
  return Status::Ok();
}

Status BatchNorm1d::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "BatchNorm1d grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "BatchNorm1d grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "BatchNorm1d grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != features_) {
    return Status::InvalidArgument("BatchNorm1d grad_output shape mismatch");
  }
  if (!cached_x_hat_.defined()) {
    return Status::RuntimeError("BatchNorm1d backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(grad_gamma_.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(grad_beta_.FillZero(ctx.stream()));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), dtype_, ctx.stream()));
  if (last_batch_ == 0) {
    if (grad_input != nullptr) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }
  auto rows = detail::RowsForGrid(features_, "BatchNorm1d");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchBatchNorm1dBackwardKernel(
        ctx, dtype_, grad_output, cached_x_hat_, gamma_, &backward_output_, &grad_gamma_,
        &grad_beta_, inv_std_, rows.value(), last_batch_, features_, last_training_));
  }

  if (grad_input != nullptr) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void BatchNorm1d::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "gamma"), &gamma_, &grad_gamma_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "beta"), &beta_, &grad_beta_});
}

} // namespace dlcuda
