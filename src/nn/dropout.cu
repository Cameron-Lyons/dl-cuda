#include "detail/dropout_kernels.cuh"

namespace dlcuda {

Dropout::Dropout(float probability, uint64_t seed)
    : probability_(probability), seed_(seed == 0ULL ? 0x1234abcd5678ef90ULL : seed) {
  if (probability_ < 0.0f || probability_ >= 1.0f) {
    init_status_ = Status::InvalidArgument("Dropout probability must be in [0, 1)");
  }
}

void Dropout::SetTraining(bool training) {
  training_ = training;
}

Status Dropout::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Dropout::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Dropout input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, input.shape(), input.dtype(), ctx.stream()));
  last_training_ = training_;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    if (training_) {
      DLCUDA_RETURN_IF_ERROR(
          EnsureTensorAsync(&mask_, input.shape(), DType::kFloat32, ctx.stream()));
      uint64_t call_seed = seed_ + 0x9e3779b97f4a7c15ULL * (++call_index_);
      DLCUDA_RETURN_IF_ERROR(LaunchDropoutForwardKernel(ctx, input, &forward_output_, &mask_,
                                                        blocks.value(), probability_, call_seed));
    } else {
      DLCUDA_RETURN_IF_ERROR(LaunchTensorCopyKernel(ctx, input, &forward_output_, blocks.value()));
    }
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Dropout::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Dropout grad_output"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    if (last_training_) {
      if (!mask_.defined() || mask_.shape() != grad_output.shape()) {
        return Status::RuntimeError("Dropout backward called before matching training forward");
      }
      DLCUDA_RETURN_IF_ERROR(
          LaunchDropoutBackwardKernel(ctx, grad_output, mask_, &backward_output_, blocks.value()));
    } else {
      DLCUDA_RETURN_IF_ERROR(
          LaunchTensorCopyKernel(ctx, grad_output, &backward_output_, blocks.value()));
    }
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Dropout::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

} // namespace dlcuda
