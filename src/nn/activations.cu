#include "detail/activation_kernels.cuh"

namespace dlcuda {

Status ReLU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("ReLU::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "ReLU input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, input.shape(), input.dtype(), ctx.stream()));
  cached_input_ = input;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchReLUForwardKernel(ctx, input, &forward_output_, blocks.value()));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status ReLU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "ReLU grad_output"));
  if (!cached_input_.defined()) {
    return Status::RuntimeError("ReLU backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_input_, "grad_output", "cached_input"));
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
    DLCUDA_RETURN_IF_ERROR(LaunchReLUBackwardKernel(ctx, grad_output, cached_input_,
                                                    &backward_output_, blocks.value()));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void ReLU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status GELU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("GELU::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "GELU input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, input.shape(), input.dtype(), ctx.stream()));
  cached_input_ = input;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchGELUForwardKernel(ctx, input, &forward_output_, blocks.value()));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status GELU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "GELU grad_output"));
  if (!cached_input_.defined()) {
    return Status::RuntimeError("GELU backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_input_, "grad_output", "cached_input"));
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
    DLCUDA_RETURN_IF_ERROR(LaunchGELUBackwardKernel(ctx, grad_output, cached_input_,
                                                    &backward_output_, blocks.value()));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void GELU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Sigmoid::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Sigmoid::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Sigmoid input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&cached_output_, input.shape(), input.dtype(), ctx.stream()));

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchSigmoidForwardKernel(ctx, input, &cached_output_, blocks.value()));
  }

  *output = cached_output_;
  return Status::Ok();
}

Status Sigmoid::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Sigmoid grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Sigmoid backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_output_, "grad_output", "cached_output"));
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
    DLCUDA_RETURN_IF_ERROR(LaunchSigmoidBackwardKernel(ctx, grad_output, cached_output_,
                                                       &backward_output_, blocks.value()));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Sigmoid::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Softmax::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Softmax::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Softmax input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Softmax input"));

  num_rows_ = input.dim(0);
  row_width_ = input.dim(1);
  if (num_rows_ > 0 && row_width_ == 0) {
    return Status::InvalidArgument("Softmax row width must be positive");
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&cached_output_, input.shape(), input.dtype(), ctx.stream()));

  auto rows = detail::RowsForGrid(num_rows_, "softmax");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(
        LaunchSoftmaxForwardKernel(ctx, input, &cached_output_, rows.value(), row_width_));
  }

  *output = cached_output_;
  return Status::Ok();
}

Status Softmax::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Softmax grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Softmax backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_output_, "grad_output", "cached_output"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  if (num_rows_ > 0 && row_width_ == 0) {
    return Status::InvalidArgument("Softmax row width must be positive");
  }
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  auto rows = detail::RowsForGrid(num_rows_, "softmax");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchSoftmaxBackwardKernel(
        ctx, grad_output, cached_output_, &backward_output_, rows.value(), num_rows_, row_width_));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Softmax::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

} // namespace dlcuda
