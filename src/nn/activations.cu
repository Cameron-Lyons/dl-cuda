#include "detail/activation_kernels.cuh"

namespace dlcuda {
namespace {

using UnaryForwardLauncher = Status (*)(RuntimeContext &, const Tensor &, Tensor *, int);
using UnaryBackwardLauncher = Status (*)(RuntimeContext &, const Tensor &, const Tensor &, Tensor *,
                                         int);

Status RunUnaryForward(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                       Tensor *cached_input, Tensor *stored_output, const char *op_name,
                       UnaryForwardLauncher launcher) {
  if (output == nullptr) {
    return Status::InvalidArgument(std::string(op_name) + "::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, (std::string(op_name) + " input").c_str()));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(stored_output, input.shape(), input.dtype(), ctx.stream()));
  if (cached_input != nullptr) {
    *cached_input = input;
  }

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(launcher(ctx, input, stored_output, blocks.value()));
  }

  *output = *stored_output;
  return Status::Ok();
}

Status RunUnaryBackward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input,
                        const Tensor &cached_tensor, Tensor *stored_grad_input, const char *op_name,
                        const char *cached_name, UnaryBackwardLauncher launcher) {
  DLCUDA_RETURN_IF_ERROR(
      ValidateFloatingTensor(grad_output, (std::string(op_name) + " grad_output").c_str()));
  if (!cached_tensor.defined()) {
    return Status::RuntimeError(std::string(op_name) + " backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_tensor, "grad_output", cached_name));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(stored_grad_input, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(
        launcher(ctx, grad_output, cached_tensor, stored_grad_input, blocks.value()));
  }

  *grad_input = *stored_grad_input;
  return Status::Ok();
}

} // namespace

Status ReLU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  return RunUnaryForward(ctx, input, output, &cached_input_, &forward_output_, "ReLU",
                         static_cast<UnaryForwardLauncher>(LaunchReLUForwardKernel));
}

Status ReLU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  return RunUnaryBackward(ctx, grad_output, grad_input, cached_input_, &backward_output_, "ReLU",
                          "cached_input",
                          static_cast<UnaryBackwardLauncher>(LaunchReLUBackwardKernel));
}

void ReLU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status GELU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  return RunUnaryForward(ctx, input, output, &cached_input_, &forward_output_, "GELU",
                         static_cast<UnaryForwardLauncher>(LaunchGELUForwardKernel));
}

Status GELU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  return RunUnaryBackward(ctx, grad_output, grad_input, cached_input_, &backward_output_, "GELU",
                          "cached_input",
                          static_cast<UnaryBackwardLauncher>(LaunchGELUBackwardKernel));
}

void GELU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Sigmoid::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  return RunUnaryForward(ctx, input, output, nullptr, &cached_output_, "Sigmoid",
                         static_cast<UnaryForwardLauncher>(LaunchSigmoidForwardKernel));
}

Status Sigmoid::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  return RunUnaryBackward(ctx, grad_output, grad_input, cached_output_, &backward_output_,
                          "Sigmoid", "cached_output",
                          static_cast<UnaryBackwardLauncher>(LaunchSigmoidBackwardKernel));
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
