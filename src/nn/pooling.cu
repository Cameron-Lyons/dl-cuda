#include "detail/pooling_kernels.cuh"

namespace dlcuda {

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride)
    : MaxPool2d(kernel_size, kernel_size, stride == 0 ? kernel_size : stride,
                stride == 0 ? kernel_size : stride, 0, 0) {}

MaxPool2d::MaxPool2d(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                     int64_t padding_h, int64_t padding_w)
    : kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w) {
  if (kernel_h_ <= 0 || kernel_w_ <= 0 || stride_h_ <= 0 || stride_w_ <= 0 || padding_h_ < 0 ||
      padding_w_ < 0) {
    init_status_ = Status::InvalidArgument("MaxPool2d parameters are invalid");
  }
}

Status MaxPool2d::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("MaxPool2d::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "MaxPool2d input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 4, "MaxPool2d input"));
  if (input.numel() > std::numeric_limits<int32_t>::max()) {
    return Status::InvalidArgument("MaxPool2d input is too large for int32 argmax indices");
  }

  last_batch_ = input.dim(0);
  last_channels_ = input.dim(1);
  last_input_h_ = input.dim(2);
  last_input_w_ = input.dim(3);
  dtype_ = input.dtype();
  auto output_h =
      SpatialOutputSize(last_input_h_, kernel_h_, stride_h_, padding_h_, "MaxPool2d height");
  if (!output_h.ok()) {
    return output_h.status();
  }
  auto output_w =
      SpatialOutputSize(last_input_w_, kernel_w_, stride_w_, padding_w_, "MaxPool2d width");
  if (!output_w.ok()) {
    return output_w.status();
  }
  last_output_h_ = output_h.value();
  last_output_w_ = output_w.value();

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(
      &forward_output_, {last_batch_, last_channels_, last_output_h_, last_output_w_},
      input.dtype(), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(
      &argmax_indices_, {last_batch_, last_channels_, last_output_h_, last_output_w_},
      DType::kInt32, ctx.stream()));
  int64_t total = forward_output_.numel();
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchMaxPool2dForwardKernel(
        ctx, input, &forward_output_, &argmax_indices_, blocks.value(), total, last_batch_,
        last_channels_, last_input_h_, last_input_w_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        padding_h_, padding_w_, last_output_h_, last_output_w_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status MaxPool2d::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "MaxPool2d grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "MaxPool2d grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 4, "MaxPool2d grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != last_channels_ ||
      grad_output.dim(2) != last_output_h_ || grad_output.dim(3) != last_output_w_) {
    return Status::InvalidArgument("MaxPool2d grad_output shape mismatch");
  }
  if (!argmax_indices_.defined()) {
    return Status::RuntimeError("MaxPool2d backward called before forward");
  }
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(
      &backward_output_, {last_batch_, last_channels_, last_input_h_, last_input_w_},
      grad_output.dtype(), ctx.stream()));
  int64_t total = backward_output_.numel();
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchMaxPool2dBackwardKernel(
        ctx, grad_output, argmax_indices_, &backward_output_, blocks.value(), total, last_batch_,
        last_channels_, last_input_h_, last_input_w_, last_output_h_, last_output_w_));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void MaxPool2d::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

} // namespace dlcuda
