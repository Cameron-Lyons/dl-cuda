#include "detail/conv2d_kernels.cuh"

namespace dlcuda {

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, RuntimeContext &ctx,
               DType dtype)
    : Conv2d(in_channels, out_channels, kernel_size, kernel_size, ctx, 1, 1, 0, 0, dtype) {}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
               RuntimeContext &ctx, int64_t stride_h, int64_t stride_w, int64_t padding_h,
               int64_t padding_w, DType dtype)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_h_(kernel_h),
      kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w), padding_h_(padding_h),
      padding_w_(padding_w), dtype_(dtype) {
  if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_h_ <= 0 || kernel_w_ <= 0 ||
      stride_h_ <= 0 || stride_w_ <= 0 || padding_h_ < 0 || padding_w_ < 0) {
    init_status_ = Status::InvalidArgument("Conv2d dimensions and strides must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "Conv2d");
  if (!init_status_.ok()) {
    return;
  }

  auto weight = Tensor::AllocateAsync({out_channels_, in_channels_, kernel_h_, kernel_w_}, dtype_,
                                      ctx.stream());
  if (!weight.ok()) {
    init_status_ = weight.status();
    return;
  }
  auto bias = Tensor::AllocateAsync({out_channels_}, dtype_, ctx.stream());
  if (!bias.ok()) {
    init_status_ = bias.status();
    return;
  }
  auto grad_weight = Tensor::AllocateAsync({out_channels_, in_channels_, kernel_h_, kernel_w_},
                                           DType::kFloat32, ctx.stream());
  if (!grad_weight.ok()) {
    init_status_ = grad_weight.status();
    return;
  }
  auto grad_bias = Tensor::AllocateAsync({out_channels_}, DType::kFloat32, ctx.stream());
  if (!grad_bias.ok()) {
    init_status_ = grad_bias.status();
    return;
  }

  weight_ = weight.value();
  bias_ = bias.value();
  grad_weight_ = grad_weight.value();
  grad_bias_ = grad_bias.value();

  std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
  float fan_in = static_cast<float>(in_channels_ * kernel_h_ * kernel_w_);
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
  std::vector<float> host_weight(static_cast<size_t>(weight_.numel()));
  for (float &v : host_weight) {
    v = dist(rng);
  }

  init_status_ = CopyHostFloatsToTensor(&weight_, host_weight, ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = bias_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_weight_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_bias_.FillZero(ctx.stream());
}

Status Conv2d::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Conv2d::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Conv2d input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "Conv2d input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 4, "Conv2d input"));
  if (input.dim(1) != in_channels_) {
    return Status::InvalidArgument("Conv2d input channel mismatch");
  }

  int64_t batch = input.dim(0);
  int64_t input_h = input.dim(2);
  int64_t input_w = input.dim(3);
  auto output_h = SpatialOutputSize(input_h, kernel_h_, stride_h_, padding_h_, "Conv2d height");
  if (!output_h.ok()) {
    return output_h.status();
  }
  auto output_w = SpatialOutputSize(input_w, kernel_w_, stride_w_, padding_w_, "Conv2d width");
  if (!output_w.ok()) {
    return output_w.status();
  }

  cached_input_ = input;
  last_batch_ = batch;
  last_input_h_ = input_h;
  last_input_w_ = input_w;
  last_output_h_ = output_h.value();
  last_output_w_ = output_w.value();
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_,
                                           {batch, out_channels_, last_output_h_, last_output_w_},
                                           dtype_, ctx.stream()));
  int64_t total = forward_output_.numel();
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchConv2dForwardKernel(
        ctx, dtype_, input, weight_, bias_, &forward_output_, blocks.value(), total, batch,
        in_channels_, input_h, input_w, out_channels_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        padding_h_, padding_w_, last_output_h_, last_output_w_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Conv2d::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Conv2d grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "Conv2d grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 4, "Conv2d grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != out_channels_ ||
      grad_output.dim(2) != last_output_h_ || grad_output.dim(3) != last_output_w_) {
    return Status::InvalidArgument("Conv2d grad_output shape mismatch");
  }
  if (!cached_input_.defined()) {
    return Status::RuntimeError("Conv2d backward called before forward");
  }

  bool need_grad_input = grad_input != nullptr;
  if (need_grad_input) {
    DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(
        &backward_output_, {last_batch_, in_channels_, last_input_h_, last_input_w_}, dtype_,
        ctx.stream()));
  }

  int64_t input_total = last_batch_ * in_channels_ * last_input_h_ * last_input_w_;
  int64_t weight_total = out_channels_ * in_channels_ * kernel_h_ * kernel_w_;
  if (last_batch_ == 0) {
    DLCUDA_RETURN_IF_ERROR(grad_weight_.FillZero(ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(grad_bias_.FillZero(ctx.stream()));
    if (need_grad_input) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }

  if (need_grad_input) {
    auto input_blocks = detail::BlocksForElements(input_total, kCudaThreads);
    if (!input_blocks.ok()) {
      return input_blocks.status();
    }
    if (input_blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchConv2dBackwardInputKernel(
          ctx, dtype_, grad_output, weight_, &backward_output_, input_blocks.value(), input_total,
          last_batch_, in_channels_, last_input_h_, last_input_w_, out_channels_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_, last_output_h_, last_output_w_));
    }
  }

  auto weight_blocks = detail::BlocksForElements(weight_total, kCudaThreads);
  if (!weight_blocks.ok()) {
    return weight_blocks.status();
  }
  if (weight_blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchConv2dBackwardWeightKernel(
        ctx, dtype_, cached_input_, grad_output, &grad_weight_, weight_blocks.value(), weight_total,
        last_batch_, in_channels_, last_input_h_, last_input_w_, out_channels_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, padding_h_, padding_w_, last_output_h_, last_output_w_));
  }

  auto bias_rows = detail::RowsForGrid(out_channels_, "conv bias");
  if (!bias_rows.ok()) {
    return bias_rows.status();
  }
  if (bias_rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(
        LaunchConv2dBackwardBiasKernel(ctx, dtype_, grad_output, &grad_bias_, bias_rows.value(),
                                       last_batch_, out_channels_, last_output_h_, last_output_w_));
  }

  if (need_grad_input) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void Conv2d::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "weight"), &weight_, &grad_weight_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "bias"), &bias_, &grad_bias_});
}

} // namespace dlcuda
