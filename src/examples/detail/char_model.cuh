#pragma once

#include "char_metadata.cuh"

namespace dlcuda {
namespace {

constexpr int kCharContextWidth = 5;

inline Status EnsureFloat2DTensor(Tensor *tensor, int64_t rows, int64_t cols, cudaStream_t stream) {
  if (tensor == nullptr) {
    return Status::InvalidArgument("EnsureFloat2DTensor received null tensor pointer");
  }
  if (tensor->defined() && tensor->dtype() == DType::kFloat32 && tensor->rank() == 2 &&
      tensor->dim(0) == rows && tensor->dim(1) == cols) {
    return Status::Ok();
  }
  auto allocated = Tensor::AllocateAsync({rows, cols}, DType::kFloat32, stream);
  if (!allocated.ok()) {
    return allocated.status();
  }
  *tensor = allocated.value();
  return Status::Ok();
}

__global__ void CausalConv1dForwardKernel(const float *input, const float *weight,
                                          const float *bias, float *output, int64_t seq_len,
                                          int64_t channels, int64_t kernel_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq_len * channels;
  if (idx >= total) {
    return;
  }

  int64_t time = idx / channels;
  int64_t out_ch = idx % channels;
  float sum = bias[out_ch];
  for (int64_t k = 0; k < kernel_width; ++k) {
    int64_t src_time = time - k;
    if (src_time < 0) {
      continue;
    }
    for (int64_t in_ch = 0; in_ch < channels; ++in_ch) {
      int64_t weight_idx = (k * channels + in_ch) * channels + out_ch;
      sum += input[src_time * channels + in_ch] * weight[weight_idx];
    }
  }
  output[idx] = sum;
}

__global__ void CausalConv1dBackwardInputKernel(const float *grad_output, const float *weight,
                                                float *grad_input, int64_t seq_len,
                                                int64_t channels, int64_t kernel_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq_len * channels;
  if (idx >= total) {
    return;
  }

  int64_t time = idx / channels;
  int64_t in_ch = idx % channels;
  float sum = 0.0f;
  int64_t max_future = time + kernel_width - 1;
  if (max_future >= seq_len) {
    max_future = seq_len - 1;
  }
  for (int64_t out_time = time; out_time <= max_future; ++out_time) {
    int64_t k = out_time - time;
    for (int64_t out_ch = 0; out_ch < channels; ++out_ch) {
      int64_t weight_idx = (k * channels + in_ch) * channels + out_ch;
      sum += grad_output[out_time * channels + out_ch] * weight[weight_idx];
    }
  }
  grad_input[idx] = sum;
}

__global__ void CausalConv1dBackwardWeightKernel(const float *input, const float *grad_output,
                                                 float *grad_weight, int64_t seq_len,
                                                 int64_t channels, int64_t kernel_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = kernel_width * channels * channels;
  if (idx >= total) {
    return;
  }

  int64_t out_ch = idx % channels;
  int64_t in_ch = (idx / channels) % channels;
  int64_t k = idx / (channels * channels);
  float sum = 0.0f;
  for (int64_t time = k; time < seq_len; ++time) {
    sum += input[(time - k) * channels + in_ch] * grad_output[time * channels + out_ch];
  }
  grad_weight[idx] = sum;
}

__global__ void CausalConv1dBackwardBiasKernel(const float *grad_output, float *grad_bias,
                                               int64_t seq_len, int64_t channels) {
  int64_t out_ch = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_ch >= channels) {
    return;
  }

  float sum = 0.0f;
  for (int64_t time = 0; time < seq_len; ++time) {
    sum += grad_output[time * channels + out_ch];
  }
  grad_bias[out_ch] = sum;
}

class CausalConv1d : public Module {
public:
  CausalConv1d(int64_t channels, int64_t kernel_width, RuntimeContext &ctx)
      : channels_(channels), kernel_width_(kernel_width) {
    if (channels_ <= 0 || kernel_width_ <= 0) {
      init_status_ = Status::InvalidArgument("CausalConv1d dimensions must be positive");
      return;
    }

    auto weight =
        Tensor::AllocateAsync({kernel_width_, channels_, channels_}, DType::kFloat32, ctx.stream());
    if (!weight.ok()) {
      init_status_ = weight.status();
      return;
    }
    auto bias = Tensor::AllocateAsync({channels_}, DType::kFloat32, ctx.stream());
    if (!bias.ok()) {
      init_status_ = bias.status();
      return;
    }
    auto grad_weight =
        Tensor::AllocateAsync({kernel_width_, channels_, channels_}, DType::kFloat32, ctx.stream());
    if (!grad_weight.ok()) {
      init_status_ = grad_weight.status();
      return;
    }
    auto grad_bias = Tensor::AllocateAsync({channels_}, DType::kFloat32, ctx.stream());
    if (!grad_bias.ok()) {
      init_status_ = grad_bias.status();
      return;
    }

    weight_ = weight.value();
    bias_ = bias.value();
    grad_weight_ = grad_weight.value();
    grad_bias_ = grad_bias.value();

    std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
    std::normal_distribution<float> dist(
        0.0f, std::sqrt(2.0f / static_cast<float>(channels_ * kernel_width_)));
    std::vector<float> host_weight(static_cast<size_t>(kernel_width_ * channels_ * channels_));
    for (float &value : host_weight) {
      value = dist(rng);
    }

    init_status_ =
        weight_.CopyFromHost(host_weight.data(), host_weight.size() * sizeof(float), ctx.stream());
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

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override {
    if (!init_status_.ok()) {
      return init_status_;
    }
    if (output == nullptr) {
      return Status::InvalidArgument("CausalConv1d::Forward output is null");
    }
    if (!input.defined()) {
      return Status::InvalidArgument("CausalConv1d input is undefined");
    }
    if (input.dtype() != DType::kFloat32) {
      return Status::InvalidArgument("CausalConv1d input must be float32");
    }
    if (input.rank() != 2) {
      return Status::InvalidArgument("CausalConv1d input must have rank 2");
    }
    if (input.dim(1) != channels_) {
      return Status::InvalidArgument("CausalConv1d input channel mismatch");
    }

    last_seq_len_ = input.dim(0);
    cached_input_ = input;
    DLCUDA_RETURN_IF_ERROR(
        EnsureFloat2DTensor(&forward_output_, last_seq_len_, channels_, ctx.stream()));

    int64_t total = last_seq_len_ * channels_;
    auto blocks = detail::BlocksForElements(total, kExampleThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      CausalConv1dForwardKernel<<<blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          input.data_as<float>(), weight_.data_as<float>(), bias_.data_as<float>(),
          forward_output_.data_as<float>(), last_seq_len_, channels_, kernel_width_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dForwardKernel"));
    }

    *output = forward_output_;
    return Status::Ok();
  }

  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override {
    if (!init_status_.ok()) {
      return init_status_;
    }
    if (!grad_output.defined()) {
      return Status::InvalidArgument("CausalConv1d grad_output is undefined");
    }
    if (grad_output.dtype() != DType::kFloat32) {
      return Status::InvalidArgument("CausalConv1d grad_output must be float32");
    }
    if (grad_output.rank() != 2 || grad_output.dim(0) != last_seq_len_ ||
        grad_output.dim(1) != channels_) {
      return Status::InvalidArgument("CausalConv1d grad_output shape mismatch");
    }
    if (!cached_input_.defined()) {
      return Status::RuntimeError("CausalConv1d backward called before forward");
    }

    if (grad_input != nullptr) {
      DLCUDA_RETURN_IF_ERROR(
          EnsureFloat2DTensor(&backward_output_, last_seq_len_, channels_, ctx.stream()));
    }

    int64_t input_total = last_seq_len_ * channels_;
    auto input_blocks = detail::BlocksForElements(input_total, kExampleThreads);
    if (!input_blocks.ok()) {
      return input_blocks.status();
    }
    if (grad_input != nullptr && input_blocks.value() > 0) {
      CausalConv1dBackwardInputKernel<<<input_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          grad_output.data_as<float>(), weight_.data_as<float>(), backward_output_.data_as<float>(),
          last_seq_len_, channels_, kernel_width_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dBackwardInputKernel"));
    }

    int64_t weight_total = kernel_width_ * channels_ * channels_;
    auto weight_blocks = detail::BlocksForElements(weight_total, kExampleThreads);
    if (!weight_blocks.ok()) {
      return weight_blocks.status();
    }
    if (weight_blocks.value() > 0) {
      CausalConv1dBackwardWeightKernel<<<weight_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          cached_input_.data_as<float>(), grad_output.data_as<float>(),
          grad_weight_.data_as<float>(), last_seq_len_, channels_, kernel_width_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dBackwardWeightKernel"));
    }

    auto bias_blocks = detail::BlocksForElements(channels_, kExampleThreads);
    if (!bias_blocks.ok()) {
      return bias_blocks.status();
    }
    if (bias_blocks.value() > 0) {
      CausalConv1dBackwardBiasKernel<<<bias_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          grad_output.data_as<float>(), grad_bias_.data_as<float>(), last_seq_len_, channels_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dBackwardBiasKernel"));
    }

    if (grad_input != nullptr) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override {
    if (out == nullptr) {
      return;
    }
    std::string base = prefix.empty() ? std::string() : prefix + ".";
    out->push_back(ParameterRef{base + "weight", &weight_, &grad_weight_});
    out->push_back(ParameterRef{base + "bias", &bias_, &grad_bias_});
  }

private:
  Status init_status_;
  int64_t channels_ = 0;
  int64_t kernel_width_ = 0;
  int64_t last_seq_len_ = 0;
  Tensor weight_;
  Tensor bias_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
};

inline Status BuildCharModel(Sequential *model, RuntimeContext &ctx, int vocab_size, int d_model) {
  if (model == nullptr) {
    return Status::InvalidArgument("BuildCharModel requires a model pointer");
  }
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Embedding>(vocab_size, d_model, ctx)));
  DLCUDA_RETURN_IF_ERROR(
      model->Add(std::make_unique<CausalConv1d>(d_model, kCharContextWidth, ctx)));
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<GELU>()));
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Linear>(d_model, vocab_size, ctx)));
  return Status::Ok();
}

} // namespace
} // namespace dlcuda
