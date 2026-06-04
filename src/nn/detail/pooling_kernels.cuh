#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename Codec>
__global__ void
MaxPool2dForwardKernel(const typename Codec::Storage *input, typename Codec::Storage *output,
                       int32_t *argmax_indices, int64_t total, int64_t batch, int64_t channels,
                       int64_t input_h, int64_t input_w, int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
                       int64_t output_h, int64_t output_w) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t ow = idx % output_w;
  int64_t tmp = idx / output_w;
  int64_t oh = tmp % output_h;
  tmp /= output_h;
  int64_t c = tmp % channels;
  int64_t n = tmp / channels;
  if (n >= batch) {
    return;
  }

  float max_value = -FLT_MAX;
  int64_t max_index = -1;
  for (int64_t kh = 0; kh < kernel_h; ++kh) {
    int64_t ih = oh * stride_h + kh - padding_h;
    if (ih < 0 || ih >= input_h) {
      continue;
    }
    for (int64_t kw = 0; kw < kernel_w; ++kw) {
      int64_t iw = ow * stride_w + kw - padding_w;
      if (iw < 0 || iw >= input_w) {
        continue;
      }
      int64_t input_index = ((n * channels + c) * input_h + ih) * input_w + iw;
      float value = Codec::Load(input, input_index);
      if (value > max_value) {
        max_value = value;
        max_index = input_index;
      }
    }
  }

  argmax_indices[idx] = static_cast<int32_t>(max_index);
  Codec::Store(output, idx, max_index >= 0 ? max_value : 0.0f);
}

template <typename Codec>
__global__ void MaxPool2dBackwardKernel(const typename Codec::Storage *grad_output,
                                        const int32_t *argmax_indices,
                                        typename Codec::Storage *grad_input, int64_t total,
                                        int64_t batch, int64_t channels, int64_t input_h,
                                        int64_t input_w, int64_t output_h, int64_t output_w) {
  int64_t input_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (input_index >= total) {
    return;
  }

  int64_t iw = input_index % input_w;
  int64_t tmp = input_index / input_w;
  int64_t ih = tmp % input_h;
  tmp /= input_h;
  int64_t c = tmp % channels;
  int64_t n = tmp / channels;
  if (n >= batch) {
    return;
  }

  float sum = 0.0f;
  for (int64_t oh = 0; oh < output_h; ++oh) {
    for (int64_t ow = 0; ow < output_w; ++ow) {
      int64_t output_index = ((n * channels + c) * output_h + oh) * output_w + ow;
      if (static_cast<int64_t>(argmax_indices[output_index]) == input_index) {
        sum += Codec::Load(grad_output, output_index);
      }
    }
  }
  Codec::Store(grad_input, input_index, sum);
}
template <typename Codec>
Status LaunchMaxPool2dForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                    Tensor *argmax_indices, int blocks, int64_t total,
                                    int64_t batch, int64_t channels, int64_t input_h,
                                    int64_t input_w, int64_t kernel_h, int64_t kernel_w,
                                    int64_t stride_h, int64_t stride_w, int64_t padding_h,
                                    int64_t padding_w, int64_t output_h, int64_t output_w) {
  MaxPool2dForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      argmax_indices->data_as<int32_t>(), total, batch, channels, input_h, input_w, kernel_h,
      kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  return detail::CheckKernelLaunch("MaxPool2d forward kernel");
}

Status LaunchMaxPool2dForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                    Tensor *argmax_indices, int blocks, int64_t total,
                                    int64_t batch, int64_t channels, int64_t input_h,
                                    int64_t input_w, int64_t kernel_h, int64_t kernel_w,
                                    int64_t stride_h, int64_t stride_w, int64_t padding_h,
                                    int64_t padding_w, int64_t output_h, int64_t output_w) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchMaxPool2dForwardKernel<detail::Float32Codec>(
        ctx, input, output, argmax_indices, blocks, total, batch, channels, input_h, input_w,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kFloat16:
    return LaunchMaxPool2dForwardKernel<detail::Float16Codec>(
        ctx, input, output, argmax_indices, blocks, total, batch, channels, input_h, input_w,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kBFloat16:
    return LaunchMaxPool2dForwardKernel<detail::BFloat16Codec>(
        ctx, input, output, argmax_indices, blocks, total, batch, channels, input_h, input_w,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("MaxPool2d does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchMaxPool2dBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                     const Tensor &argmax_indices, Tensor *grad_input, int blocks,
                                     int64_t total, int64_t batch, int64_t channels,
                                     int64_t input_h, int64_t input_w, int64_t output_h,
                                     int64_t output_w) {
  MaxPool2dBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), argmax_indices.data_as<int32_t>(),
      grad_input->data_as<typename Codec::Storage>(), total, batch, channels, input_h, input_w,
      output_h, output_w);
  return detail::CheckKernelLaunch("MaxPool2d backward kernel");
}

Status LaunchMaxPool2dBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                     const Tensor &argmax_indices, Tensor *grad_input, int blocks,
                                     int64_t total, int64_t batch, int64_t channels,
                                     int64_t input_h, int64_t input_w, int64_t output_h,
                                     int64_t output_w) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchMaxPool2dBackwardKernel<detail::Float32Codec>(
        ctx, grad_output, argmax_indices, grad_input, blocks, total, batch, channels, input_h,
        input_w, output_h, output_w);
  case DType::kFloat16:
    return LaunchMaxPool2dBackwardKernel<detail::Float16Codec>(
        ctx, grad_output, argmax_indices, grad_input, blocks, total, batch, channels, input_h,
        input_w, output_h, output_w);
  case DType::kBFloat16:
    return LaunchMaxPool2dBackwardKernel<detail::BFloat16Codec>(
        ctx, grad_output, argmax_indices, grad_input, blocks, total, batch, channels, input_h,
        input_w, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("MaxPool2d backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
}

} // namespace
} // namespace dlcuda
