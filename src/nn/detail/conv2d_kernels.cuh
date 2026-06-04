#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename Codec>
__global__ void
Conv2dForwardKernel(const typename Codec::Storage *input, const typename Codec::Storage *weight,
                    const typename Codec::Storage *bias, typename Codec::Storage *output,
                    int64_t total, int64_t batch, int64_t in_channels, int64_t input_h,
                    int64_t input_w, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
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
  int64_t oc = tmp % out_channels;
  int64_t n = tmp / out_channels;
  if (n >= batch) {
    return;
  }

  float sum = Codec::Load(bias, oc);
  for (int64_t ic = 0; ic < in_channels; ++ic) {
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
        int64_t input_index = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
        int64_t weight_index = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
        sum += Codec::Load(input, input_index) * Codec::Load(weight, weight_index);
      }
    }
  }

  Codec::Store(output, idx, sum);
}

template <typename Codec>
__global__ void Conv2dBackwardInputKernel(const typename Codec::Storage *grad_output,
                                          const typename Codec::Storage *weight,
                                          typename Codec::Storage *grad_input, int64_t total,
                                          int64_t batch, int64_t in_channels, int64_t input_h,
                                          int64_t input_w, int64_t out_channels, int64_t kernel_h,
                                          int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                          int64_t padding_h, int64_t padding_w, int64_t output_h,
                                          int64_t output_w) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t iw = idx % input_w;
  int64_t tmp = idx / input_w;
  int64_t ih = tmp % input_h;
  tmp /= input_h;
  int64_t ic = tmp % in_channels;
  int64_t n = tmp / in_channels;
  if (n >= batch) {
    return;
  }

  float sum = 0.0f;
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      int64_t oh_unstrided = ih + padding_h - kh;
      if (oh_unstrided < 0 || oh_unstrided % stride_h != 0) {
        continue;
      }
      int64_t oh = oh_unstrided / stride_h;
      if (oh < 0 || oh >= output_h) {
        continue;
      }
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        int64_t ow_unstrided = iw + padding_w - kw;
        if (ow_unstrided < 0 || ow_unstrided % stride_w != 0) {
          continue;
        }
        int64_t ow = ow_unstrided / stride_w;
        if (ow < 0 || ow >= output_w) {
          continue;
        }
        int64_t output_index = ((n * out_channels + oc) * output_h + oh) * output_w + ow;
        int64_t weight_index = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
        sum += Codec::Load(grad_output, output_index) * Codec::Load(weight, weight_index);
      }
    }
  }

  Codec::Store(grad_input, idx, sum);
}

template <typename Codec>
__global__ void Conv2dBackwardWeightKernel(const typename Codec::Storage *input,
                                           const typename Codec::Storage *grad_output,
                                           float *grad_weight, int64_t total, int64_t batch,
                                           int64_t in_channels, int64_t input_h, int64_t input_w,
                                           int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
                                           int64_t stride_h, int64_t stride_w, int64_t padding_h,
                                           int64_t padding_w, int64_t output_h, int64_t output_w) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t kw = idx % kernel_w;
  int64_t tmp = idx / kernel_w;
  int64_t kh = tmp % kernel_h;
  tmp /= kernel_h;
  int64_t ic = tmp % in_channels;
  int64_t oc = tmp / in_channels;
  if (oc >= out_channels) {
    return;
  }

  float sum = 0.0f;
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t oh = 0; oh < output_h; ++oh) {
      int64_t ih = oh * stride_h + kh - padding_h;
      if (ih < 0 || ih >= input_h) {
        continue;
      }
      for (int64_t ow = 0; ow < output_w; ++ow) {
        int64_t iw = ow * stride_w + kw - padding_w;
        if (iw < 0 || iw >= input_w) {
          continue;
        }
        int64_t input_index = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
        int64_t output_index = ((n * out_channels + oc) * output_h + oh) * output_w + ow;
        sum += Codec::Load(input, input_index) * Codec::Load(grad_output, output_index);
      }
    }
  }

  grad_weight[idx] = sum;
}

template <typename Codec>
__global__ void Conv2dBackwardBiasKernel(const typename Codec::Storage *grad_output,
                                         float *grad_bias, int64_t batch, int64_t out_channels,
                                         int64_t output_h, int64_t output_w) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;

  int64_t oc = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (oc >= out_channels) {
    return;
  }

  float sum = 0.0f;
  int64_t elements_per_channel = batch * output_h * output_w;
  for (int64_t i = tid; i < elements_per_channel; i += blockDim.x) {
    int64_t ow = i % output_w;
    int64_t tmp = i / output_w;
    int64_t oh = tmp % output_h;
    int64_t n = tmp / output_h;
    int64_t output_index = ((n * out_channels + oc) * output_h + oh) * output_w + ow;
    sum += Codec::Load(grad_output, output_index);
  }
  float block_sum = CudaBlockReduce(reduce_storage).Sum(sum);
  if (tid == 0) {
    grad_bias[oc] = block_sum;
  }
}
template <typename Codec>
Status LaunchConv2dForwardKernel(RuntimeContext &ctx, const Tensor &input, const Tensor &weight,
                                 const Tensor &bias, Tensor *output, int blocks, int64_t total,
                                 int64_t batch, int64_t in_channels, int64_t input_h,
                                 int64_t input_w, int64_t out_channels, int64_t kernel_h,
                                 int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                 int64_t padding_h, int64_t padding_w, int64_t output_h,
                                 int64_t output_w) {
  Conv2dForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), weight.data_as<typename Codec::Storage>(),
      bias.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(), total,
      batch, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w, output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d forward kernel");
}

Status LaunchConv2dForwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                 const Tensor &weight, const Tensor &bias, Tensor *output,
                                 int blocks, int64_t total, int64_t batch, int64_t in_channels,
                                 int64_t input_h, int64_t input_w, int64_t out_channels,
                                 int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                 int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                 int64_t output_h, int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dForwardKernel<detail::Float32Codec>(
        ctx, input, weight, bias, output, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kFloat16:
    return LaunchConv2dForwardKernel<detail::Float16Codec>(
        ctx, input, weight, bias, output, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kBFloat16:
    return LaunchConv2dForwardKernel<detail::BFloat16Codec>(
        ctx, input, weight, bias, output, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d does not support dtype " + std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchConv2dBackwardInputKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                       const Tensor &weight, Tensor *grad_input, int blocks,
                                       int64_t total, int64_t batch, int64_t in_channels,
                                       int64_t input_h, int64_t input_w, int64_t out_channels,
                                       int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                       int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                       int64_t output_h, int64_t output_w) {
  Conv2dBackwardInputKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), weight.data_as<typename Codec::Storage>(),
      grad_input->data_as<typename Codec::Storage>(), total, batch, in_channels, input_h, input_w,
      out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
      output_w);
  return detail::CheckKernelLaunch("Conv2d backward-input kernel");
}

Status LaunchConv2dBackwardInputKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                       const Tensor &weight, Tensor *grad_input, int blocks,
                                       int64_t total, int64_t batch, int64_t in_channels,
                                       int64_t input_h, int64_t input_w, int64_t out_channels,
                                       int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                       int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                       int64_t output_h, int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dBackwardInputKernel<detail::Float32Codec>(
        ctx, grad_output, weight, grad_input, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kFloat16:
    return LaunchConv2dBackwardInputKernel<detail::Float16Codec>(
        ctx, grad_output, weight, grad_input, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kBFloat16:
    return LaunchConv2dBackwardInputKernel<detail::BFloat16Codec>(
        ctx, grad_output, weight, grad_input, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d backward-input does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchConv2dBackwardWeightKernel(RuntimeContext &ctx, const Tensor &input,
                                        const Tensor &grad_output, Tensor *grad_weight, int blocks,
                                        int64_t total, int64_t batch, int64_t in_channels,
                                        int64_t input_h, int64_t input_w, int64_t out_channels,
                                        int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                        int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                        int64_t output_h, int64_t output_w) {
  Conv2dBackwardWeightKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), grad_output.data_as<typename Codec::Storage>(),
      grad_weight->data_as<float>(), total, batch, in_channels, input_h, input_w, out_channels,
      kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d backward-weight kernel");
}

Status LaunchConv2dBackwardWeightKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                        const Tensor &grad_output, Tensor *grad_weight, int blocks,
                                        int64_t total, int64_t batch, int64_t in_channels,
                                        int64_t input_h, int64_t input_w, int64_t out_channels,
                                        int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                        int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                        int64_t output_h, int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dBackwardWeightKernel<detail::Float32Codec>(
        ctx, input, grad_output, grad_weight, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kFloat16:
    return LaunchConv2dBackwardWeightKernel<detail::Float16Codec>(
        ctx, input, grad_output, grad_weight, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kBFloat16:
    return LaunchConv2dBackwardWeightKernel<detail::BFloat16Codec>(
        ctx, input, grad_output, grad_weight, blocks, total, batch, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_h,
        output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d backward-weight does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchConv2dBackwardBiasKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                      Tensor *grad_bias, int rows, int64_t batch,
                                      int64_t out_channels, int64_t output_h, int64_t output_w) {
  Conv2dBackwardBiasKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), grad_bias->data_as<float>(), batch,
      out_channels, output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d backward-bias kernel");
}

Status LaunchConv2dBackwardBiasKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                      Tensor *grad_bias, int rows, int64_t batch,
                                      int64_t out_channels, int64_t output_h, int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dBackwardBiasKernel<detail::Float32Codec>(
        ctx, grad_output, grad_bias, rows, batch, out_channels, output_h, output_w);
  case DType::kFloat16:
    return LaunchConv2dBackwardBiasKernel<detail::Float16Codec>(
        ctx, grad_output, grad_bias, rows, batch, out_channels, output_h, output_w);
  case DType::kBFloat16:
    return LaunchConv2dBackwardBiasKernel<detail::BFloat16Codec>(
        ctx, grad_output, grad_bias, rows, batch, out_channels, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d backward-bias does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

} // namespace
} // namespace dlcuda
