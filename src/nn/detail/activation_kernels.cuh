#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename Codec>
__global__ void ReLUForwardKernel(const typename Codec::Storage *input,
                                  typename Codec::Storage *output, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float v = Codec::Load(input, idx);
    Codec::Store(output, idx, v > 0.0f ? v : 0.0f);
  }
}

template <typename Codec>
__global__ void ReLUBackwardKernel(const typename Codec::Storage *grad_output,
                                   const typename Codec::Storage *cached_input,
                                   typename Codec::Storage *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = Codec::Load(cached_input, idx) > 0.0f ? Codec::Load(grad_output, idx) : 0.0f;
    Codec::Store(grad_input, idx, value);
  }
}

template <typename Codec>
__global__ void SigmoidForwardKernel(const typename Codec::Storage *input,
                                     typename Codec::Storage *output, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = Codec::Load(input, idx);
    Codec::Store(output, idx, 1.0f / (1.0f + expf(-x)));
  }
}

template <typename Codec>
__global__ void SigmoidBackwardKernel(const typename Codec::Storage *grad_output,
                                      const typename Codec::Storage *cached_output,
                                      typename Codec::Storage *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float s = Codec::Load(cached_output, idx);
    Codec::Store(grad_input, idx, Codec::Load(grad_output, idx) * s * (1.0f - s));
  }
}

template <typename Codec>
__global__ void SoftmaxForwardKernel(const typename Codec::Storage *input,
                                     typename Codec::Storage *output, int64_t num_rows,
                                     int64_t row_width) {
  __shared__ typename CudaBlockReduce::TempStorage max_storage;
  __shared__ typename CudaBlockReduce::TempStorage sum_storage;
  __shared__ float row_max_shared;
  __shared__ float row_sum_shared;

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  const typename Codec::Storage *in_row = input + row * row_width;
  typename Codec::Storage *out_row = output + row * row_width;

  float local_max = -FLT_MAX;
  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    local_max = fmaxf(local_max, Codec::Load(in_row, c));
  }
  float row_max = CudaBlockReduce(max_storage).Reduce(local_max, cub::Max());
  if (tid == 0) {
    row_max_shared = row_max;
  }
  __syncthreads();
  row_max = row_max_shared;

  float local_sum = 0.0f;
  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    float e = expf(Codec::Load(in_row, c) - row_max);
    local_sum += e;
  }
  float row_sum = CudaBlockReduce(sum_storage).Sum(local_sum);
  if (tid == 0) {
    row_sum_shared = row_sum;
  }
  __syncthreads();
  row_sum = row_sum_shared;
  float inv_sum = 1.0f / (row_sum + 1e-20f);

  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    Codec::Store(out_row, c, expf(Codec::Load(in_row, c) - row_max) * inv_sum);
  }
}

template <typename Codec>
__global__ void SoftmaxBackwardKernel(const typename Codec::Storage *grad_output,
                                      const typename Codec::Storage *softmax_output,
                                      typename Codec::Storage *grad_input, int64_t num_rows,
                                      int64_t row_width) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;
  __shared__ float dot_shared;

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  const typename Codec::Storage *dy = grad_output + row * row_width;
  const typename Codec::Storage *s = softmax_output + row * row_width;
  typename Codec::Storage *dx = grad_input + row * row_width;

  float local_dot = 0.0f;
  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    local_dot += Codec::Load(dy, c) * Codec::Load(s, c);
  }
  float dot = CudaBlockReduce(reduce_storage).Sum(local_dot);
  if (tid == 0) {
    dot_shared = dot;
  }
  __syncthreads();
  dot = dot_shared;

  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    float softmax_value = Codec::Load(s, c);
    Codec::Store(dx, c, softmax_value * (Codec::Load(dy, c) - dot));
  }
}
template <typename Codec>
__global__ void GELUForwardKernel(const typename Codec::Storage *input,
                                  typename Codec::Storage *output, int64_t size) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = Codec::Load(input, idx);
    Codec::Store(output, idx, 0.5f * x * (1.0f + erff(x * kInvSqrt2)));
  }
}

template <typename Codec>
__global__ void GELUBackwardKernel(const typename Codec::Storage *grad_output,
                                   const typename Codec::Storage *cached_input,
                                   typename Codec::Storage *grad_input, int64_t size) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = Codec::Load(cached_input, idx);
    float cdf = 0.5f * (1.0f + erff(x * kInvSqrt2));
    float pdf_term = x * expf(-0.5f * x * x) * kInvSqrt2Pi;
    Codec::Store(grad_input, idx, Codec::Load(grad_output, idx) * (cdf + pdf_term));
  }
}
template <typename Codec>
Status LaunchUnaryForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                int blocks, const char *op_name) {
  if (std::string(op_name) == "ReLU") {
    ReLUForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
        input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
        input.numel());
  } else {
    SigmoidForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
        input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
        input.numel());
  }
  return detail::CheckKernelLaunch(std::string(op_name) + " forward kernel");
}

Status LaunchReLUForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                               int blocks) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchUnaryForwardKernel<detail::Float32Codec>(ctx, input, output, blocks, "ReLU");
  case DType::kFloat16:
    return LaunchUnaryForwardKernel<detail::Float16Codec>(ctx, input, output, blocks, "ReLU");
  case DType::kBFloat16:
    return LaunchUnaryForwardKernel<detail::BFloat16Codec>(ctx, input, output, blocks, "ReLU");
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ReLU does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

Status LaunchSigmoidForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  int blocks) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchUnaryForwardKernel<detail::Float32Codec>(ctx, input, output, blocks, "Sigmoid");
  case DType::kFloat16:
    return LaunchUnaryForwardKernel<detail::Float16Codec>(ctx, input, output, blocks, "Sigmoid");
  case DType::kBFloat16:
    return LaunchUnaryForwardKernel<detail::BFloat16Codec>(ctx, input, output, blocks, "Sigmoid");
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Sigmoid does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchReLUBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                const Tensor &cached_input, Tensor *grad_input, int blocks) {
  ReLUBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(),
      cached_input.data_as<typename Codec::Storage>(),
      grad_input->data_as<typename Codec::Storage>(), grad_output.numel());
  return detail::CheckKernelLaunch("ReLU backward kernel");
}

Status LaunchReLUBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                const Tensor &cached_input, Tensor *grad_input, int blocks) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchReLUBackwardKernel<detail::Float32Codec>(ctx, grad_output, cached_input,
                                                          grad_input, blocks);
  case DType::kFloat16:
    return LaunchReLUBackwardKernel<detail::Float16Codec>(ctx, grad_output, cached_input,
                                                          grad_input, blocks);
  case DType::kBFloat16:
    return LaunchReLUBackwardKernel<detail::BFloat16Codec>(ctx, grad_output, cached_input,
                                                           grad_input, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("ReLU backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
}

template <typename Codec>
Status LaunchSigmoidBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &cached_output, Tensor *grad_input, int blocks) {
  SigmoidBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(),
      cached_output.data_as<typename Codec::Storage>(),
      grad_input->data_as<typename Codec::Storage>(), grad_output.numel());
  return detail::CheckKernelLaunch("Sigmoid backward kernel");
}

Status LaunchSigmoidBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &cached_output, Tensor *grad_input, int blocks) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchSigmoidBackwardKernel<detail::Float32Codec>(ctx, grad_output, cached_output,
                                                             grad_input, blocks);
  case DType::kFloat16:
    return LaunchSigmoidBackwardKernel<detail::Float16Codec>(ctx, grad_output, cached_output,
                                                             grad_input, blocks);
  case DType::kBFloat16:
    return LaunchSigmoidBackwardKernel<detail::BFloat16Codec>(ctx, grad_output, cached_output,
                                                              grad_input, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Sigmoid backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
}

template <typename Codec>
Status LaunchSoftmaxForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  int rows, int64_t row_width) {
  SoftmaxForwardKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      input.dim(0), row_width);
  return detail::CheckKernelLaunch("Softmax forward kernel");
}

Status LaunchSoftmaxForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  int rows, int64_t row_width) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchSoftmaxForwardKernel<detail::Float32Codec>(ctx, input, output, rows, row_width);
  case DType::kFloat16:
    return LaunchSoftmaxForwardKernel<detail::Float16Codec>(ctx, input, output, rows, row_width);
  case DType::kBFloat16:
    return LaunchSoftmaxForwardKernel<detail::BFloat16Codec>(ctx, input, output, rows, row_width);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Softmax does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchSoftmaxBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &cached_output, Tensor *grad_input, int rows,
                                   int64_t num_rows, int64_t row_width) {
  SoftmaxBackwardKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(),
      cached_output.data_as<typename Codec::Storage>(),
      grad_input->data_as<typename Codec::Storage>(), num_rows, row_width);
  return detail::CheckKernelLaunch("Softmax backward kernel");
}

Status LaunchSoftmaxBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &cached_output, Tensor *grad_input, int rows,
                                   int64_t num_rows, int64_t row_width) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchSoftmaxBackwardKernel<detail::Float32Codec>(ctx, grad_output, cached_output,
                                                             grad_input, rows, num_rows, row_width);
  case DType::kFloat16:
    return LaunchSoftmaxBackwardKernel<detail::Float16Codec>(ctx, grad_output, cached_output,
                                                             grad_input, rows, num_rows, row_width);
  case DType::kBFloat16:
    return LaunchSoftmaxBackwardKernel<detail::BFloat16Codec>(
        ctx, grad_output, cached_output, grad_input, rows, num_rows, row_width);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Softmax backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
}
template <typename Codec>
Status LaunchGELUForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                               int blocks) {
  GELUForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      input.numel());
  return detail::CheckKernelLaunch("GELU forward kernel");
}

Status LaunchGELUForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                               int blocks) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchGELUForwardKernel<detail::Float32Codec>(ctx, input, output, blocks);
  case DType::kFloat16:
    return LaunchGELUForwardKernel<detail::Float16Codec>(ctx, input, output, blocks);
  case DType::kBFloat16:
    return LaunchGELUForwardKernel<detail::BFloat16Codec>(ctx, input, output, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("GELU does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchGELUBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                const Tensor &cached_input, Tensor *grad_input, int blocks) {
  GELUBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(),
      cached_input.data_as<typename Codec::Storage>(),
      grad_input->data_as<typename Codec::Storage>(), grad_output.numel());
  return detail::CheckKernelLaunch("GELU backward kernel");
}

Status LaunchGELUBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                const Tensor &cached_input, Tensor *grad_input, int blocks) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchGELUBackwardKernel<detail::Float32Codec>(ctx, grad_output, cached_input,
                                                          grad_input, blocks);
  case DType::kFloat16:
    return LaunchGELUBackwardKernel<detail::Float16Codec>(ctx, grad_output, cached_input,
                                                          grad_input, blocks);
  case DType::kBFloat16:
    return LaunchGELUBackwardKernel<detail::BFloat16Codec>(ctx, grad_output, cached_input,
                                                           grad_input, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("GELU backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
}

} // namespace
} // namespace dlcuda
