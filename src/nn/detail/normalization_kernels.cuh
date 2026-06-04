#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

template <typename Codec>
__global__ void
LayerNormForwardKernel(const typename Codec::Storage *input, const typename Codec::Storage *gamma,
                       const typename Codec::Storage *beta, typename Codec::Storage *output,
                       typename Codec::Storage *x_hat, float *inv_std, int64_t rows, int64_t width,
                       float eps) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;
  __shared__ float mean_shared;
  __shared__ float inv_std_shared;

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const typename Codec::Storage *in_row = input + row * width;
  typename Codec::Storage *out_row = output + row * width;
  typename Codec::Storage *xhat_row = x_hat + row * width;

  float local_sum = 0.0f;
  for (int64_t col = tid; col < width; col += blockDim.x) {
    local_sum += Codec::Load(in_row, col);
  }
  float sum = CudaBlockReduce(reduce_storage).Sum(local_sum);
  if (tid == 0) {
    mean_shared = sum / static_cast<float>(width);
  }
  __syncthreads();
  float mean = mean_shared;

  float local_var = 0.0f;
  for (int64_t col = tid; col < width; col += blockDim.x) {
    float centered = Codec::Load(in_row, col) - mean;
    local_var += centered * centered;
  }
  float var_sum = CudaBlockReduce(reduce_storage).Sum(local_var);
  if (tid == 0) {
    inv_std_shared = rsqrtf(var_sum / static_cast<float>(width) + eps);
    inv_std[row] = inv_std_shared;
  }
  __syncthreads();
  float row_inv_std = inv_std_shared;

  for (int64_t col = tid; col < width; col += blockDim.x) {
    float normalized = (Codec::Load(in_row, col) - mean) * row_inv_std;
    Codec::Store(xhat_row, col, normalized);
    Codec::Store(out_row, col, normalized * Codec::Load(gamma, col) + Codec::Load(beta, col));
  }
}

template <typename Codec>
__global__ void
LayerNormBackwardKernel(const typename Codec::Storage *grad_output,
                        const typename Codec::Storage *x_hat, const typename Codec::Storage *gamma,
                        typename Codec::Storage *grad_input, float *grad_gamma, float *grad_beta,
                        const float *inv_std, int64_t rows, int64_t width) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;
  __shared__ float mean_dy_gamma_shared;
  __shared__ float mean_dy_gamma_xhat_shared;

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const typename Codec::Storage *dy_row = grad_output + row * width;
  const typename Codec::Storage *xhat_row = x_hat + row * width;
  typename Codec::Storage *dx_row = grad_input + row * width;

  float local_sum = 0.0f;
  for (int64_t col = tid; col < width; col += blockDim.x) {
    local_sum += Codec::Load(dy_row, col) * Codec::Load(gamma, col);
  }
  float dy_gamma_sum = CudaBlockReduce(reduce_storage).Sum(local_sum);
  if (tid == 0) {
    mean_dy_gamma_shared = dy_gamma_sum / static_cast<float>(width);
  }
  __syncthreads();

  float local_xhat_sum = 0.0f;
  for (int64_t col = tid; col < width; col += blockDim.x) {
    float dy_gamma = Codec::Load(dy_row, col) * Codec::Load(gamma, col);
    local_xhat_sum += dy_gamma * Codec::Load(xhat_row, col);
  }
  float dy_gamma_xhat_sum = CudaBlockReduce(reduce_storage).Sum(local_xhat_sum);
  if (tid == 0) {
    mean_dy_gamma_xhat_shared = dy_gamma_xhat_sum / static_cast<float>(width);
  }
  __syncthreads();

  float row_inv_std = inv_std[row];
  float mean_dy_gamma = mean_dy_gamma_shared;
  float mean_dy_gamma_xhat = mean_dy_gamma_xhat_shared;
  for (int64_t col = tid; col < width; col += blockDim.x) {
    float dy = Codec::Load(dy_row, col);
    float normalized = Codec::Load(xhat_row, col);
    float dy_gamma = dy * Codec::Load(gamma, col);
    float dx = (dy_gamma - mean_dy_gamma - normalized * mean_dy_gamma_xhat) * row_inv_std;
    Codec::Store(dx_row, col, dx);
    atomicAdd(&grad_gamma[col], dy * normalized);
    atomicAdd(&grad_beta[col], dy);
  }
}

template <typename Codec>
__global__ void BatchNorm1dForwardTrainingKernel(
    const typename Codec::Storage *input, const typename Codec::Storage *gamma,
    const typename Codec::Storage *beta, typename Codec::Storage *output,
    typename Codec::Storage *x_hat, float *inv_std, float *running_mean, float *running_var,
    int64_t batch, int64_t features, float eps, float momentum) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;
  __shared__ float mean_shared;
  __shared__ float inv_std_shared;

  int64_t feature = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (feature >= features) {
    return;
  }

  float local_sum = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    local_sum += Codec::Load(input, n * features + feature);
  }
  float sum = CudaBlockReduce(reduce_storage).Sum(local_sum);
  if (tid == 0) {
    mean_shared = sum / static_cast<float>(batch);
  }
  __syncthreads();
  float mean = mean_shared;

  float local_var = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    float centered = Codec::Load(input, n * features + feature) - mean;
    local_var += centered * centered;
  }
  float var_sum = CudaBlockReduce(reduce_storage).Sum(local_var);
  if (tid == 0) {
    float variance = var_sum / static_cast<float>(batch);
    inv_std_shared = rsqrtf(variance + eps);
    inv_std[feature] = inv_std_shared;
    running_mean[feature] = (1.0f - momentum) * running_mean[feature] + momentum * mean;
    running_var[feature] = (1.0f - momentum) * running_var[feature] + momentum * variance;
  }
  __syncthreads();
  float feature_inv_std = inv_std_shared;

  for (int64_t n = tid; n < batch; n += blockDim.x) {
    int64_t idx = n * features + feature;
    float normalized = (Codec::Load(input, idx) - mean) * feature_inv_std;
    Codec::Store(x_hat, idx, normalized);
    Codec::Store(output, idx,
                 normalized * Codec::Load(gamma, feature) + Codec::Load(beta, feature));
  }
}

template <typename Codec>
__global__ void BatchNorm1dForwardEvalKernel(const typename Codec::Storage *input,
                                             const typename Codec::Storage *gamma,
                                             const typename Codec::Storage *beta,
                                             typename Codec::Storage *output,
                                             typename Codec::Storage *x_hat, float *inv_std,
                                             const float *running_mean, const float *running_var,
                                             int64_t batch, int64_t features, float eps) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * features;
  if (idx < total) {
    int64_t feature = idx % features;
    float feature_inv_std = rsqrtf(running_var[feature] + eps);
    if (idx < features) {
      inv_std[feature] = feature_inv_std;
    }
    float normalized = (Codec::Load(input, idx) - running_mean[feature]) * feature_inv_std;
    Codec::Store(x_hat, idx, normalized);
    Codec::Store(output, idx,
                 normalized * Codec::Load(gamma, feature) + Codec::Load(beta, feature));
  }
}

template <typename Codec>
__global__ void BatchNorm1dBackwardTrainingKernel(
    const typename Codec::Storage *grad_output, const typename Codec::Storage *x_hat,
    const typename Codec::Storage *gamma, typename Codec::Storage *grad_input, float *grad_gamma,
    float *grad_beta, const float *inv_std, int64_t batch, int64_t features) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;
  __shared__ float mean_dy_gamma_shared;
  __shared__ float mean_dy_gamma_xhat_shared;

  int64_t feature = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (feature >= features) {
    return;
  }

  float local_dy_sum = 0.0f;
  float local_dgamma_sum = 0.0f;
  float local_dy_gamma_sum = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    int64_t idx = n * features + feature;
    float dy = Codec::Load(grad_output, idx);
    float normalized = Codec::Load(x_hat, idx);
    local_dy_sum += dy;
    local_dgamma_sum += dy * normalized;
    local_dy_gamma_sum += dy * Codec::Load(gamma, feature);
  }
  float dy_sum = CudaBlockReduce(reduce_storage).Sum(local_dy_sum);
  __syncthreads();
  float dgamma_sum = CudaBlockReduce(reduce_storage).Sum(local_dgamma_sum);
  __syncthreads();
  float dy_gamma_sum = CudaBlockReduce(reduce_storage).Sum(local_dy_gamma_sum);
  if (tid == 0) {
    grad_beta[feature] = dy_sum;
    grad_gamma[feature] = dgamma_sum;
    mean_dy_gamma_shared = dy_gamma_sum / static_cast<float>(batch);
  }
  __syncthreads();

  float local_dy_gamma_xhat_sum = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    int64_t idx = n * features + feature;
    float dy_gamma = Codec::Load(grad_output, idx) * Codec::Load(gamma, feature);
    local_dy_gamma_xhat_sum += dy_gamma * Codec::Load(x_hat, idx);
  }
  float dy_gamma_xhat_sum = CudaBlockReduce(reduce_storage).Sum(local_dy_gamma_xhat_sum);
  if (tid == 0) {
    mean_dy_gamma_xhat_shared = dy_gamma_xhat_sum / static_cast<float>(batch);
  }
  __syncthreads();

  float feature_inv_std = inv_std[feature];
  float mean_dy_gamma = mean_dy_gamma_shared;
  float mean_dy_gamma_xhat = mean_dy_gamma_xhat_shared;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    int64_t idx = n * features + feature;
    float normalized = Codec::Load(x_hat, idx);
    float dy_gamma = Codec::Load(grad_output, idx) * Codec::Load(gamma, feature);
    float dx = (dy_gamma - mean_dy_gamma - normalized * mean_dy_gamma_xhat) * feature_inv_std;
    Codec::Store(grad_input, idx, dx);
  }
}

template <typename Codec>
__global__ void BatchNorm1dBackwardEvalKernel(
    const typename Codec::Storage *grad_output, const typename Codec::Storage *x_hat,
    const typename Codec::Storage *gamma, typename Codec::Storage *grad_input, float *grad_gamma,
    float *grad_beta, const float *inv_std, int64_t batch, int64_t features) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;

  int64_t feature = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (feature >= features) {
    return;
  }

  float local_dy_sum = 0.0f;
  float local_dgamma_sum = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    int64_t idx = n * features + feature;
    float dy = Codec::Load(grad_output, idx);
    local_dy_sum += dy;
    local_dgamma_sum += dy * Codec::Load(x_hat, idx);
    Codec::Store(grad_input, idx, dy * Codec::Load(gamma, feature) * inv_std[feature]);
  }
  float dy_sum = CudaBlockReduce(reduce_storage).Sum(local_dy_sum);
  __syncthreads();
  float dgamma_sum = CudaBlockReduce(reduce_storage).Sum(local_dgamma_sum);
  if (tid == 0) {
    grad_beta[feature] = dy_sum;
    grad_gamma[feature] = dgamma_sum;
  }
}
template <typename Codec>
Status LaunchLayerNormForwardKernel(RuntimeContext &ctx, const Tensor &input, const Tensor &gamma,
                                    const Tensor &beta, Tensor *output, Tensor *x_hat,
                                    Tensor *inv_std, int rows, int64_t width, float eps) {
  LayerNormForwardKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), gamma.data_as<typename Codec::Storage>(),
      beta.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      x_hat->data_as<typename Codec::Storage>(), inv_std->data_as<float>(), rows, width, eps);
  return detail::CheckKernelLaunch("LayerNorm forward kernel");
}

Status LaunchLayerNormForwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                    const Tensor &gamma, const Tensor &beta, Tensor *output,
                                    Tensor *x_hat, Tensor *inv_std, int rows, int64_t width,
                                    float eps) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchLayerNormForwardKernel<detail::Float32Codec>(ctx, input, gamma, beta, output,
                                                              x_hat, inv_std, rows, width, eps);
  case DType::kFloat16:
    return LaunchLayerNormForwardKernel<detail::Float16Codec>(ctx, input, gamma, beta, output,
                                                              x_hat, inv_std, rows, width, eps);
  case DType::kBFloat16:
    return LaunchLayerNormForwardKernel<detail::BFloat16Codec>(ctx, input, gamma, beta, output,
                                                               x_hat, inv_std, rows, width, eps);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("LayerNorm does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchLayerNormBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                     const Tensor &x_hat, const Tensor &gamma, Tensor *grad_input,
                                     Tensor *grad_gamma, Tensor *grad_beta, const Tensor &inv_std,
                                     int rows, int64_t width) {
  LayerNormBackwardKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), x_hat.data_as<typename Codec::Storage>(),
      gamma.data_as<typename Codec::Storage>(), grad_input->data_as<typename Codec::Storage>(),
      grad_gamma->data_as<float>(), grad_beta->data_as<float>(), inv_std.data_as<float>(), rows,
      width);
  return detail::CheckKernelLaunch("LayerNorm backward kernel");
}

Status LaunchLayerNormBackwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                     const Tensor &x_hat, const Tensor &gamma, Tensor *grad_input,
                                     Tensor *grad_gamma, Tensor *grad_beta, const Tensor &inv_std,
                                     int rows, int64_t width) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchLayerNormBackwardKernel<detail::Float32Codec>(
        ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta, inv_std, rows, width);
  case DType::kFloat16:
    return LaunchLayerNormBackwardKernel<detail::Float16Codec>(
        ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta, inv_std, rows, width);
  case DType::kBFloat16:
    return LaunchLayerNormBackwardKernel<detail::BFloat16Codec>(
        ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta, inv_std, rows, width);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("LayerNorm backward does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchBatchNorm1dForwardTrainingKernel(RuntimeContext &ctx, const Tensor &input,
                                              const Tensor &gamma, const Tensor &beta,
                                              Tensor *output, Tensor *x_hat, Tensor *inv_std,
                                              Tensor *running_mean, Tensor *running_var, int rows,
                                              int64_t batch, int64_t features, float eps,
                                              float momentum) {
  BatchNorm1dForwardTrainingKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), gamma.data_as<typename Codec::Storage>(),
      beta.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      x_hat->data_as<typename Codec::Storage>(), inv_std->data_as<float>(),
      running_mean->data_as<float>(), running_var->data_as<float>(), batch, features, eps,
      momentum);
  return detail::CheckKernelLaunch("BatchNorm1d forward-training kernel");
}

Status LaunchBatchNorm1dForwardTrainingKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                              const Tensor &gamma, const Tensor &beta,
                                              Tensor *output, Tensor *x_hat, Tensor *inv_std,
                                              Tensor *running_mean, Tensor *running_var, int rows,
                                              int64_t batch, int64_t features, float eps,
                                              float momentum) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchBatchNorm1dForwardTrainingKernel<detail::Float32Codec>(
        ctx, input, gamma, beta, output, x_hat, inv_std, running_mean, running_var, rows, batch,
        features, eps, momentum);
  case DType::kFloat16:
    return LaunchBatchNorm1dForwardTrainingKernel<detail::Float16Codec>(
        ctx, input, gamma, beta, output, x_hat, inv_std, running_mean, running_var, rows, batch,
        features, eps, momentum);
  case DType::kBFloat16:
    return LaunchBatchNorm1dForwardTrainingKernel<detail::BFloat16Codec>(
        ctx, input, gamma, beta, output, x_hat, inv_std, running_mean, running_var, rows, batch,
        features, eps, momentum);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("BatchNorm1d does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchBatchNorm1dForwardEvalKernel(RuntimeContext &ctx, const Tensor &input,
                                          const Tensor &gamma, const Tensor &beta, Tensor *output,
                                          Tensor *x_hat, Tensor *inv_std,
                                          const Tensor &running_mean, const Tensor &running_var,
                                          int blocks, int64_t batch, int64_t features, float eps) {
  BatchNorm1dForwardEvalKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), gamma.data_as<typename Codec::Storage>(),
      beta.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      x_hat->data_as<typename Codec::Storage>(), inv_std->data_as<float>(),
      running_mean.data_as<float>(), running_var.data_as<float>(), batch, features, eps);
  return detail::CheckKernelLaunch("BatchNorm1d forward-eval kernel");
}

Status LaunchBatchNorm1dForwardEvalKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                          const Tensor &gamma, const Tensor &beta, Tensor *output,
                                          Tensor *x_hat, Tensor *inv_std,
                                          const Tensor &running_mean, const Tensor &running_var,
                                          int blocks, int64_t batch, int64_t features, float eps) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchBatchNorm1dForwardEvalKernel<detail::Float32Codec>(
        ctx, input, gamma, beta, output, x_hat, inv_std, running_mean, running_var, blocks, batch,
        features, eps);
  case DType::kFloat16:
    return LaunchBatchNorm1dForwardEvalKernel<detail::Float16Codec>(
        ctx, input, gamma, beta, output, x_hat, inv_std, running_mean, running_var, blocks, batch,
        features, eps);
  case DType::kBFloat16:
    return LaunchBatchNorm1dForwardEvalKernel<detail::BFloat16Codec>(
        ctx, input, gamma, beta, output, x_hat, inv_std, running_mean, running_var, blocks, batch,
        features, eps);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("BatchNorm1d does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchBatchNorm1dBackwardTrainingKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                               const Tensor &x_hat, const Tensor &gamma,
                                               Tensor *grad_input, Tensor *grad_gamma,
                                               Tensor *grad_beta, const Tensor &inv_std, int rows,
                                               int64_t batch, int64_t features) {
  BatchNorm1dBackwardTrainingKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), x_hat.data_as<typename Codec::Storage>(),
      gamma.data_as<typename Codec::Storage>(), grad_input->data_as<typename Codec::Storage>(),
      grad_gamma->data_as<float>(), grad_beta->data_as<float>(), inv_std.data_as<float>(), batch,
      features);
  return detail::CheckKernelLaunch("BatchNorm1d backward-training kernel");
}

template <typename Codec>
Status LaunchBatchNorm1dBackwardEvalKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                           const Tensor &x_hat, const Tensor &gamma,
                                           Tensor *grad_input, Tensor *grad_gamma,
                                           Tensor *grad_beta, const Tensor &inv_std, int rows,
                                           int64_t batch, int64_t features) {
  BatchNorm1dBackwardEvalKernel<Codec><<<rows, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), x_hat.data_as<typename Codec::Storage>(),
      gamma.data_as<typename Codec::Storage>(), grad_input->data_as<typename Codec::Storage>(),
      grad_gamma->data_as<float>(), grad_beta->data_as<float>(), inv_std.data_as<float>(), batch,
      features);
  return detail::CheckKernelLaunch("BatchNorm1d backward-eval kernel");
}

Status LaunchBatchNorm1dBackwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                       const Tensor &x_hat, const Tensor &gamma, Tensor *grad_input,
                                       Tensor *grad_gamma, Tensor *grad_beta, const Tensor &inv_std,
                                       int rows, int64_t batch, int64_t features, bool training) {
  switch (dtype) {
  case DType::kFloat32:
    return training ? LaunchBatchNorm1dBackwardTrainingKernel<detail::Float32Codec>(
                          ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta,
                          inv_std, rows, batch, features)
                    : LaunchBatchNorm1dBackwardEvalKernel<detail::Float32Codec>(
                          ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta,
                          inv_std, rows, batch, features);
  case DType::kFloat16:
    return training ? LaunchBatchNorm1dBackwardTrainingKernel<detail::Float16Codec>(
                          ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta,
                          inv_std, rows, batch, features)
                    : LaunchBatchNorm1dBackwardEvalKernel<detail::Float16Codec>(
                          ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta,
                          inv_std, rows, batch, features);
  case DType::kBFloat16:
    return training ? LaunchBatchNorm1dBackwardTrainingKernel<detail::BFloat16Codec>(
                          ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta,
                          inv_std, rows, batch, features)
                    : LaunchBatchNorm1dBackwardEvalKernel<detail::BFloat16Codec>(
                          ctx, grad_output, x_hat, gamma, grad_input, grad_gamma, grad_beta,
                          inv_std, rows, batch, features);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("BatchNorm1d backward does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

} // namespace
} // namespace dlcuda
