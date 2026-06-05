#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

struct Conv2dGemmDims {
  int batch = 0;
  int spatial = 0;
  int in_features = 0;
  int out_channels = 0;
  int batch_spatial = 0;
  int64_t spatial64 = 0;
  int64_t in_features64 = 0;
  int64_t batch_spatial64 = 0;
};

Result<int64_t> Conv2dCheckedMul(int64_t lhs, int64_t rhs, const char *name) {
  if (lhs < 0 || rhs < 0) {
    return Status::Unsupported(std::string(name) + " must be non-negative for Conv2d GEMM");
  }
  if (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return Status::Unsupported(std::string(name) + " is too large for Conv2d GEMM");
  }
  return lhs * rhs;
}

Result<int> Conv2dCublasInt(int64_t value, const char *name) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << name << " is outside cuBLAS int range: " << value;
    return Status::Unsupported(oss.str());
  }
  return static_cast<int>(value);
}

Result<Conv2dGemmDims> Conv2dBuildGemmDims(int64_t batch, int64_t in_channels, int64_t out_channels,
                                           int64_t kernel_h, int64_t kernel_w, int64_t output_h,
                                           int64_t output_w) {
  if (batch <= 0) {
    return Status::Unsupported("Conv2d GEMM requires a positive batch size");
  }

  auto spatial = Conv2dCheckedMul(output_h, output_w, "Conv2d output spatial size");
  if (!spatial.ok()) {
    return spatial.status();
  }
  auto kernel_area = Conv2dCheckedMul(kernel_h, kernel_w, "Conv2d kernel area");
  if (!kernel_area.ok()) {
    return kernel_area.status();
  }
  auto in_features = Conv2dCheckedMul(in_channels, kernel_area.value(), "Conv2d im2col rows");
  if (!in_features.ok()) {
    return in_features.status();
  }
  auto batch_spatial = Conv2dCheckedMul(batch, spatial.value(), "Conv2d batch-spatial columns");
  if (!batch_spatial.ok()) {
    return batch_spatial.status();
  }

  auto batch_int = Conv2dCublasInt(batch, "Conv2d batch");
  if (!batch_int.ok()) {
    return batch_int.status();
  }
  auto spatial_int = Conv2dCublasInt(spatial.value(), "Conv2d output spatial size");
  if (!spatial_int.ok()) {
    return spatial_int.status();
  }
  auto in_features_int = Conv2dCublasInt(in_features.value(), "Conv2d im2col rows");
  if (!in_features_int.ok()) {
    return in_features_int.status();
  }
  auto out_channels_int = Conv2dCublasInt(out_channels, "Conv2d out_channels");
  if (!out_channels_int.ok()) {
    return out_channels_int.status();
  }
  auto batch_spatial_int = Conv2dCublasInt(batch_spatial.value(), "Conv2d batch-spatial columns");
  if (!batch_spatial_int.ok()) {
    return batch_spatial_int.status();
  }

  Conv2dGemmDims dims;
  dims.batch = batch_int.value();
  dims.spatial = spatial_int.value();
  dims.in_features = in_features_int.value();
  dims.out_channels = out_channels_int.value();
  dims.batch_spatial = batch_spatial_int.value();
  dims.spatial64 = spatial.value();
  dims.in_features64 = in_features.value();
  dims.batch_spatial64 = batch_spatial.value();
  return dims;
}

Status Conv2dCublasStatus(cublasStatus_t status, const std::string &context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) {
    return Status::Unsupported(context + " is not supported by cuBLAS");
  }
  return detail::CublasStatus(status, context);
}

Status Conv2dCublasGemm(RuntimeContext &ctx, cublasOperation_t trans_a, cublasOperation_t trans_b,
                        int m, int n, int k, const Tensor &a, DType a_dtype, int lda,
                        const Tensor &b, DType b_dtype, int ldb, Tensor *c, DType c_dtype, int ldc,
                        const char *op_name) {
  if (c == nullptr) {
    return Status::InvalidArgument(std::string(op_name) + " output is null");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (a_dtype == DType::kFloat32 && b_dtype == DType::kFloat32 && c_dtype == DType::kFloat32) {
    cublasStatus_t status =
        cublasSgemm(ctx.cublas_handle(), trans_a, trans_b, m, n, k, &alpha, a.data_as<float>(), lda,
                    b.data_as<float>(), ldb, &beta, c->data_as<float>(), ldc);
    return Conv2dCublasStatus(status, op_name);
  }

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  auto a_type = detail::CublasCudaDataType(a_dtype, "Conv2d cuBLAS");
  if (!a_type.ok()) {
    return a_type.status();
  }
  auto b_type = detail::CublasCudaDataType(b_dtype, "Conv2d cuBLAS");
  if (!b_type.ok()) {
    return b_type.status();
  }
  auto c_type = detail::CublasCudaDataType(c_dtype, "Conv2d cuBLAS");
  if (!c_type.ok()) {
    return c_type.status();
  }
  cublasStatus_t status =
      cublasGemmEx(ctx.cublas_handle(), trans_a, trans_b, m, n, k, &alpha, a.data(), a_type.value(),
                   lda, b.data(), b_type.value(), ldb, &beta, c->data(), c_type.value(), ldc,
                   detail::CublasComputeType(ctx.tf32(), a_dtype), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return Conv2dCublasStatus(status, op_name);
#else
  (void)ctx;
  (void)trans_a;
  (void)trans_b;
  (void)m;
  (void)n;
  (void)k;
  (void)a;
  (void)a_dtype;
  (void)lda;
  (void)b;
  (void)b_dtype;
  (void)ldb;
  (void)c;
  (void)c_dtype;
  (void)ldc;
  return Status::Unsupported(std::string(op_name) + " requires CUDA 11 cuBLAS for this dtype");
#endif
}

Status Conv2dCublasStridedBatchedGemm(RuntimeContext &ctx, cublasOperation_t trans_a,
                                      cublasOperation_t trans_b, int m, int n, int k,
                                      const Tensor &a, DType a_dtype, int lda, int64_t stride_a,
                                      const Tensor &b, DType b_dtype, int ldb, int64_t stride_b,
                                      Tensor *c, DType c_dtype, int ldc, int64_t stride_c,
                                      int batch_count, const char *op_name) {
  if (c == nullptr) {
    return Status::InvalidArgument(std::string(op_name) + " output is null");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (a_dtype == DType::kFloat32 && b_dtype == DType::kFloat32 && c_dtype == DType::kFloat32) {
    cublasStatus_t status = cublasSgemmStridedBatched(
        ctx.cublas_handle(), trans_a, trans_b, m, n, k, &alpha, a.data_as<float>(), lda,
        static_cast<long long int>(stride_a), b.data_as<float>(), ldb,
        static_cast<long long int>(stride_b), &beta, c->data_as<float>(), ldc,
        static_cast<long long int>(stride_c), batch_count);
    return Conv2dCublasStatus(status, op_name);
  }

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  auto a_type = detail::CublasCudaDataType(a_dtype, "Conv2d cuBLAS");
  if (!a_type.ok()) {
    return a_type.status();
  }
  auto b_type = detail::CublasCudaDataType(b_dtype, "Conv2d cuBLAS");
  if (!b_type.ok()) {
    return b_type.status();
  }
  auto c_type = detail::CublasCudaDataType(c_dtype, "Conv2d cuBLAS");
  if (!c_type.ok()) {
    return c_type.status();
  }
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      ctx.cublas_handle(), trans_a, trans_b, m, n, k, &alpha, a.data(), a_type.value(), lda,
      static_cast<long long int>(stride_a), b.data(), b_type.value(), ldb,
      static_cast<long long int>(stride_b), &beta, c->data(), c_type.value(), ldc,
      static_cast<long long int>(stride_c), batch_count,
      detail::CublasComputeType(ctx.tf32(), a_dtype), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return Conv2dCublasStatus(status, op_name);
#else
  (void)ctx;
  (void)trans_a;
  (void)trans_b;
  (void)m;
  (void)n;
  (void)k;
  (void)a;
  (void)a_dtype;
  (void)lda;
  (void)stride_a;
  (void)b;
  (void)b_dtype;
  (void)ldb;
  (void)stride_b;
  (void)c;
  (void)c_dtype;
  (void)ldc;
  (void)stride_c;
  (void)batch_count;
  return Status::Unsupported(std::string(op_name) + " requires CUDA 11 cuBLAS for this dtype");
#endif
}

template <typename Codec>
__global__ void Conv2dIm2ColKernel(const typename Codec::Storage *input,
                                   typename Codec::Storage *columns, int64_t total, int64_t batch,
                                   int64_t in_channels, int64_t input_h, int64_t input_w,
                                   int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                   int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                   int64_t output_h, int64_t output_w) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t in_features = in_channels * kernel_h * kernel_w;
  int64_t k = idx % in_features;
  int64_t col = idx / in_features;
  int64_t spatial = output_h * output_w;
  int64_t s = col % spatial;
  int64_t n = col / spatial;
  if (n >= batch) {
    return;
  }

  int64_t kw = k % kernel_w;
  int64_t tmp = k / kernel_w;
  int64_t kh = tmp % kernel_h;
  int64_t ic = tmp / kernel_h;
  int64_t oh = s / output_w;
  int64_t ow = s % output_w;
  int64_t ih = oh * stride_h + kh - padding_h;
  int64_t iw = ow * stride_w + kw - padding_w;

  float value = 0.0f;
  if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
    int64_t input_index = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
    value = Codec::Load(input, input_index);
  }
  Codec::Store(columns, idx, value);
}

template <typename Codec>
__global__ void Conv2dCol2ImKernel(const typename Codec::Storage *columns,
                                   typename Codec::Storage *grad_input, int64_t total,
                                   int64_t batch, int64_t in_channels, int64_t input_h,
                                   int64_t input_w, int64_t kernel_h, int64_t kernel_w,
                                   int64_t stride_h, int64_t stride_w, int64_t padding_h,
                                   int64_t padding_w, int64_t output_h, int64_t output_w) {
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

  int64_t in_features = in_channels * kernel_h * kernel_w;
  int64_t spatial = output_h * output_w;
  float sum = 0.0f;
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
      int64_t k = (ic * kernel_h + kh) * kernel_w + kw;
      int64_t col = n * spatial + oh * output_w + ow;
      sum += Codec::Load(columns, col * in_features + k);
    }
  }

  Codec::Store(grad_input, idx, sum);
}

template <typename Codec>
__global__ void Conv2dAddBiasNchwKernel(typename Codec::Storage *output,
                                        const typename Codec::Storage *bias, int64_t total,
                                        int64_t out_channels, int64_t output_h, int64_t output_w) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t spatial = output_h * output_w;
  int64_t oc = (idx / spatial) % out_channels;
  float value = Codec::Load(output, idx) + Codec::Load(bias, oc);
  Codec::Store(output, idx, value);
}

template <typename Codec>
__global__ void Conv2dPackGradOutputKernel(const typename Codec::Storage *grad_output,
                                           typename Codec::Storage *grad_output_matrix,
                                           int64_t total, int64_t batch, int64_t out_channels,
                                           int64_t output_h, int64_t output_w) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t spatial = output_h * output_w;
  int64_t batch_spatial = batch * spatial;
  int64_t row = idx % batch_spatial;
  int64_t oc = idx / batch_spatial;
  int64_t n = row / spatial;
  int64_t s = row % spatial;
  int64_t grad_index = n * out_channels * spatial + oc * spatial + s;
  Codec::Store(grad_output_matrix, idx, Codec::Load(grad_output, grad_index));
}

template <typename Codec>
Status LaunchConv2dIm2ColKernel(RuntimeContext &ctx, const Tensor &input, Tensor *columns,
                                const Conv2dGemmDims &dims, int64_t batch, int64_t in_channels,
                                int64_t input_h, int64_t input_w, int64_t kernel_h,
                                int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                int64_t padding_h, int64_t padding_w, int64_t output_h,
                                int64_t output_w) {
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(columns, {dims.batch_spatial64, dims.in_features64},
                                           input.dtype(), ctx.stream()));
  auto total = Conv2dCheckedMul(dims.batch_spatial64, dims.in_features64, "Conv2d im2col size");
  if (!total.ok()) {
    return total.status();
  }
  auto blocks = detail::BlocksForElements(total.value(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() == 0) {
    return Status::Ok();
  }
  Conv2dIm2ColKernel<Codec><<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), columns->data_as<typename Codec::Storage>(),
      total.value(), batch, in_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w, output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d im2col kernel");
}

Status LaunchConv2dIm2ColKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                Tensor *columns, const Conv2dGemmDims &dims, int64_t batch,
                                int64_t in_channels, int64_t input_h, int64_t input_w,
                                int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                int64_t output_h, int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dIm2ColKernel<detail::Float32Codec>(
        ctx, input, columns, dims, batch, in_channels, input_h, input_w, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kFloat16:
    return LaunchConv2dIm2ColKernel<detail::Float16Codec>(
        ctx, input, columns, dims, batch, in_channels, input_h, input_w, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kBFloat16:
    return LaunchConv2dIm2ColKernel<detail::BFloat16Codec>(
        ctx, input, columns, dims, batch, in_channels, input_h, input_w, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d im2col does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchConv2dCol2ImKernel(RuntimeContext &ctx, const Tensor &columns, Tensor *grad_input,
                                int blocks, int64_t total, int64_t batch, int64_t in_channels,
                                int64_t input_h, int64_t input_w, int64_t kernel_h,
                                int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                int64_t padding_h, int64_t padding_w, int64_t output_h,
                                int64_t output_w) {
  Conv2dCol2ImKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      columns.data_as<typename Codec::Storage>(), grad_input->data_as<typename Codec::Storage>(),
      total, batch, in_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w,
      padding_h, padding_w, output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d col2im kernel");
}

Status LaunchConv2dCol2ImKernel(RuntimeContext &ctx, DType dtype, const Tensor &columns,
                                Tensor *grad_input, int blocks, int64_t total, int64_t batch,
                                int64_t in_channels, int64_t input_h, int64_t input_w,
                                int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                int64_t output_h, int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dCol2ImKernel<detail::Float32Codec>(
        ctx, columns, grad_input, blocks, total, batch, in_channels, input_h, input_w, kernel_h,
        kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kFloat16:
    return LaunchConv2dCol2ImKernel<detail::Float16Codec>(
        ctx, columns, grad_input, blocks, total, batch, in_channels, input_h, input_w, kernel_h,
        kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kBFloat16:
    return LaunchConv2dCol2ImKernel<detail::BFloat16Codec>(
        ctx, columns, grad_input, blocks, total, batch, in_channels, input_h, input_w, kernel_h,
        kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d col2im does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchConv2dAddBiasNchwKernel(RuntimeContext &ctx, const Tensor &bias, Tensor *output,
                                     int blocks, int64_t total, int64_t out_channels,
                                     int64_t output_h, int64_t output_w) {
  Conv2dAddBiasNchwKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      output->data_as<typename Codec::Storage>(), bias.data_as<typename Codec::Storage>(), total,
      out_channels, output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d add-bias kernel");
}

Status LaunchConv2dAddBiasNchwKernel(RuntimeContext &ctx, DType dtype, const Tensor &bias,
                                     Tensor *output, int64_t total, int64_t out_channels,
                                     int64_t output_h, int64_t output_w) {
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() == 0) {
    return Status::Ok();
  }
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dAddBiasNchwKernel<detail::Float32Codec>(
        ctx, bias, output, blocks.value(), total, out_channels, output_h, output_w);
  case DType::kFloat16:
    return LaunchConv2dAddBiasNchwKernel<detail::Float16Codec>(
        ctx, bias, output, blocks.value(), total, out_channels, output_h, output_w);
  case DType::kBFloat16:
    return LaunchConv2dAddBiasNchwKernel<detail::BFloat16Codec>(
        ctx, bias, output, blocks.value(), total, out_channels, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d add-bias does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchConv2dPackGradOutputKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                        Tensor *grad_output_matrix, const Conv2dGemmDims &dims,
                                        int64_t batch, int64_t out_channels, int64_t output_h,
                                        int64_t output_w) {
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(grad_output_matrix, {out_channels, dims.batch_spatial64},
                                           grad_output.dtype(), ctx.stream()));
  auto total =
      Conv2dCheckedMul(dims.batch_spatial64, out_channels, "Conv2d packed grad_output size");
  if (!total.ok()) {
    return total.status();
  }
  auto blocks = detail::BlocksForElements(total.value(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() == 0) {
    return Status::Ok();
  }
  Conv2dPackGradOutputKernel<Codec><<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(),
      grad_output_matrix->data_as<typename Codec::Storage>(), total.value(), batch, out_channels,
      output_h, output_w);
  return detail::CheckKernelLaunch("Conv2d pack grad_output kernel");
}

Status LaunchConv2dPackGradOutputKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                        Tensor *grad_output_matrix, const Conv2dGemmDims &dims,
                                        int64_t batch, int64_t out_channels, int64_t output_h,
                                        int64_t output_w) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchConv2dPackGradOutputKernel<detail::Float32Codec>(
        ctx, grad_output, grad_output_matrix, dims, batch, out_channels, output_h, output_w);
  case DType::kFloat16:
    return LaunchConv2dPackGradOutputKernel<detail::Float16Codec>(
        ctx, grad_output, grad_output_matrix, dims, batch, out_channels, output_h, output_w);
  case DType::kBFloat16:
    return LaunchConv2dPackGradOutputKernel<detail::BFloat16Codec>(
        ctx, grad_output, grad_output_matrix, dims, batch, out_channels, output_h, output_w);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Conv2d pack grad_output does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

Status LaunchConv2dForwardGemm(RuntimeContext &ctx, DType dtype, const Tensor &input,
                               const Tensor &weight, const Tensor &bias, Tensor *output,
                               Tensor *columns, int64_t batch, int64_t in_channels, int64_t input_h,
                               int64_t input_w, int64_t out_channels, int64_t kernel_h,
                               int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                               int64_t padding_h, int64_t padding_w, int64_t output_h,
                               int64_t output_w) {
  auto dims =
      Conv2dBuildGemmDims(batch, in_channels, out_channels, kernel_h, kernel_w, output_h, output_w);
  if (!dims.ok()) {
    return dims.status();
  }
  DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
  DLCUDA_RETURN_IF_ERROR(LaunchConv2dIm2ColKernel(
      ctx, dtype, input, columns, dims.value(), batch, in_channels, input_h, input_w, kernel_h,
      kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w));

  DLCUDA_RETURN_IF_ERROR(Conv2dCublasStridedBatchedGemm(
      ctx, CUBLAS_OP_T, CUBLAS_OP_N, dims.value().spatial, dims.value().out_channels,
      dims.value().in_features, *columns, dtype, dims.value().in_features,
      dims.value().in_features64 * dims.value().spatial64, weight, dtype, dims.value().in_features,
      0, output, dtype, dims.value().spatial, dims.value().spatial64 * out_channels,
      dims.value().batch, "Conv2d forward im2col cuBLAS"));

  return LaunchConv2dAddBiasNchwKernel(ctx, dtype, bias, output, output->numel(), out_channels,
                                       output_h, output_w);
}

Status LaunchConv2dBackwardInputGemm(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                     const Tensor &weight, Tensor *grad_input, Tensor *grad_columns,
                                     int64_t batch, int64_t in_channels, int64_t input_h,
                                     int64_t input_w, int64_t out_channels, int64_t kernel_h,
                                     int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                     int64_t padding_h, int64_t padding_w, int64_t output_h,
                                     int64_t output_w) {
  auto dims =
      Conv2dBuildGemmDims(batch, in_channels, out_channels, kernel_h, kernel_w, output_h, output_w);
  if (!dims.ok()) {
    return dims.status();
  }
  DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(grad_columns, {dims.value().batch_spatial64, dims.value().in_features64},
                        dtype, ctx.stream()));

  DLCUDA_RETURN_IF_ERROR(Conv2dCublasStridedBatchedGemm(
      ctx, CUBLAS_OP_N, CUBLAS_OP_T, dims.value().in_features, dims.value().spatial,
      dims.value().out_channels, weight, dtype, dims.value().in_features, 0, grad_output, dtype,
      dims.value().spatial, dims.value().spatial64 * out_channels, grad_columns, dtype,
      dims.value().in_features, dims.value().in_features64 * dims.value().spatial64,
      dims.value().batch, "Conv2d backward-input cuBLAS"));

  auto blocks = detail::BlocksForElements(grad_input->numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() == 0) {
    return Status::Ok();
  }
  return LaunchConv2dCol2ImKernel(ctx, dtype, *grad_columns, grad_input, blocks.value(),
                                  grad_input->numel(), batch, in_channels, input_h, input_w,
                                  kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
                                  output_h, output_w);
}

Status LaunchConv2dBackwardWeightGemm(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                      const Tensor &grad_output, Tensor *grad_weight,
                                      Tensor *columns, Tensor *grad_output_matrix, int64_t batch,
                                      int64_t in_channels, int64_t input_h, int64_t input_w,
                                      int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w, int64_t padding_h,
                                      int64_t padding_w, int64_t output_h, int64_t output_w) {
  auto dims =
      Conv2dBuildGemmDims(batch, in_channels, out_channels, kernel_h, kernel_w, output_h, output_w);
  if (!dims.ok()) {
    return dims.status();
  }
  DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
  DLCUDA_RETURN_IF_ERROR(LaunchConv2dIm2ColKernel(
      ctx, dtype, input, columns, dims.value(), batch, in_channels, input_h, input_w, kernel_h,
      kernel_w, stride_h, stride_w, padding_h, padding_w, output_h, output_w));
  DLCUDA_RETURN_IF_ERROR(LaunchConv2dPackGradOutputKernel(ctx, dtype, grad_output,
                                                          grad_output_matrix, dims.value(), batch,
                                                          out_channels, output_h, output_w));

  return Conv2dCublasGemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, dims.value().in_features,
                          dims.value().out_channels, dims.value().batch_spatial, *columns, dtype,
                          dims.value().in_features, *grad_output_matrix, dtype,
                          dims.value().batch_spatial, grad_weight, DType::kFloat32,
                          dims.value().in_features, "Conv2d backward-weight cuBLAS");
}

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
