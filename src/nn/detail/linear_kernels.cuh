#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

struct CublasLtMatmulDescOwner {
  ~CublasLtMatmulDescOwner() {
    if (desc != nullptr) {
      cublasLtMatmulDescDestroy(desc);
    }
  }

  cublasLtMatmulDesc_t desc = nullptr;
};

struct CublasLtMatrixLayoutOwner {
  ~CublasLtMatrixLayoutOwner() {
    if (layout != nullptr) {
      cublasLtMatrixLayoutDestroy(layout);
    }
  }

  cublasLtMatrixLayout_t layout = nullptr;
};

struct CublasLtPreferenceOwner {
  ~CublasLtPreferenceOwner() {
    if (preference != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
  }

  cublasLtMatmulPreference_t preference = nullptr;
};

bool IsCublasLtUnsupported(cublasStatus_t status) {
  return status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH;
}

Status CublasLtStatus(cublasStatus_t status, const std::string &context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  if (IsCublasLtUnsupported(status)) {
    return Status::Unsupported(context + " is not supported by cuBLASLt");
  }
  return detail::CublasStatus(status, context);
}

cublasComputeType_t LinearForwardComputeType(const RuntimeContext &ctx) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  if (ctx.tf32()) {
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  }
#else
  (void)ctx;
#endif
  return CUBLAS_COMPUTE_32F;
}

Status LinearForwardCublasLt(RuntimeContext &ctx, const Tensor &input, const Tensor &weight,
                             const Tensor &bias, Tensor *forward_output, int out_features,
                             int batch, int in_features) {
  Status lt_status = ctx.EnsureCublasLt();
  if (!lt_status.ok()) {
    return Status::Unsupported("cuBLASLt unavailable: " + lt_status.message());
  }

  CublasLtMatmulDescOwner op_desc;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&op_desc.desc, LinearForwardComputeType(ctx), CUDA_R_32F);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cublasLtMatmulDescCreate"));

  cublasOperation_t trans = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(op_desc.desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans,
                                          sizeof(trans));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt transA"));
  status = cublasLtMatmulDescSetAttribute(op_desc.desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans,
                                          sizeof(trans));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt transB"));

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  status = cublasLtMatmulDescSetAttribute(op_desc.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                          sizeof(epilogue));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt bias epilogue"));
  const void *bias_ptr = bias.data();
  status = cublasLtMatmulDescSetAttribute(op_desc.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &bias_ptr, sizeof(bias_ptr));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt bias pointer"));

  CublasLtMatrixLayoutOwner weight_desc;
  status = cublasLtMatrixLayoutCreate(&weight_desc.layout, CUDA_R_32F, out_features, in_features,
                                      out_features);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt weight layout"));

  CublasLtMatrixLayoutOwner input_desc;
  status =
      cublasLtMatrixLayoutCreate(&input_desc.layout, CUDA_R_32F, in_features, batch, in_features);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt input layout"));

  CublasLtMatrixLayoutOwner output_desc;
  status = cublasLtMatrixLayoutCreate(&output_desc.layout, CUDA_R_32F, out_features, batch,
                                      out_features);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt output layout"));

  CublasLtPreferenceOwner preference;
  status = cublasLtMatmulPreferenceCreate(&preference.preference);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cublasLtMatmulPreferenceCreate"));

  constexpr size_t kWorkspaceBytes = 0;
  status = cublasLtMatmulPreferenceSetAttribute(preference.preference,
                                                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &kWorkspaceBytes, sizeof(kWorkspaceBytes));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt workspace preference"));

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned_results = 0;
  status = cublasLtMatmulAlgoGetHeuristic(ctx.cublaslt_handle(), op_desc.desc, weight_desc.layout,
                                          input_desc.layout, output_desc.layout, output_desc.layout,
                                          preference.preference, 1, &heuristic, &returned_results);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cublasLtMatmulAlgoGetHeuristic"));
  if (returned_results == 0) {
    return Status::Unsupported("Linear forward cuBLASLt found no supported matmul algorithm");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasLtMatmul(ctx.cublaslt_handle(), op_desc.desc, &alpha, weight.data_as<float>(),
                          weight_desc.layout, input.data_as<float>(), input_desc.layout, &beta,
                          forward_output->data_as<float>(), output_desc.layout,
                          forward_output->data_as<float>(), output_desc.layout, &heuristic.algo,
                          nullptr, kWorkspaceBytes, ctx.stream());
  return CublasLtStatus(status, "Linear forward cublasLtMatmul");
}

__global__ void AddBiasKernel(float *output, const float *bias, int64_t batch,
                              int64_t out_features) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * out_features;
  if (idx < total) {
    output[idx] += bias[idx % out_features];
  }
}

template <typename Codec>
__global__ void
LinearForwardKernel(const typename Codec::Storage *input, const typename Codec::Storage *weight,
                    const typename Codec::Storage *bias, typename Codec::Storage *output,
                    int64_t batch, int64_t in_features, int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch && col < out_features) {
    float sum = 0.0f;
    for (int64_t i = 0; i < in_features; ++i) {
      sum +=
          Codec::Load(input, row * in_features + i) * Codec::Load(weight, i * out_features + col);
    }
    Codec::Store(output, row * out_features + col, sum + Codec::Load(bias, col));
  }
}

template <typename Codec>
__global__ void LinearBackwardInputKernel(const typename Codec::Storage *grad_output,
                                          const typename Codec::Storage *weight,
                                          typename Codec::Storage *grad_input, int64_t batch,
                                          int64_t in_features, int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch && col < in_features) {
    float sum = 0.0f;
    for (int64_t j = 0; j < out_features; ++j) {
      sum += Codec::Load(grad_output, row * out_features + j) *
             Codec::Load(weight, col * out_features + j);
    }
    Codec::Store(grad_input, row * in_features + col, sum);
  }
}

template <typename Codec>
__global__ void LinearBackwardWeightKernel(const typename Codec::Storage *input,
                                           const typename Codec::Storage *grad_output,
                                           float *grad_weight, int64_t batch, int64_t in_features,
                                           int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < in_features && col < out_features) {
    float sum = 0.0f;
    for (int64_t n = 0; n < batch; ++n) {
      sum += Codec::Load(input, n * in_features + row) *
             Codec::Load(grad_output, n * out_features + col);
    }
    grad_weight[row * out_features + col] = sum;
  }
}

template <typename Codec>
__global__ void LinearBackwardBiasKernel(const typename Codec::Storage *grad_output,
                                         float *grad_bias, int64_t batch, int64_t out_features) {
  __shared__ typename CudaBlockReduce::TempStorage reduce_storage;

  int64_t col = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (col >= out_features) {
    return;
  }

  float sum = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    sum += Codec::Load(grad_output, n * out_features + col);
  }
  float block_sum = CudaBlockReduce(reduce_storage).Sum(sum);

  if (tid == 0) {
    grad_bias[col] = block_sum;
  }
}
template <typename Codec>
Status LaunchLinearForwardKernel(RuntimeContext &ctx, const Tensor &input, const Tensor &weight,
                                 const Tensor &bias, Tensor *output, dim3 blocks, dim3 threads,
                                 int64_t batch, int64_t in_features, int64_t out_features) {
  LinearForwardKernel<Codec><<<blocks, threads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), weight.data_as<typename Codec::Storage>(),
      bias.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(), batch,
      in_features, out_features);
  return detail::CheckKernelLaunch("Linear forward kernel");
}

Status LaunchLinearForwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                 const Tensor &weight, const Tensor &bias, Tensor *output,
                                 dim3 blocks, dim3 threads, int64_t batch, int64_t in_features,
                                 int64_t out_features) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchLinearForwardKernel<detail::Float32Codec>(
        ctx, input, weight, bias, output, blocks, threads, batch, in_features, out_features);
  case DType::kFloat16:
    return LaunchLinearForwardKernel<detail::Float16Codec>(
        ctx, input, weight, bias, output, blocks, threads, batch, in_features, out_features);
  case DType::kBFloat16:
    return LaunchLinearForwardKernel<detail::BFloat16Codec>(
        ctx, input, weight, bias, output, blocks, threads, batch, in_features, out_features);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Linear forward does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchLinearBackwardInputKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                       const Tensor &weight, Tensor *grad_input, dim3 blocks,
                                       dim3 threads, int64_t batch, int64_t in_features,
                                       int64_t out_features) {
  LinearBackwardInputKernel<Codec><<<blocks, threads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), weight.data_as<typename Codec::Storage>(),
      grad_input->data_as<typename Codec::Storage>(), batch, in_features, out_features);
  return detail::CheckKernelLaunch("Linear backward-input kernel");
}

Status LaunchLinearBackwardInputKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                       const Tensor &weight, Tensor *grad_input, dim3 blocks,
                                       dim3 threads, int64_t batch, int64_t in_features,
                                       int64_t out_features) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchLinearBackwardInputKernel<detail::Float32Codec>(
        ctx, grad_output, weight, grad_input, blocks, threads, batch, in_features, out_features);
  case DType::kFloat16:
    return LaunchLinearBackwardInputKernel<detail::Float16Codec>(
        ctx, grad_output, weight, grad_input, blocks, threads, batch, in_features, out_features);
  case DType::kBFloat16:
    return LaunchLinearBackwardInputKernel<detail::BFloat16Codec>(
        ctx, grad_output, weight, grad_input, blocks, threads, batch, in_features, out_features);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Linear backward-input does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchLinearBackwardWeightKernel(RuntimeContext &ctx, const Tensor &input,
                                        const Tensor &grad_output, Tensor *grad_weight, dim3 blocks,
                                        dim3 threads, int64_t batch, int64_t in_features,
                                        int64_t out_features) {
  LinearBackwardWeightKernel<Codec><<<blocks, threads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), grad_output.data_as<typename Codec::Storage>(),
      grad_weight->data_as<float>(), batch, in_features, out_features);
  return detail::CheckKernelLaunch("Linear backward-weight kernel");
}

Status LaunchLinearBackwardWeightKernel(RuntimeContext &ctx, DType dtype, const Tensor &input,
                                        const Tensor &grad_output, Tensor *grad_weight, dim3 blocks,
                                        dim3 threads, int64_t batch, int64_t in_features,
                                        int64_t out_features) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchLinearBackwardWeightKernel<detail::Float32Codec>(
        ctx, input, grad_output, grad_weight, blocks, threads, batch, in_features, out_features);
  case DType::kFloat16:
    return LaunchLinearBackwardWeightKernel<detail::Float16Codec>(
        ctx, input, grad_output, grad_weight, blocks, threads, batch, in_features, out_features);
  case DType::kBFloat16:
    return LaunchLinearBackwardWeightKernel<detail::BFloat16Codec>(
        ctx, input, grad_output, grad_weight, blocks, threads, batch, in_features, out_features);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Linear backward-weight does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchLinearBackwardBiasKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                      Tensor *grad_bias, int rows, int64_t batch,
                                      int64_t out_features) {
  LinearBackwardBiasKernel<Codec>
      <<<rows, kCudaThreads, 0, ctx.stream()>>>(grad_output.data_as<typename Codec::Storage>(),
                                                grad_bias->data_as<float>(), batch, out_features);
  return detail::CheckKernelLaunch("Linear backward-bias kernel");
}

Status LaunchLinearBackwardBiasKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                      Tensor *grad_bias, int rows, int64_t batch,
                                      int64_t out_features) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchLinearBackwardBiasKernel<detail::Float32Codec>(ctx, grad_output, grad_bias, rows,
                                                                batch, out_features);
  case DType::kFloat16:
    return LaunchLinearBackwardBiasKernel<detail::Float16Codec>(ctx, grad_output, grad_bias, rows,
                                                                batch, out_features);
  case DType::kBFloat16:
    return LaunchLinearBackwardBiasKernel<detail::BFloat16Codec>(ctx, grad_output, grad_bias, rows,
                                                                 batch, out_features);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Linear backward-bias does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

} // namespace
} // namespace dlcuda
