#pragma once

#include "common.cuh"

namespace dlcuda {

#if defined(DLCUDA_HAS_CUBLASLT)
struct LinearCublasLtForwardPlan {
  ~LinearCublasLtForwardPlan() {
    ResetDescriptors();
  }

  LinearCublasLtForwardPlan() = default;
  LinearCublasLtForwardPlan(const LinearCublasLtForwardPlan &) = delete;
  LinearCublasLtForwardPlan &operator=(const LinearCublasLtForwardPlan &) = delete;

  void ResetDescriptors() {
    if (op_desc != nullptr) {
      cublasLtMatmulDescDestroy(op_desc);
      op_desc = nullptr;
    }
    if (weight_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(weight_layout);
      weight_layout = nullptr;
    }
    if (input_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(input_layout);
      input_layout = nullptr;
    }
    if (output_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(output_layout);
      output_layout = nullptr;
    }
    if (preference != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference);
      preference = nullptr;
    }
    initialized = false;
    out_features = 0;
    batch = 0;
    in_features = 0;
    dtype = DType::kFloat32;
    tf32 = false;
    bias_data = nullptr;
    workspace_bytes = 0;
  }

  [[nodiscard]] bool Matches(int out_features_in, int batch_in, int in_features_in, DType dtype_in,
                             bool tf32_in, const void *bias_data_in) const {
    return initialized && out_features == out_features_in && batch == batch_in &&
           in_features == in_features_in && dtype == dtype_in && tf32 == tf32_in &&
           bias_data == bias_data_in;
  }

  bool initialized = false;
  int out_features = 0;
  int batch = 0;
  int in_features = 0;
  DType dtype = DType::kFloat32;
  bool tf32 = false;
  const void *bias_data = nullptr;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t input_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algo = {};
  Tensor workspace;
  size_t workspace_bytes = 0;
};
#else
struct LinearCublasLtForwardPlan {};
#endif

namespace {

#if defined(DLCUDA_HAS_CUBLASLT)

constexpr size_t kLinearCublasLtWorkspaceLimitBytes = 4 * 1024 * 1024;

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

#endif

#if defined(DLCUDA_HAS_CUBLASLT)

Result<int64_t> LinearCublasLtWorkspaceElements(size_t bytes) {
  size_t elements = (bytes + sizeof(float) - 1) / sizeof(float);
  if (elements > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return Status::InvalidArgument("Linear forward cuBLASLt workspace is too large");
  }
  return static_cast<int64_t>(elements);
}

Status BuildLinearForwardCublasLtPlan(RuntimeContext &ctx, const Tensor &bias,
                                      LinearCublasLtForwardPlan *plan, int out_features, int batch,
                                      int in_features, DType dtype) {
  if (plan == nullptr) {
    return Status::InvalidArgument("Linear forward cuBLASLt plan is null");
  }

  plan->ResetDescriptors();

  auto data_type = detail::CublasCudaDataType(dtype, "Linear cuBLAS");
  if (!data_type.ok()) {
    return data_type.status();
  }

  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->op_desc, detail::CublasComputeType(ctx.tf32(), dtype), CUDA_R_32F);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cublasLtMatmulDescCreate"));

  cublasOperation_t trans = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans,
                                          sizeof(trans));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt transA"));
  status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans,
                                          sizeof(trans));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt transB"));

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                          sizeof(epilogue));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt bias epilogue"));
  const void *bias_ptr = bias.data();
  status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &bias_ptr, sizeof(bias_ptr));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt bias pointer"));

  status = cublasLtMatrixLayoutCreate(&plan->weight_layout, data_type.value(), out_features,
                                      in_features, out_features);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt weight layout"));

  status = cublasLtMatrixLayoutCreate(&plan->input_layout, data_type.value(), in_features, batch,
                                      in_features);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt input layout"));

  status = cublasLtMatrixLayoutCreate(&plan->output_layout, data_type.value(), out_features, batch,
                                      out_features);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt output layout"));

  status = cublasLtMatmulPreferenceCreate(&plan->preference);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cublasLtMatmulPreferenceCreate"));
  status = cublasLtMatmulPreferenceSetAttribute(
      plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kLinearCublasLtWorkspaceLimitBytes, sizeof(kLinearCublasLtWorkspaceLimitBytes));
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cuBLASLt workspace preference"));

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned_results = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      ctx.cublaslt_handle(), plan->op_desc, plan->weight_layout, plan->input_layout,
      plan->output_layout, plan->output_layout, plan->preference, 1, &heuristic, &returned_results);
  DLCUDA_RETURN_IF_ERROR(CublasLtStatus(status, "Linear forward cublasLtMatmulAlgoGetHeuristic"));
  if (returned_results == 0) {
    return Status::Unsupported("Linear forward cuBLASLt found no supported matmul algorithm");
  }

  plan->workspace_bytes = heuristic.workspaceSize;
  if (plan->workspace_bytes > 0) {
    auto workspace_elements = LinearCublasLtWorkspaceElements(plan->workspace_bytes);
    if (!workspace_elements.ok()) {
      return workspace_elements.status();
    }
    DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&plan->workspace, {workspace_elements.value()},
                                             DType::kFloat32, ctx.stream()));
  }

  plan->algo = heuristic.algo;
  plan->out_features = out_features;
  plan->batch = batch;
  plan->in_features = in_features;
  plan->dtype = dtype;
  plan->tf32 = ctx.tf32();
  plan->bias_data = bias.data();
  plan->initialized = true;
  return Status::Ok();
}

Status LinearForwardCublasLt(RuntimeContext &ctx, const Tensor &input, const Tensor &weight,
                             const Tensor &bias, Tensor *forward_output, int out_features,
                             int batch, int in_features, DType dtype,
                             LinearCublasLtForwardPlan *plan) {
  Status lt_status = ctx.EnsureCublasLt();
  if (!lt_status.ok()) {
    return Status::Unsupported("cuBLASLt unavailable: " + lt_status.message());
  }
  if (plan == nullptr) {
    return Status::InvalidArgument("Linear forward cuBLASLt plan is null");
  }

  if (!plan->Matches(out_features, batch, in_features, dtype, ctx.tf32(), bias.data())) {
    DLCUDA_RETURN_IF_ERROR(
        BuildLinearForwardCublasLtPlan(ctx, bias, plan, out_features, batch, in_features, dtype));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  void *workspace = plan->workspace_bytes == 0 ? nullptr : plan->workspace.data();
  cublasStatus_t status = cublasLtMatmul(
      ctx.cublaslt_handle(), plan->op_desc, &alpha, weight.data(), plan->weight_layout,
      input.data(), plan->input_layout, &beta, forward_output->data(), plan->output_layout,
      forward_output->data(), plan->output_layout, &plan->algo, workspace, plan->workspace_bytes,
      ctx.stream());
  return CublasLtStatus(status, "Linear forward cublasLtMatmul");
}

#endif

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000

Status CublasGemmExStatus(cublasStatus_t status, const std::string &context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) {
    return Status::Unsupported(context + " is not supported by cuBLAS");
  }
  return detail::CublasStatus(status, context);
}

Status LinearCublasGemmEx(RuntimeContext &ctx, cublasOperation_t trans_a, cublasOperation_t trans_b,
                          int m, int n, int k, const Tensor &a, DType a_dtype, int lda,
                          const Tensor &b, DType b_dtype, int ldb, Tensor *c, DType c_dtype,
                          int ldc, const char *op_name) {
  if (c == nullptr) {
    return Status::InvalidArgument(std::string(op_name) + " output is null");
  }
  auto a_type = detail::CublasCudaDataType(a_dtype, "Linear cuBLAS");
  if (!a_type.ok()) {
    return a_type.status();
  }
  auto b_type = detail::CublasCudaDataType(b_dtype, "Linear cuBLAS");
  if (!b_type.ok()) {
    return b_type.status();
  }
  auto c_type = detail::CublasCudaDataType(c_dtype, "Linear cuBLAS");
  if (!c_type.ok()) {
    return c_type.status();
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status =
      cublasGemmEx(ctx.cublas_handle(), trans_a, trans_b, m, n, k, &alpha, a.data(), a_type.value(),
                   lda, b.data(), b_type.value(), ldb, &beta, c->data(), c_type.value(), ldc,
                   detail::CublasComputeType(ctx.tf32(), a_dtype), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return CublasGemmExStatus(status, op_name);
}

#endif

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
