#include "dl_cuda/nn.hpp"

#include "dl_cuda/detail/cuda_dtype.cuh"
#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/detail/tensor_validation.hpp"
#include "dl_cuda/tensor_ops.hpp"

#if defined(DLCUDA_HAS_CUBLASLT)
#include <cublasLt.h>
#endif
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dlcuda {
namespace {

constexpr int kCudaThreads = 256;
constexpr int kLinearTile = 16;

using CudaBlockReduce = cub::BlockReduce<float, kCudaThreads>;
using detail::EnsureDType;
using detail::EnsureSameShapeAndType;
using detail::ValidateFloatingTensor;
using detail::ValidateIntTensor;
using detail::ValidateRank;

#if defined(DLCUDA_HAS_CUBLASLT)

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

#endif

Status ValidateFloatingDType(DType dtype, const char *name) {
  if (!IsFloatingPointDType(dtype)) {
    return Status::InvalidArgument(std::string(name) + " dtype must be floating point");
  }
  return Status::Ok();
}

Status CopyHostFloatsToTensor(Tensor *tensor, const std::vector<float> &values,
                              cudaStream_t stream) {
  if (tensor == nullptr || !tensor->defined()) {
    return Status::InvalidArgument("CopyHostFloatsToTensor received undefined tensor");
  }
  if (tensor->numel() != static_cast<int64_t>(values.size())) {
    return Status::InvalidArgument("CopyHostFloatsToTensor size mismatch");
  }
  switch (tensor->dtype()) {
  case DType::kFloat32:
    return tensor->CopyFromHost(values.data(), values.size() * sizeof(float), stream);
  case DType::kFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = detail::FloatToFloat16Bits(values[i]);
    }
    return tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), stream);
  }
  case DType::kBFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = detail::FloatToBFloat16Bits(values[i]);
    }
    return tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), stream);
  }
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("CopyHostFloatsToTensor requires a floating-point tensor");
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
__global__ void EmbeddingForwardKernel(const typename Codec::Storage *table,
                                       const int32_t *token_ids, typename Codec::Storage *output,
                                       int64_t num_tokens, int64_t embedding_dim,
                                       int64_t vocab_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_tokens * embedding_dim;
  if (idx < total) {
    int64_t token = idx / embedding_dim;
    int64_t dim = idx % embedding_dim;
    int32_t token_id = token_ids[token];
    if (token_id >= 0 && static_cast<int64_t>(token_id) < vocab_size) {
      Codec::Store(output, idx,
                   Codec::Load(table, static_cast<int64_t>(token_id) * embedding_dim + dim));
    } else {
      Codec::Store(output, idx, 0.0f);
    }
  }
}

template <typename Codec>
__global__ void EmbeddingBackwardKernel(const typename Codec::Storage *grad_output,
                                        const int32_t *token_ids, float *grad_table,
                                        int64_t num_tokens, int64_t embedding_dim,
                                        int64_t vocab_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_tokens * embedding_dim;
  if (idx < total) {
    int64_t token = idx / embedding_dim;
    int64_t dim = idx % embedding_dim;
    int32_t token_id = token_ids[token];
    if (token_id >= 0 && static_cast<int64_t>(token_id) < vocab_size) {
      int64_t table_index = static_cast<int64_t>(token_id) * embedding_dim + dim;
      atomicAdd(&grad_table[table_index], Codec::Load(grad_output, idx));
    }
  }
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

__device__ uint32_t DropoutHash(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<uint32_t>(value >> 32);
}

template <typename Codec>
__global__ void DropoutForwardKernel(const typename Codec::Storage *input,
                                     typename Codec::Storage *output, float *mask, int64_t size,
                                     float probability, uint64_t seed) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float keep_probability = 1.0f - probability;
    float uniform = static_cast<float>(DropoutHash(seed ^ static_cast<uint64_t>(idx)) >> 8) *
                    (1.0f / 16777216.0f);
    float multiplier = uniform < keep_probability ? 1.0f / keep_probability : 0.0f;
    mask[idx] = multiplier;
    Codec::Store(output, idx, Codec::Load(input, idx) * multiplier);
  }
}

template <typename Codec>
__global__ void DropoutBackwardKernel(const typename Codec::Storage *grad_output, const float *mask,
                                      typename Codec::Storage *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    Codec::Store(grad_input, idx, Codec::Load(grad_output, idx) * mask[idx]);
  }
}

template <typename Codec>
__global__ void TensorCopyKernel(const typename Codec::Storage *input,
                                 typename Codec::Storage *output, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    Codec::Store(output, idx, Codec::Load(input, idx));
  }
}

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
Status LaunchEmbeddingForwardKernel(RuntimeContext &ctx, const Tensor &table,
                                    const Tensor &token_ids, Tensor *output, int blocks,
                                    int64_t num_tokens, int64_t embedding_dim, int64_t vocab_size) {
  EmbeddingForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      table.data_as<typename Codec::Storage>(), token_ids.data_as<int32_t>(),
      output->data_as<typename Codec::Storage>(), num_tokens, embedding_dim, vocab_size);
  return detail::CheckKernelLaunch("Embedding forward kernel");
}

Status LaunchEmbeddingForwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &table,
                                    const Tensor &token_ids, Tensor *output, int blocks,
                                    int64_t num_tokens, int64_t embedding_dim, int64_t vocab_size) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchEmbeddingForwardKernel<detail::Float32Codec>(
        ctx, table, token_ids, output, blocks, num_tokens, embedding_dim, vocab_size);
  case DType::kFloat16:
    return LaunchEmbeddingForwardKernel<detail::Float16Codec>(
        ctx, table, token_ids, output, blocks, num_tokens, embedding_dim, vocab_size);
  case DType::kBFloat16:
    return LaunchEmbeddingForwardKernel<detail::BFloat16Codec>(
        ctx, table, token_ids, output, blocks, num_tokens, embedding_dim, vocab_size);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Embedding does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

template <typename Codec>
Status LaunchEmbeddingBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                     const Tensor &token_ids, Tensor *grad_table, int blocks,
                                     int64_t num_tokens, int64_t embedding_dim,
                                     int64_t vocab_size) {
  EmbeddingBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), token_ids.data_as<int32_t>(),
      grad_table->data_as<float>(), num_tokens, embedding_dim, vocab_size);
  return detail::CheckKernelLaunch("Embedding backward kernel");
}

Status LaunchEmbeddingBackwardKernel(RuntimeContext &ctx, DType dtype, const Tensor &grad_output,
                                     const Tensor &token_ids, Tensor *grad_table, int blocks,
                                     int64_t num_tokens, int64_t embedding_dim,
                                     int64_t vocab_size) {
  switch (dtype) {
  case DType::kFloat32:
    return LaunchEmbeddingBackwardKernel<detail::Float32Codec>(
        ctx, grad_output, token_ids, grad_table, blocks, num_tokens, embedding_dim, vocab_size);
  case DType::kFloat16:
    return LaunchEmbeddingBackwardKernel<detail::Float16Codec>(
        ctx, grad_output, token_ids, grad_table, blocks, num_tokens, embedding_dim, vocab_size);
  case DType::kBFloat16:
    return LaunchEmbeddingBackwardKernel<detail::BFloat16Codec>(
        ctx, grad_output, token_ids, grad_table, blocks, num_tokens, embedding_dim, vocab_size);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Embedding backward does not support dtype " +
                                 std::string(DTypeName(dtype)));
}

Result<int64_t> SpatialOutputSize(int64_t input, int64_t kernel, int64_t stride, int64_t padding,
                                  const char *name) {
  if (input < 0) {
    return Status::InvalidArgument(std::string(name) + " input size must be non-negative");
  }
  if (kernel <= 0 || stride <= 0 || padding < 0) {
    return Status::InvalidArgument(std::string(name) +
                                   " kernel/stride/padding parameters are invalid");
  }
  int64_t numerator = input + 2 * padding - kernel;
  if (numerator < 0) {
    return Status::InvalidArgument(std::string(name) + " output size is non-positive");
  }
  return numerator / stride + 1;
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

template <typename Codec>
Status LaunchTensorCopyKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                              int blocks) {
  TensorCopyKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      input.numel());
  return detail::CheckKernelLaunch("Tensor copy kernel");
}

Status LaunchTensorCopyKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                              int blocks) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchTensorCopyKernel<detail::Float32Codec>(ctx, input, output, blocks);
  case DType::kFloat16:
    return LaunchTensorCopyKernel<detail::Float16Codec>(ctx, input, output, blocks);
  case DType::kBFloat16:
    return LaunchTensorCopyKernel<detail::BFloat16Codec>(ctx, input, output, blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Tensor copy does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchDropoutForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  Tensor *mask, int blocks, float probability, uint64_t seed) {
  DropoutForwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<typename Codec::Storage>(),
      mask->data_as<float>(), input.numel(), probability, seed);
  return detail::CheckKernelLaunch("Dropout forward kernel");
}

Status LaunchDropoutForwardKernel(RuntimeContext &ctx, const Tensor &input, Tensor *output,
                                  Tensor *mask, int blocks, float probability, uint64_t seed) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchDropoutForwardKernel<detail::Float32Codec>(ctx, input, output, mask, blocks,
                                                            probability, seed);
  case DType::kFloat16:
    return LaunchDropoutForwardKernel<detail::Float16Codec>(ctx, input, output, mask, blocks,
                                                            probability, seed);
  case DType::kBFloat16:
    return LaunchDropoutForwardKernel<detail::BFloat16Codec>(ctx, input, output, mask, blocks,
                                                             probability, seed);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Dropout does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

template <typename Codec>
Status LaunchDropoutBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &mask, Tensor *grad_input, int blocks) {
  DropoutBackwardKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), mask.data_as<float>(),
      grad_input->data_as<typename Codec::Storage>(), grad_output.numel());
  return detail::CheckKernelLaunch("Dropout backward kernel");
}

Status LaunchDropoutBackwardKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                   const Tensor &mask, Tensor *grad_input, int blocks) {
  switch (grad_output.dtype()) {
  case DType::kFloat32:
    return LaunchDropoutBackwardKernel<detail::Float32Codec>(ctx, grad_output, mask, grad_input,
                                                             blocks);
  case DType::kFloat16:
    return LaunchDropoutBackwardKernel<detail::Float16Codec>(ctx, grad_output, mask, grad_input,
                                                             blocks);
  case DType::kBFloat16:
    return LaunchDropoutBackwardKernel<detail::BFloat16Codec>(ctx, grad_output, mask, grad_input,
                                                              blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Dropout backward does not support dtype " +
                                 std::string(DTypeName(grad_output.dtype())));
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

std::string JoinParameterName(const std::string &prefix, const char *name) {
  if (prefix.empty()) {
    return std::string(name);
  }
  return prefix + "." + name;
}

} // namespace

Status Sequential::Add(std::unique_ptr<Module> module) {
  if (!module) {
    return Status::InvalidArgument("Sequential::Add received null module");
  }
  modules_.push_back(std::move(module));
  RebuildParameterCache();
  return Status::Ok();
}

Status Sequential::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Forward output pointer is null");
  }

  Tensor current = input;
  for (size_t i = 0; i < modules_.size(); ++i) {
    Tensor next;
    Status status = modules_[i]->Forward(ctx, current, &next);
    if (!status.ok()) {
      return Status::RuntimeError("Forward failed in module " + std::to_string(i) + ": " +
                                  status.message());
    }
    if (!next.defined()) {
      return Status::RuntimeError("Forward output became undefined in module " + std::to_string(i));
    }
    current = next;
  }

  *output = current;
  return Status::Ok();
}

Status Sequential::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }

  Tensor current = grad_output;
  for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
    Tensor next;
    Tensor *next_out = (i == 0 && grad_input == nullptr) ? nullptr : &next;
    Status status = modules_[static_cast<size_t>(i)]->Backward(ctx, current, next_out);
    if (!status.ok()) {
      return Status::RuntimeError("Backward failed in module " + std::to_string(i) + ": " +
                                  status.message());
    }
    if (i > 0 && !next.defined()) {
      return Status::RuntimeError("Backward gradient became undefined before first module");
    }
    current = next;
  }

  if (grad_input != nullptr) {
    *grad_input = current;
  }
  return Status::Ok();
}

void Sequential::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  for (size_t i = 0; i < modules_.size(); ++i) {
    std::string child_name = "layers." + std::to_string(i);
    std::string child_prefix = prefix.empty() ? child_name : prefix + "." + child_name;
    modules_[i]->AppendParameters(child_prefix, out);
  }
}

void Sequential::RebuildParameterCache() {
  parameter_cache_.clear();
  AppendParameters("", &parameter_cache_);
}

Residual::Residual(std::unique_ptr<Module> branch) : branch_(std::move(branch)) {}

Status Residual::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!branch_) {
    return Status::InvalidArgument("Residual branch is null");
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Residual::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(branch_->Forward(ctx, input, &branch_output_));
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(input, branch_output_, "Residual input", "Residual branch output"));
  DLCUDA_RETURN_IF_ERROR(TensorAdd(ctx, input, branch_output_, &forward_output_));
  *output = forward_output_;
  return Status::Ok();
}

Status Residual::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!branch_) {
    return Status::InvalidArgument("Residual branch is null");
  }
  if (grad_input == nullptr) {
    return branch_->Backward(ctx, grad_output, nullptr);
  }
  DLCUDA_RETURN_IF_ERROR(branch_->Backward(ctx, grad_output, &branch_grad_));
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, branch_grad_, "Residual grad_output", "branch grad"));
  DLCUDA_RETURN_IF_ERROR(TensorAdd(ctx, grad_output, branch_grad_, &backward_output_));
  *grad_input = backward_output_;
  return Status::Ok();
}

void Residual::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (branch_ == nullptr || out == nullptr) {
    return;
  }
  branch_->AppendParameters(JoinParameterName(prefix, "branch"), out);
}

Linear::Linear(int64_t in_features, int64_t out_features, RuntimeContext &ctx, DType dtype)
    : in_features_(in_features), out_features_(out_features), dtype_(dtype) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    init_status_ = Status::InvalidArgument("Linear dimensions must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "Linear");
  if (!init_status_.ok()) {
    return;
  }

  auto weight = Tensor::AllocateAsync({in_features_, out_features_}, dtype_, ctx.stream());
  if (!weight.ok()) {
    init_status_ = weight.status();
    return;
  }
  auto bias = Tensor::AllocateAsync({out_features_}, dtype_, ctx.stream());
  if (!bias.ok()) {
    init_status_ = bias.status();
    return;
  }
  auto grad_weight =
      Tensor::AllocateAsync({in_features_, out_features_}, DType::kFloat32, ctx.stream());
  if (!grad_weight.ok()) {
    init_status_ = grad_weight.status();
    return;
  }
  auto grad_bias = Tensor::AllocateAsync({out_features_}, DType::kFloat32, ctx.stream());
  if (!grad_bias.ok()) {
    init_status_ = grad_bias.status();
    return;
  }

  weight_ = weight.value();
  bias_ = bias.value();
  grad_weight_ = grad_weight.value();
  grad_bias_ = grad_bias.value();

  std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features_));

  std::vector<float> host_weight(static_cast<size_t>(in_features_ * out_features_));
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

Status Linear::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Linear::Forward output is null");
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Linear input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "Linear input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Linear input"));

  int64_t batch = input.dim(0);
  int64_t in_features = input.dim(1);
  if (in_features != in_features_) {
    std::ostringstream oss;
    oss << "Linear input feature mismatch: expected " << in_features_ << " got " << in_features;
    return Status::InvalidArgument(oss.str());
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, {batch, out_features_}, dtype_, ctx.stream()));
  cached_input_ = input;
  last_batch_ = batch;
  if (batch == 0) {
    *output = forward_output_;
    return Status::Ok();
  }

  if (ctx.use_cublas() && dtype_ == DType::kFloat32) {
    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();

    auto out_features_int = detail::CheckedInt(out_features_, "out_features");
    if (!out_features_int.ok()) {
      return out_features_int.status();
    }
    auto batch_int = detail::CheckedInt(batch, "batch");
    if (!batch_int.ok()) {
      return batch_int.status();
    }
    auto in_features_int = detail::CheckedInt(in_features_, "in_features");
    if (!in_features_int.ok()) {
      return in_features_int.status();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    bool used_cublaslt = false;
#if defined(DLCUDA_HAS_CUBLASLT)
    Status lt_status =
        LinearForwardCublasLt(ctx, input, weight_, bias_, &forward_output_,
                              out_features_int.value(), batch_int.value(), in_features_int.value());
    if (lt_status.ok()) {
      used_cublaslt = true;
    } else if (lt_status.code() != StatusCode::kUnsupported) {
      return lt_status;
    }
#endif
    if (!used_cublaslt) {
      DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features_int.value(), batch_int.value(),
                      in_features_int.value(), &alpha, weight_.data_as<float>(),
                      out_features_int.value(), input.data_as<float>(), in_features_int.value(),
                      &beta, forward_output_.data_as<float>(), out_features_int.value()),
          "Linear forward cublasSgemm"));

      int64_t total = batch * out_features_;
      auto blocks = detail::BlocksForElements(total, kCudaThreads);
      if (!blocks.ok()) {
        return blocks.status();
      }
      AddBiasKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
          forward_output_.data_as<float>(), bias_.data_as<float>(), batch, out_features_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear add-bias kernel"));
    }
  } else {
    auto x_blocks = detail::BlocksForElements(out_features_, kLinearTile);
    if (!x_blocks.ok()) {
      return x_blocks.status();
    }
    auto y_blocks = detail::BlocksForElements(batch, kLinearTile);
    if (!y_blocks.ok()) {
      return y_blocks.status();
    }
    dim3 threads(kLinearTile, kLinearTile);
    dim3 blocks(static_cast<unsigned int>(x_blocks.value()),
                static_cast<unsigned int>(y_blocks.value()));
    DLCUDA_RETURN_IF_ERROR(LaunchLinearForwardKernel(ctx, dtype_, input, weight_, bias_,
                                                     &forward_output_, blocks, threads, batch,
                                                     in_features_, out_features_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Linear::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  bool need_grad_input = grad_input != nullptr;

  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Linear grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "Linear grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Linear grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != out_features_) {
    return Status::InvalidArgument("Linear grad_output shape mismatch");
  }
  if (!cached_input_.defined()) {
    return Status::RuntimeError("Linear backward called before forward");
  }

  if (need_grad_input) {
    DLCUDA_RETURN_IF_ERROR(
        EnsureTensorAsync(&backward_output_, {last_batch_, in_features_}, dtype_, ctx.stream()));
  }
  if (last_batch_ == 0) {
    DLCUDA_RETURN_IF_ERROR(grad_weight_.FillZero(ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(grad_bias_.FillZero(ctx.stream()));
    if (need_grad_input) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }

  if (ctx.use_cublas() && dtype_ == DType::kFloat32) {
    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();

    auto in_features_int = detail::CheckedInt(in_features_, "in_features");
    if (!in_features_int.ok()) {
      return in_features_int.status();
    }
    auto batch_int = detail::CheckedInt(last_batch_, "batch");
    if (!batch_int.ok()) {
      return batch_int.status();
    }
    auto out_features_int = detail::CheckedInt(out_features_, "out_features");
    if (!out_features_int.ok()) {
      return out_features_int.status();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (need_grad_input) {
      DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
          cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features_int.value(), batch_int.value(),
                      out_features_int.value(), &alpha, weight_.data_as<float>(),
                      out_features_int.value(), grad_output.data_as<float>(),
                      out_features_int.value(), &beta, backward_output_.data_as<float>(),
                      in_features_int.value()),
          "Linear backward-input cublasSgemm"));
    }

    DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features_int.value(),
                    in_features_int.value(), batch_int.value(), &alpha,
                    grad_output.data_as<float>(), out_features_int.value(),
                    cached_input_.data_as<float>(), in_features_int.value(), &beta,
                    grad_weight_.data_as<float>(), out_features_int.value()),
        "Linear backward-weight cublasSgemm"));

    auto rows = detail::RowsForGrid(out_features_, "linear bias");
    if (!rows.ok()) {
      return rows.status();
    }
    DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardBiasKernel(
        ctx, dtype_, grad_output, &grad_bias_, rows.value(), last_batch_, out_features_));
  } else {
    dim3 threads(kLinearTile, kLinearTile);
    if (need_grad_input) {
      auto input_x_blocks = detail::BlocksForElements(in_features_, kLinearTile);
      if (!input_x_blocks.ok()) {
        return input_x_blocks.status();
      }
      auto input_y_blocks = detail::BlocksForElements(last_batch_, kLinearTile);
      if (!input_y_blocks.ok()) {
        return input_y_blocks.status();
      }
      dim3 blocks_input(static_cast<unsigned int>(input_x_blocks.value()),
                        static_cast<unsigned int>(input_y_blocks.value()));
      DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardInputKernel(
          ctx, dtype_, grad_output, weight_, &backward_output_, blocks_input, threads, last_batch_,
          in_features_, out_features_));
    }

    auto weight_x_blocks = detail::BlocksForElements(out_features_, kLinearTile);
    if (!weight_x_blocks.ok()) {
      return weight_x_blocks.status();
    }
    auto weight_y_blocks = detail::BlocksForElements(in_features_, kLinearTile);
    if (!weight_y_blocks.ok()) {
      return weight_y_blocks.status();
    }
    dim3 blocks_weight(static_cast<unsigned int>(weight_x_blocks.value()),
                       static_cast<unsigned int>(weight_y_blocks.value()));
    DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardWeightKernel(
        ctx, dtype_, cached_input_, grad_output, &grad_weight_, blocks_weight, threads, last_batch_,
        in_features_, out_features_));

    auto bias_rows = detail::RowsForGrid(out_features_, "linear bias");
    if (!bias_rows.ok()) {
      return bias_rows.status();
    }
    DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardBiasKernel(
        ctx, dtype_, grad_output, &grad_bias_, bias_rows.value(), last_batch_, out_features_));
  }

  if (need_grad_input) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void Linear::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "weight"), &weight_, &grad_weight_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "bias"), &bias_, &grad_bias_});
}

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

Status ReLU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("ReLU::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "ReLU input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, input.shape(), input.dtype(), ctx.stream()));
  cached_input_ = input;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchReLUForwardKernel(ctx, input, &forward_output_, blocks.value()));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status ReLU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "ReLU grad_output"));
  if (!cached_input_.defined()) {
    return Status::RuntimeError("ReLU backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_input_, "grad_output", "cached_input"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchReLUBackwardKernel(ctx, grad_output, cached_input_,
                                                    &backward_output_, blocks.value()));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void ReLU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status GELU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("GELU::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "GELU input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, input.shape(), input.dtype(), ctx.stream()));
  cached_input_ = input;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchGELUForwardKernel(ctx, input, &forward_output_, blocks.value()));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status GELU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "GELU grad_output"));
  if (!cached_input_.defined()) {
    return Status::RuntimeError("GELU backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_input_, "grad_output", "cached_input"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchGELUBackwardKernel(ctx, grad_output, cached_input_,
                                                    &backward_output_, blocks.value()));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void GELU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Sigmoid::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Sigmoid::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Sigmoid input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&cached_output_, input.shape(), input.dtype(), ctx.stream()));

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchSigmoidForwardKernel(ctx, input, &cached_output_, blocks.value()));
  }

  *output = cached_output_;
  return Status::Ok();
}

Status Sigmoid::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Sigmoid grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Sigmoid backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_output_, "grad_output", "cached_output"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchSigmoidBackwardKernel(ctx, grad_output, cached_output_,
                                                       &backward_output_, blocks.value()));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Sigmoid::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Softmax::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Softmax::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Softmax input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Softmax input"));

  num_rows_ = input.dim(0);
  row_width_ = input.dim(1);
  if (num_rows_ > 0 && row_width_ == 0) {
    return Status::InvalidArgument("Softmax row width must be positive");
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&cached_output_, input.shape(), input.dtype(), ctx.stream()));

  auto rows = detail::RowsForGrid(num_rows_, "softmax");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(
        LaunchSoftmaxForwardKernel(ctx, input, &cached_output_, rows.value(), row_width_));
  }

  *output = cached_output_;
  return Status::Ok();
}

Status Softmax::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Softmax grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Softmax backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_output_, "grad_output", "cached_output"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  if (num_rows_ > 0 && row_width_ == 0) {
    return Status::InvalidArgument("Softmax row width must be positive");
  }
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  auto rows = detail::RowsForGrid(num_rows_, "softmax");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchSoftmaxBackwardKernel(
        ctx, grad_output, cached_output_, &backward_output_, rows.value(), num_rows_, row_width_));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Softmax::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Dropout::Dropout(float probability, uint64_t seed)
    : probability_(probability), seed_(seed == 0ULL ? 0x1234abcd5678ef90ULL : seed) {
  if (probability_ < 0.0f || probability_ >= 1.0f) {
    init_status_ = Status::InvalidArgument("Dropout probability must be in [0, 1)");
  }
}

void Dropout::SetTraining(bool training) {
  training_ = training;
}

Status Dropout::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Dropout::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Dropout input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, input.shape(), input.dtype(), ctx.stream()));
  last_training_ = training_;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    if (training_) {
      DLCUDA_RETURN_IF_ERROR(
          EnsureTensorAsync(&mask_, input.shape(), DType::kFloat32, ctx.stream()));
      uint64_t call_seed = seed_ + 0x9e3779b97f4a7c15ULL * (++call_index_);
      DLCUDA_RETURN_IF_ERROR(LaunchDropoutForwardKernel(ctx, input, &forward_output_, &mask_,
                                                        blocks.value(), probability_, call_seed));
    } else {
      DLCUDA_RETURN_IF_ERROR(LaunchTensorCopyKernel(ctx, input, &forward_output_, blocks.value()));
    }
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Dropout::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Dropout grad_output"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), grad_output.dtype(), ctx.stream()));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    if (last_training_) {
      if (!mask_.defined() || mask_.shape() != grad_output.shape()) {
        return Status::RuntimeError("Dropout backward called before matching training forward");
      }
      DLCUDA_RETURN_IF_ERROR(
          LaunchDropoutBackwardKernel(ctx, grad_output, mask_, &backward_output_, blocks.value()));
    } else {
      DLCUDA_RETURN_IF_ERROR(
          LaunchTensorCopyKernel(ctx, grad_output, &backward_output_, blocks.value()));
    }
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Dropout::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

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

LayerNorm::LayerNorm(int64_t normalized_size, RuntimeContext &ctx, float eps, DType dtype)
    : normalized_size_(normalized_size), eps_(eps), dtype_(dtype) {
  if (normalized_size_ <= 0 || eps_ <= 0.0f) {
    init_status_ = Status::InvalidArgument("LayerNorm normalized_size and eps must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "LayerNorm");
  if (!init_status_.ok()) {
    return;
  }

  auto gamma = Tensor::AllocateAsync({normalized_size_}, dtype_, ctx.stream());
  auto beta = Tensor::AllocateAsync({normalized_size_}, dtype_, ctx.stream());
  auto grad_gamma = Tensor::AllocateAsync({normalized_size_}, DType::kFloat32, ctx.stream());
  auto grad_beta = Tensor::AllocateAsync({normalized_size_}, DType::kFloat32, ctx.stream());
  if (!gamma.ok() || !beta.ok() || !grad_gamma.ok() || !grad_beta.ok()) {
    init_status_ = !gamma.ok()        ? gamma.status()
                   : !beta.ok()       ? beta.status()
                   : !grad_gamma.ok() ? grad_gamma.status()
                                      : grad_beta.status();
    return;
  }
  gamma_ = gamma.value();
  beta_ = beta.value();
  grad_gamma_ = grad_gamma.value();
  grad_beta_ = grad_beta.value();

  std::vector<float> host_gamma(static_cast<size_t>(normalized_size_), 1.0f);
  init_status_ = CopyHostFloatsToTensor(&gamma_, host_gamma, ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = beta_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_gamma_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_beta_.FillZero(ctx.stream());
}

Status LayerNorm::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("LayerNorm::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "LayerNorm input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "LayerNorm input"));
  if (input.rank() < 1 || input.dim(static_cast<int>(input.rank() - 1)) != normalized_size_) {
    return Status::InvalidArgument("LayerNorm input last dimension mismatch");
  }

  last_rows_ = input.numel() / normalized_size_;
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&cached_x_hat_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&inv_std_, {last_rows_}, DType::kFloat32, ctx.stream()));

  auto rows = detail::RowsForGrid(last_rows_, "LayerNorm");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchLayerNormForwardKernel(ctx, dtype_, input, gamma_, beta_,
                                                        &forward_output_, &cached_x_hat_, &inv_std_,
                                                        rows.value(), normalized_size_, eps_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status LayerNorm::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "LayerNorm grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "LayerNorm grad_output"));
  if (!cached_x_hat_.defined()) {
    return Status::RuntimeError("LayerNorm backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_x_hat_, "grad_output", "cached_x_hat"));
  DLCUDA_RETURN_IF_ERROR(grad_gamma_.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(grad_beta_.FillZero(ctx.stream()));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), dtype_, ctx.stream()));
  auto rows = detail::RowsForGrid(last_rows_, "LayerNorm");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchLayerNormBackwardKernel(
        ctx, dtype_, grad_output, cached_x_hat_, gamma_, &backward_output_, &grad_gamma_,
        &grad_beta_, inv_std_, rows.value(), normalized_size_));
  }

  if (grad_input != nullptr) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void LayerNorm::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "gamma"), &gamma_, &grad_gamma_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "beta"), &beta_, &grad_beta_});
}

BatchNorm1d::BatchNorm1d(int64_t features, RuntimeContext &ctx, float eps, float momentum,
                         DType dtype)
    : features_(features), eps_(eps), momentum_(momentum), dtype_(dtype) {
  if (features_ <= 0 || eps_ <= 0.0f || momentum_ < 0.0f || momentum_ > 1.0f) {
    init_status_ = Status::InvalidArgument("BatchNorm1d features/eps/momentum are invalid");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "BatchNorm1d");
  if (!init_status_.ok()) {
    return;
  }

  auto gamma = Tensor::AllocateAsync({features_}, dtype_, ctx.stream());
  auto beta = Tensor::AllocateAsync({features_}, dtype_, ctx.stream());
  auto grad_gamma = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  auto grad_beta = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  auto running_mean = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  auto running_var = Tensor::AllocateAsync({features_}, DType::kFloat32, ctx.stream());
  if (!gamma.ok() || !beta.ok() || !grad_gamma.ok() || !grad_beta.ok() || !running_mean.ok() ||
      !running_var.ok()) {
    init_status_ = !gamma.ok()          ? gamma.status()
                   : !beta.ok()         ? beta.status()
                   : !grad_gamma.ok()   ? grad_gamma.status()
                   : !grad_beta.ok()    ? grad_beta.status()
                   : !running_mean.ok() ? running_mean.status()
                                        : running_var.status();
    return;
  }
  gamma_ = gamma.value();
  beta_ = beta.value();
  grad_gamma_ = grad_gamma.value();
  grad_beta_ = grad_beta.value();
  running_mean_ = running_mean.value();
  running_var_ = running_var.value();

  std::vector<float> host_ones(static_cast<size_t>(features_), 1.0f);
  init_status_ = CopyHostFloatsToTensor(&gamma_, host_ones, ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ =
      running_var_.CopyFromHost(host_ones.data(), host_ones.size() * sizeof(float), ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = beta_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = running_mean_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_gamma_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_beta_.FillZero(ctx.stream());
}

void BatchNorm1d::SetTraining(bool training) {
  training_ = training;
}

Status BatchNorm1d::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("BatchNorm1d::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "BatchNorm1d input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "BatchNorm1d input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "BatchNorm1d input"));
  if (input.dim(1) != features_) {
    return Status::InvalidArgument("BatchNorm1d feature dimension mismatch");
  }
  if (training_ && input.dim(0) <= 0) {
    return Status::InvalidArgument("BatchNorm1d training batch size must be positive");
  }

  last_batch_ = input.dim(0);
  last_training_ = training_;
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&cached_x_hat_, input.shape(), dtype_, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&inv_std_, {features_}, DType::kFloat32, ctx.stream()));

  if (last_batch_ == 0) {
    *output = forward_output_;
    return Status::Ok();
  }

  if (training_) {
    auto rows = detail::RowsForGrid(features_, "BatchNorm1d");
    if (!rows.ok()) {
      return rows.status();
    }
    DLCUDA_RETURN_IF_ERROR(LaunchBatchNorm1dForwardTrainingKernel(
        ctx, dtype_, input, gamma_, beta_, &forward_output_, &cached_x_hat_, &inv_std_,
        &running_mean_, &running_var_, rows.value(), last_batch_, features_, eps_, momentum_));
  } else {
    auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchBatchNorm1dForwardEvalKernel(
          ctx, dtype_, input, gamma_, beta_, &forward_output_, &cached_x_hat_, &inv_std_,
          running_mean_, running_var_, blocks.value(), last_batch_, features_, eps_));
    }
  }

  *output = forward_output_;
  return Status::Ok();
}

Status BatchNorm1d::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "BatchNorm1d grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "BatchNorm1d grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "BatchNorm1d grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != features_) {
    return Status::InvalidArgument("BatchNorm1d grad_output shape mismatch");
  }
  if (!cached_x_hat_.defined()) {
    return Status::RuntimeError("BatchNorm1d backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(grad_gamma_.FillZero(ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(grad_beta_.FillZero(ctx.stream()));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&backward_output_, grad_output.shape(), dtype_, ctx.stream()));
  if (last_batch_ == 0) {
    if (grad_input != nullptr) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }
  auto rows = detail::RowsForGrid(features_, "BatchNorm1d");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchBatchNorm1dBackwardKernel(
        ctx, dtype_, grad_output, cached_x_hat_, gamma_, &backward_output_, &grad_gamma_,
        &grad_beta_, inv_std_, rows.value(), last_batch_, features_, last_training_));
  }

  if (grad_input != nullptr) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void BatchNorm1d::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "gamma"), &gamma_, &grad_gamma_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "beta"), &beta_, &grad_beta_});
}

Embedding::Embedding(int64_t vocab_size, int64_t embedding_dim, RuntimeContext &ctx, DType dtype)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim), dtype_(dtype) {
  if (vocab_size_ <= 0 || embedding_dim_ <= 0) {
    init_status_ = Status::InvalidArgument("Embedding dimensions must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "Embedding");
  if (!init_status_.ok()) {
    return;
  }

  auto table = Tensor::AllocateAsync({vocab_size_, embedding_dim_}, dtype_, ctx.stream());
  if (!table.ok()) {
    init_status_ = table.status();
    return;
  }
  auto grad_table =
      Tensor::AllocateAsync({vocab_size_, embedding_dim_}, DType::kFloat32, ctx.stream());
  if (!grad_table.ok()) {
    init_status_ = grad_table.status();
    return;
  }

  table_ = table.value();
  grad_table_ = grad_table.value();

  std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embedding_dim_));
  std::vector<float> host_table(static_cast<size_t>(vocab_size_ * embedding_dim_));
  for (float &v : host_table) {
    v = dist(rng);
  }

  init_status_ = CopyHostFloatsToTensor(&table_, host_table, ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_table_.FillZero(ctx.stream());
}

Status Embedding::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Embedding::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateIntTensor(input, "Embedding input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 1, "Embedding input"));

  last_num_tokens_ = input.dim(0);

  cached_token_ids_ = input;

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_, {last_num_tokens_, embedding_dim_},
                                           dtype_, ctx.stream()));

  int64_t total = last_num_tokens_ * embedding_dim_;
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchEmbeddingForwardKernel(
        ctx, dtype_, table_, cached_token_ids_, &forward_output_, blocks.value(), last_num_tokens_,
        embedding_dim_, vocab_size_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Embedding::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Embedding grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "Embedding grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Embedding grad_output"));
  if (grad_output.dim(0) != last_num_tokens_ || grad_output.dim(1) != embedding_dim_) {
    return Status::InvalidArgument("Embedding grad_output shape mismatch");
  }
  if (!cached_token_ids_.defined()) {
    return Status::RuntimeError("Embedding backward called before forward");
  }

  DLCUDA_RETURN_IF_ERROR(grad_table_.FillZero(ctx.stream()));

  int64_t total = last_num_tokens_ * embedding_dim_;
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchEmbeddingBackwardKernel(
        ctx, dtype_, grad_output, cached_token_ids_, &grad_table_, blocks.value(), last_num_tokens_,
        embedding_dim_, vocab_size_));
  }

  // Token IDs are non-differentiable; upstream gradient terminates here.
  if (grad_input != nullptr) {
    *grad_input = Tensor();
  }
  return Status::Ok();
}

void Embedding::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "table"), &table_, &grad_table_});
}

} // namespace dlcuda
