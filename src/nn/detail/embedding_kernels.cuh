#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

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

} // namespace
} // namespace dlcuda
