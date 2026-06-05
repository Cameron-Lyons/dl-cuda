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
__global__ void EmbeddingBackwardWarpAggregatedKernel(const typename Codec::Storage *grad_output,
                                                      const int32_t *token_ids, float *grad_table,
                                                      int64_t num_tokens, int64_t embedding_dim,
                                                      int64_t vocab_size, int64_t token_blocks) {
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t dim = block / token_blocks;
  int64_t token_block = block - dim * token_blocks;
  int64_t token = token_block * blockDim.x + threadIdx.x;

  bool in_range = dim < embedding_dim && token < num_tokens;
  int32_t token_id = in_range ? token_ids[token] : -1;
  bool valid = in_range && token_id >= 0 && static_cast<int64_t>(token_id) < vocab_size;
  unsigned valid_mask = __ballot_sync(0xffffffffu, valid);
  if (!valid) {
    return;
  }

  int lane = threadIdx.x & 31;
  float value = Codec::Load(grad_output, token * embedding_dim + dim);
  float sum = 0.0f;
  bool leader = true;

  for (int src_lane = 0; src_lane < 32; ++src_lane) {
    unsigned src_mask = 1u << src_lane;
    int32_t other_token_id = __shfl_sync(valid_mask, token_id, src_lane);
    float other_value = __shfl_sync(valid_mask, value, src_lane);
    if ((valid_mask & src_mask) != 0u && other_token_id == token_id) {
      sum += other_value;
      if (src_lane < lane) {
        leader = false;
      }
    }
  }

  if (leader) {
    int64_t table_index = static_cast<int64_t>(token_id) * embedding_dim + dim;
    atomicAdd(&grad_table[table_index], sum);
  }
}

Result<int> EmbeddingWarpAggregatedBlocks(int64_t num_tokens, int64_t embedding_dim,
                                          int64_t *token_blocks_out) {
  if (token_blocks_out == nullptr) {
    return Status::InvalidArgument("Embedding warp aggregation token block output is null");
  }
  auto token_blocks = detail::BlocksForElements(num_tokens, kCudaThreads);
  if (!token_blocks.ok()) {
    return token_blocks.status();
  }
  int64_t total_blocks = static_cast<int64_t>(token_blocks.value()) * embedding_dim;
  if (total_blocks > std::numeric_limits<int>::max()) {
    return Status::Unsupported("Embedding warp-aggregated backward grid is too large");
  }
  *token_blocks_out = token_blocks.value();
  return static_cast<int>(total_blocks);
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
Status LaunchEmbeddingBackwardWarpAggregatedKernel(RuntimeContext &ctx, const Tensor &grad_output,
                                                   const Tensor &token_ids, Tensor *grad_table,
                                                   int blocks, int64_t num_tokens,
                                                   int64_t embedding_dim, int64_t vocab_size,
                                                   int64_t token_blocks) {
  EmbeddingBackwardWarpAggregatedKernel<Codec><<<blocks, kCudaThreads, 0, ctx.stream()>>>(
      grad_output.data_as<typename Codec::Storage>(), token_ids.data_as<int32_t>(),
      grad_table->data_as<float>(), num_tokens, embedding_dim, vocab_size, token_blocks);
  return detail::CheckKernelLaunch("Embedding backward warp-aggregated kernel");
}

Status LaunchEmbeddingBackwardWarpAggregatedKernel(RuntimeContext &ctx, DType dtype,
                                                   const Tensor &grad_output,
                                                   const Tensor &token_ids, Tensor *grad_table,
                                                   int64_t num_tokens, int64_t embedding_dim,
                                                   int64_t vocab_size) {
  int64_t token_blocks = 0;
  auto blocks = EmbeddingWarpAggregatedBlocks(num_tokens, embedding_dim, &token_blocks);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() == 0) {
    return Status::Ok();
  }
  switch (dtype) {
  case DType::kFloat32:
    return LaunchEmbeddingBackwardWarpAggregatedKernel<detail::Float32Codec>(
        ctx, grad_output, token_ids, grad_table, blocks.value(), num_tokens, embedding_dim,
        vocab_size, token_blocks);
  case DType::kFloat16:
    return LaunchEmbeddingBackwardWarpAggregatedKernel<detail::Float16Codec>(
        ctx, grad_output, token_ids, grad_table, blocks.value(), num_tokens, embedding_dim,
        vocab_size, token_blocks);
  case DType::kBFloat16:
    return LaunchEmbeddingBackwardWarpAggregatedKernel<detail::BFloat16Codec>(
        ctx, grad_output, token_ids, grad_table, blocks.value(), num_tokens, embedding_dim,
        vocab_size, token_blocks);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("Embedding warp-aggregated backward does not support dtype " +
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
