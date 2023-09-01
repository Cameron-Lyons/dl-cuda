#include <cuda_runtime.h>

__global__ void scaled_dot_product_attention_kernel(
    float *output, const float *queries, const float *keys, const float *values,
    int sequence_length, int d_k, float scaling_factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sequence_length && j < sequence_length) {
    float sum = 0.0;
    for (int d = 0; d < d_k; d++) {
      sum += queries[i * d_k + d] * keys[j * d_k + d];
    }
    float score = expf(sum / scaling_factor);

    float weighted_sum = 0.0;
    for (int d = 0; d < d_k; d++) {
      weighted_sum += score * values[j * d_k + d];
    }

    output[i * d_k + j] = weighted_sum;
  }
}

void scaled_dot_product_attention(float *output, const float *queries,
                                  const float *keys, const float *values,
                                  int sequence_length, int d_k) {
  float scaling_factor = sqrtf((float)d_k);
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((sequence_length + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (sequence_length + threadsPerBlock.y - 1) / threadsPerBlock.y);

  scaled_dot_product_attention_kernel<<<numBlocks, threadsPerBlock>>>(
      output, queries, keys, values, sequence_length, d_k, scaling_factor);

  cudaDeviceSynchronize();
}
