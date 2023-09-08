#include <cuda_runtime.h>

const short N_THREADS = 256;

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

__global__ void layer_norm_kernel(float *output, const float *input,
                                  const float *gamma, const float *beta,
                                  int feature_size, float epsilon) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < feature_size) {
    float mean = 0.0;
    float var = 0.0;
    for (int d = 0; d < feature_size; d++) {
      mean += input[i * feature_size + d];
    }
    mean /= feature_size;

    for (int d = 0; d < feature_size; d++) {
      var += (input[i * feature_size + d] - mean) *
             (input[i * feature_size + d] - mean);
    }
    var /= feature_size;

    // Normalize and scale/shift
    for (int d = 0; d < feature_size; d++) {
      output[i * feature_size + d] = gamma[d] *
                                         (input[i * feature_size + d] - mean) /
                                         sqrtf(var + epsilon) +
                                     beta[d];
    }
  }
}

void layer_normalization(float *output, const float *input, const float *gamma,
                         const float *beta, int batch_size, int feature_size,
                         float epsilon = 1e-5) {
  dim3 threadsPerBlock(N_THREADS);
  dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

  layer_norm_kernel<<<numBlocks, threadsPerBlock>>>(output, input, gamma, beta,
                                                    feature_size, epsilon);

  cudaDeviceSynchronize();
}

void transformerLayer(float *output, const float *input, const float *q_weights,
                      const float *k_weights, const float *v_weights,
                      const float *ff_weights, const float *ff_bias,
                      const float *gamma, const float *beta, int batch_size,
                      int sequence_length, int d_model, int num_heads,
                      float dropout_prob) {
  float *queries;
  float *keys;
  float *values;

  int d_k = d_model / num_heads;

  linearKernel<<<..., ...>>>(queries, input, q_weights, ...);
  linearKernel<<<..., ...>>>(keys, input, k_weights, ...);
  linearKernel<<<..., ...>>>(values, input, v_weights, ...);

  float *attention_output;
  scaled_dot_product_attention_kernel<<<..., ...>>>(attention_output, queries,
                                                    keys, values, ...);

  float *add_norm_output;
  layer_norm_kernel<<<..., ...>>>(add_norm_output, attention_output, gamma,
                                  beta, ...);

  float *ff_output;
  linearKernel<<<..., ...>>>(ff_output, add_norm_output, ff_weights, ff_bias,
                             ...);
  reluKernel<<<..., ...>>>(ff_output, ff_output, ...);

  layer_norm_kernel<<<..., ...>>>(output, ff_output, gamma, beta, ...);

  float *dropout_mask;
  dropoutKernel<<<..., ...>>>(output, output, dropout_mask, dropout_prob, ...);

  cudaFree(queries);
  cudaFree(keys);
  cudaFree(values);
  cudaFree(attention_output);
  cudaFree(add_norm_output);
  cudaFree(ff_output);
  cudaFree(dropout_mask);
}
