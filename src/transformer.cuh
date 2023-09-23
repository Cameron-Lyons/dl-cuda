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
                                  const float *gammas, const float *beta,
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

class TransformerLayer : public Operation {
private:
  float *d_input, *d_output, *d_q_weights, *d_k_weights, *d_v_weights,
      *d_ff_weights, *d_ff_bias, *d_gamma, *d_beta, *d_dropout_mask;
  float *d_queries, *d_keys, *d_values, *d_attention_output, *d_add_norm_output,
      *d_ff_output;
  int batch_size, sequence_length, d_model, num_heads;
  float dropout_prob;

  void allocateDeviceMemory() {
    int d_k = d_model / num_heads;

    checkCudaErrors(cudaMalloc(&d_input, batch_size * sequence_length *
                                             d_model * sizeof(float)),
                    "Allocate d_input");
    checkCudaErrors(cudaMalloc(&d_output, batch_size * sequence_length *
                                              d_model * sizeof(float)),
                    "Allocate d_output");
    checkCudaErrors(cudaMalloc(&d_q_weights, d_model * d_model * sizeof(float)),
                    "Allocate d_q_weights");
    checkCudaErrors(cudaMalloc(&d_k_weights, d_model * d_model * sizeof(float)),
                    "Allocate d_k_weights");
    checkCudaErrors(cudaMalloc(&d_v_weights, d_model * d_model * sizeof(float)),
                    "Allocate d_v_weights");
    checkCudaErrors(
        cudaMalloc(&d_ff_weights, d_model * d_model * sizeof(float)),
        "Allocate d_ff_weights");
    checkCudaErrors(cudaMalloc(&d_ff_bias, d_model * sizeof(float)),
                    "Allocate d_ff_bias");
    checkCudaErrors(cudaMalloc(&d_gamma, d_model * sizeof(float)),
                    "Allocate d_gamma");
    checkCudaErrors(cudaMalloc(&d_beta, d_model * sizeof(float)),
                    "Allocate d_beta");
    checkCudaErrors(cudaMalloc(&d_dropout_mask, batch_size * sequence_length *
                                                    d_model * sizeof(float)),
                    "Allocate d_dropout_mask");

    // Intermediate tensors
    checkCudaErrors(cudaMalloc(&d_queries, batch_size * sequence_length * d_k *
                                               sizeof(float)),
                    "Allocate d_queries");
    checkCudaErrors(
        cudaMalloc(&d_keys, batch_size * sequence_length * d_k * sizeof(float)),
        "Allocate d_keys");
    checkCudaErrors(cudaMalloc(&d_values, batch_size * sequence_length * d_k *
                                              sizeof(float)),
                    "Allocate d_values");
    checkCudaErrors(
        cudaMalloc(&d_attention_output,
                   batch_size * sequence_length * d_k * sizeof(float)),
        "Allocate d_attention_output");
    checkCudaErrors(
        cudaMalloc(&d_add_norm_output,
                   batch_size * sequence_length * d_model * sizeof(float)),
        "Allocate d_add_norm_output");
    checkCudaErrors(cudaMalloc(&d_ff_output, batch_size * sequence_length *
                                                 d_model * sizeof(float)),
                    "Allocate d_ff_output");
  }

  void freeDeviceMemory() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_q_weights);
    cudaFree(d_k_weights);
    cudaFree(d_v_weights);
    cudaFree(d_ff_weights);
    cudaFree(d_ff_bias);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_dropout_mask);
    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_attention_output);
    cudaFree(d_add_norm_output);
    cudaFree(d_ff_output);
  }

public:
  TransformerLayer(int batch_size, int sequence_length, int d_model,
                   int num_heads, float dropout_prob)
      : batch_size(batch_size), sequence_length(sequence_length),
        d_model(d_model), num_heads(num_heads), dropout_prob(dropout_prob) {
    allocateDeviceMemory();
  }

  ~TransformerLayer() { freeDeviceMemory(); }

  void forward(const float *input, const float *q_weights,
               const float *k_weights, const float *v_weights,
               const float *ff_weights, const float *ff_bias,
               const float *gamma, const float *beta, float *output) {
    cudaMemcpy(d_input, input,
               batch_size * sequence_length * d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_weights, q_weights, d_model * d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_weights, k_weights, d_model * d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_weights, v_weights, d_model * d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ff_weights, ff_weights, d_model * d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ff_bias, ff_bias, d_model * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, d_model * sizeof(float), cudaMemcpyHostToDevice);

    // I'm using placeholder ellipses (...) for kernel configuration and
    // parameters, which you need to replace
    linearKernel<<<...>>>(d_queries, d_input, d_q_weights, ...);
    linearKernel<<<...>>>(d_keys, d_input, d_k_weights, ...);
    linearKernel<<<...>>>(d_values, d_input, d_v_weights, ...);

    scaled_dot_product_attention_kernel<<<...>>>(d_attention_output, d_queries,
                                                 d_keys, d_values, ...);

    layer_norm_kernel<<<...>>>(d_add_norm_output, d_attention_output, d_gamma,
                               d_beta, ...);

    linearKernel<<<...>>>(d_ff_output, d_add_norm_output, d_ff_weights,
                          d_ff_bias, ...);
    reluKernel<<<...>>>(d_ff_output, d_ff_output, ...);

    layer_norm_kernel<<<...>>>(d_output, d_ff_output, d_gamma, d_beta, ...);
    dropoutKernel<<<...>>>(d_output, d_output, d_dropout_mask, dropout_prob,
                           ...);

    cudaMemcpy(output, d_output,
               batch_size * sequence_length * d_model * sizeof(float),
               cudaMemcpyDeviceToHost);
  }
};
d_norm_output, attention_output, gamma,
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
