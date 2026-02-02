#pragma once

#include "activation.cuh"
#include "dropout.cuh"
#include "layers.cuh"
#include "sequential.cuh"

__global__ void scaled_dot_product_attention_kernel(
    float *output, const float *queries, const float *keys, const float *values,
    int sequence_length, int d_k, float scaling_factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sequence_length && j < sequence_length) {
    float sum = 0.0f;
    for (int d = 0; d < d_k; d++) {
      sum += queries[i * d_k + d] * keys[j * d_k + d];
    }
    float score = expf(sum / scaling_factor);

    float weighted_sum = 0.0f;
    for (int d = 0; d < d_k; d++) {
      weighted_sum += score * values[j * d_k + d];
    }

    output[i * d_k + j] = weighted_sum;
  }
}

void scaled_dot_product_attention(float *output, const float *queries,
                                  const float *keys, const float *values,
                                  int sequence_length, int d_k) {
  float scaling_factor = sqrtf(static_cast<float>(d_k));
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
    float mean = 0.0f;
    float var = 0.0f;
    for (int d = 0; d < feature_size; d++) {
      mean += input[i * feature_size + d];
    }
    mean /= feature_size;

    for (int d = 0; d < feature_size; d++) {
      var += (input[i * feature_size + d] - mean) *
             (input[i * feature_size + d] - mean);
    }
    var /= feature_size;

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
                         float epsilon = 1e-5f) {
  dim3 threadsPerBlock(256);
  dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);

  layer_norm_kernel<<<numBlocks, threadsPerBlock>>>(output, input, gamma, beta,
                                                    feature_size, epsilon);

  cudaDeviceSynchronize();
}

__global__ void residualAddKernel(const float *a, const float *b, float *out,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

class TransformerLayer : public Operation {
private:
  float *d_input, *d_output, *d_q_weights, *d_k_weights, *d_v_weights,
      *d_ff_weights, *d_ff_bias, *d_gamma1, *d_beta1, *d_gamma2, *d_beta2;
  float *d_queries, *d_keys, *d_values, *d_attention_output,
      *d_add_norm_output, *d_ff_output, *d_residual_tmp;
  int batch_size_, sequence_length_, d_model_, num_heads_;
  float dropout_prob_;

  void allocateDeviceMemory() {
    int seq_total = batch_size_ * sequence_length_;
    int d_k = d_model_ / num_heads_;

    CUDA_CHECK(cudaMalloc(&d_input, seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_q_weights, d_model_ * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_k_weights, d_model_ * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_v_weights, d_model_ * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_ff_weights, d_model_ * d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_bias, d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1, d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta1, d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2, d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta2, d_model_ * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&d_queries, seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_keys, seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_values, seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attention_output,
                           seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_add_norm_output,
                           seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_ff_output, seq_total * d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_tmp,
                           seq_total * d_model_ * sizeof(float)));
  }

  void freeDeviceMemory() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_q_weights);
    cudaFree(d_k_weights);
    cudaFree(d_v_weights);
    cudaFree(d_ff_weights);
    cudaFree(d_ff_bias);
    cudaFree(d_gamma1);
    cudaFree(d_beta1);
    cudaFree(d_gamma2);
    cudaFree(d_beta2);
    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_attention_output);
    cudaFree(d_add_norm_output);
    cudaFree(d_ff_output);
    cudaFree(d_residual_tmp);
  }

public:
  TransformerLayer(int batch_size, int sequence_length, int d_model,
                   int num_heads, float dropout_prob)
      : batch_size_(batch_size), sequence_length_(sequence_length),
        d_model_(d_model), num_heads_(num_heads), dropout_prob_(dropout_prob) {
    allocateDeviceMemory();

    float ones = 1.0f;
    for (int i = 0; i < d_model_; i++) {
      cudaMemcpy(d_gamma1 + i, &ones, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_gamma2 + i, &ones, sizeof(float), cudaMemcpyHostToDevice);
    }
    CUDA_CHECK(cudaMemset(d_beta1, 0, d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta2, 0, d_model_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ff_bias, 0, d_model_ * sizeof(float)));
  }

  ~TransformerLayer() { freeDeviceMemory(); }

  int input_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }
  int output_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }

  void forward(float *input, float *output) override {
    int seq_total = batch_size_ * sequence_length_;
    int total_elements = seq_total * d_model_;

    CUDA_CHECK(cudaMemcpy(d_input, input, total_elements * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 proj_blocks((d_model_ + 15) / 16, (seq_total + 15) / 16);

    linearLayerKernel<<<proj_blocks, threadsPerBlock>>>(
        d_input, d_q_weights, nullptr, d_queries, seq_total, d_model_,
        d_model_);
    linearLayerKernel<<<proj_blocks, threadsPerBlock>>>(
        d_input, d_k_weights, nullptr, d_keys, seq_total, d_model_, d_model_);
    linearLayerKernel<<<proj_blocks, threadsPerBlock>>>(
        d_input, d_v_weights, nullptr, d_values, seq_total, d_model_,
        d_model_);
    CUDA_CHECK(cudaDeviceSynchronize());

    int d_k = d_model_ / num_heads_;
    float scaling_factor = sqrtf(static_cast<float>(d_k));
    dim3 attn_threads(16, 16);
    dim3 attn_blocks((sequence_length_ + 15) / 16,
                     (sequence_length_ + 15) / 16);
    scaled_dot_product_attention_kernel<<<attn_blocks, attn_threads>>>(
        d_attention_output, d_queries, d_keys, d_values, sequence_length_, d_k,
        scaling_factor);
    CUDA_CHECK(cudaDeviceSynchronize());

    int add_blocks = (total_elements + 255) / 256;
    residualAddKernel<<<add_blocks, 256>>>(d_input, d_attention_output,
                                           d_residual_tmp, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 ln_blocks((seq_total + 255) / 256);
    layer_norm_kernel<<<ln_blocks, 256>>>(d_add_norm_output, d_residual_tmp,
                                          d_gamma1, d_beta1, d_model_, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    linearLayerKernel<<<proj_blocks, threadsPerBlock>>>(
        d_add_norm_output, d_ff_weights, d_ff_bias, d_ff_output, seq_total,
        d_model_, d_model_);
    CUDA_CHECK(cudaDeviceSynchronize());

    reluKernel<<<add_blocks, 256>>>(d_ff_output, d_ff_output, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    residualAddKernel<<<add_blocks, 256>>>(d_add_norm_output, d_ff_output,
                                           d_residual_tmp, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    layer_norm_kernel<<<ln_blocks, 256>>>(output, d_residual_tmp, d_gamma2,
                                          d_beta2, d_model_, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward(float * /*d_output_grad*/, float *d_input_grad) override {
    CUDA_CHECK(
        cudaMemset(d_input_grad, 0, input_size() * sizeof(float)));
  }
};
