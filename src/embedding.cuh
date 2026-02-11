#pragma once

#include "sequential.cuh"
#include <cmath>
#include <curand_kernel.h>

__global__ void embeddingForwardKernel(const float *embedding_table,
                                       const int *token_ids, float *output,
                                       int num_tokens, int embedding_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tokens * embedding_dim) {
    int token = idx / embedding_dim;
    int dim = idx % embedding_dim;
    output[idx] = embedding_table[token_ids[token] * embedding_dim + dim];
  }
}

__global__ void embeddingBackwardKernel(const float *output_grad,
                                        const int *token_ids,
                                        float *embedding_grad, int num_tokens,
                                        int embedding_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tokens * embedding_dim) {
    int token = idx / embedding_dim;
    int dim = idx % embedding_dim;
    atomicAdd(&embedding_grad[token_ids[token] * embedding_dim + dim],
              output_grad[idx]);
  }
}

class EmbeddingLayer : public Operation {
private:
  float *d_embedding_;
  float *d_embedding_grad_;
  int *d_token_ids_;
  int vocab_size_, embedding_dim_, num_tokens_;

public:
  EmbeddingLayer(int vocab_size, int embedding_dim, int num_tokens)
      : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
        num_tokens_(num_tokens) {
    int table_size = vocab_size * embedding_dim;
    CUDA_CHECK(cudaMalloc(&d_embedding_, table_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_embedding_grad_, table_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_token_ids_, num_tokens * sizeof(int)));

    curandState_t *d_states;
    CUDA_CHECK(cudaMalloc(&d_states, table_size * sizeof(curandState_t)));
    float scale = sqrtf(2.0f / embedding_dim);
    int blocks = (table_size + 255) / 256;
    initWeightsKernel<<<blocks, 256>>>(d_embedding_, table_size, scale,
                                       d_states);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_states);
  }

  ~EmbeddingLayer() {
    cudaFree(d_embedding_);
    cudaFree(d_embedding_grad_);
    cudaFree(d_token_ids_);
  }

  int input_size() const override { return num_tokens_; }
  int output_size() const override { return num_tokens_ * embedding_dim_; }

  void set_token_ids(const int *h_token_ids) {
    CUDA_CHECK(cudaMemcpy(d_token_ids_, h_token_ids,
                           num_tokens_ * sizeof(int),
                           cudaMemcpyHostToDevice));
  }

  void forward(float *d_input, float *d_output) override {
    (void)d_input;
    int total = num_tokens_ * embedding_dim_;
    int blocks = (total + 255) / 256;
    embeddingForwardKernel<<<blocks, 256>>>(d_embedding_, d_token_ids_,
                                            d_output, num_tokens_,
                                            embedding_dim_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    (void)d_input_grad;
    int table_size = vocab_size_ * embedding_dim_;
    CUDA_CHECK(cudaMemset(d_embedding_grad_, 0, table_size * sizeof(float)));

    int total = num_tokens_ * embedding_dim_;
    int blocks = (total + 255) / 256;
    embeddingBackwardKernel<<<blocks, 256>>>(d_output_grad, d_token_ids_,
                                             d_embedding_grad_, num_tokens_,
                                             embedding_dim_);
  }

  void update_weights(float lr) override {
    int table_size = vocab_size_ * embedding_dim_;
    int blocks = (table_size + 255) / 256;
    sgdUpdateKernel<<<blocks, 256>>>(d_embedding_, d_embedding_grad_, lr,
                                     table_size);
  }

  std::vector<ParamGroup> get_param_groups() override {
    return {{d_embedding_, d_embedding_grad_, vocab_size_ * embedding_dim_}};
  }
};
