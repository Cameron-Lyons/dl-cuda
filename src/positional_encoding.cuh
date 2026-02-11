#pragma once

#include "sequential.cuh"
#include <cmath>
#include <vector>

__global__ void addPositionalEncodingKernel(const float *input,
                                            const float *pe_table,
                                            float *output, int batch_size,
                                            int seq_len, int d_model) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * seq_len * d_model;
  if (idx < total) {
    int pe_idx = idx % (seq_len * d_model);
    output[idx] = input[idx] + pe_table[pe_idx];
  }
}

class PositionalEncoding : public Operation {
private:
  float *d_pe_table_;
  float *d_cached_input_;
  int batch_size_, seq_len_, d_model_;

public:
  PositionalEncoding(int batch_size, int max_seq_len, int d_model)
      : batch_size_(batch_size), seq_len_(max_seq_len), d_model_(d_model) {
    int table_size = max_seq_len * d_model;
    CUDA_CHECK(cudaMalloc(&d_pe_table_, table_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cached_input_,
                           batch_size * table_size * sizeof(float)));

    std::vector<float> h_pe(table_size);
    for (int pos = 0; pos < max_seq_len; pos++) {
      for (int i = 0; i < d_model; i++) {
        float angle =
            static_cast<float>(pos) /
            powf(10000.0f, static_cast<float>(2 * (i / 2)) / d_model);
        h_pe[pos * d_model + i] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
      }
    }
    CUDA_CHECK(cudaMemcpy(d_pe_table_, h_pe.data(), table_size * sizeof(float),
                           cudaMemcpyHostToDevice));
  }

  ~PositionalEncoding() {
    cudaFree(d_pe_table_);
    cudaFree(d_cached_input_);
  }

  int input_size() const override {
    return batch_size_ * seq_len_ * d_model_;
  }
  int output_size() const override {
    return batch_size_ * seq_len_ * d_model_;
  }

  void forward(float *d_input, float *d_output) override {
    int total = batch_size_ * seq_len_ * d_model_;
    int blocks = (total + 255) / 256;
    addPositionalEncodingKernel<<<blocks, 256>>>(d_input, d_pe_table_, d_output,
                                                  batch_size_, seq_len_,
                                                  d_model_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int total = batch_size_ * seq_len_ * d_model_;
    CUDA_CHECK(cudaMemcpy(d_input_grad, d_output_grad, total * sizeof(float),
                           cudaMemcpyDeviceToDevice));
  }
};
