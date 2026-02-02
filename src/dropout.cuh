#pragma once

#include "sequential.cuh"
#include <curand_kernel.h>

static const int ELEMENTS_PER_THREAD = 2;

__global__ void initializeCurandStates(curandState_t *states,
                                       unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed + idx, idx, 0, &states[idx]);
}

__global__ void dropoutKernel(const float *input, float *output, float *mask,
                              float dropout_prob, int size,
                              curandState_t *states) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

  for (int i = 0; i < ELEMENTS_PER_THREAD && (idx + i) < size; i++) {
    float rand_val = curand_uniform(&states[idx + i]);
    mask[idx + i] = (rand_val > dropout_prob) ? 1.0f : 0.0f;
    output[idx + i] = input[idx + i] * mask[idx + i];
  }
}

__global__ void dropoutBackwardKernel(const float *d_output_grad,
                                      const float *mask, float *d_input_grad,
                                      int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_input_grad[idx] = d_output_grad[idx] * mask[idx];
  }
}

class DropoutLayer : public Operation {
private:
  int size_;
  float dropout_prob_;
  bool training_;
  float *d_mask;
  curandState_t *d_states;

public:
  DropoutLayer(int size, float dropout_prob)
      : size_(size), dropout_prob_(dropout_prob), training_(true),
        d_mask(nullptr), d_states(nullptr) {
    CUDA_CHECK(cudaMalloc(&d_mask, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, size * sizeof(curandState_t)));
    int blocks = (size + 255) / 256;
    initializeCurandStates<<<blocks, 256>>>(d_states, 42);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  ~DropoutLayer() {
    if (d_mask)
      cudaFree(d_mask);
    if (d_states)
      cudaFree(d_states);
  }

  void set_training(bool training) { training_ = training; }

  int input_size() const override { return size_; }
  int output_size() const override { return size_; }

  void forward(float *d_input, float *d_output) override {
    if (!training_) {
      CUDA_CHECK(cudaMemcpy(d_output, d_input, size_ * sizeof(float),
                             cudaMemcpyDeviceToDevice));
      return;
    }
    int total_threads = (size_ + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    int blocks = (total_threads + 255) / 256;
    dropoutKernel<<<blocks, 256>>>(d_input, d_output, d_mask, dropout_prob_,
                                   size_, d_states);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    if (!training_) {
      CUDA_CHECK(cudaMemcpy(d_input_grad, d_output_grad,
                             size_ * sizeof(float), cudaMemcpyDeviceToDevice));
      return;
    }
    int blocks = (size_ + 255) / 256;
    dropoutBackwardKernel<<<blocks, 256>>>(d_output_grad, d_mask, d_input_grad,
                                           size_);
  }
};
