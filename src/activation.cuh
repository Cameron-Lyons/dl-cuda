#pragma once

#include "sequential.cuh"

__global__ void reluKernel(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
  }
}

__global__ void reluBackwardKernel(const float *d_output_grad,
                                   const float *d_forward_input,
                                   float *d_input_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_input_grad[idx] =
        d_output_grad[idx] * ((d_forward_input[idx] > 0.0f) ? 1.0f : 0.0f);
  }
}

__global__ void sigmoidKernel(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

__global__ void sigmoidBackwardKernel(const float *d_output_grad,
                                      const float *d_forward_output,
                                      float *d_input_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float s = d_forward_output[idx];
    d_input_grad[idx] = d_output_grad[idx] * s * (1.0f - s);
  }
}

__global__ void tanhActivationKernel(const float *input, float *output,
                                     int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = tanhf(input[idx]);
  }
}

__global__ void tanhBackwardKernel(const float *d_output_grad,
                                   const float *d_forward_output,
                                   float *d_input_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float t = d_forward_output[idx];
    d_input_grad[idx] = d_output_grad[idx] * (1.0f - t * t);
  }
}

class ReLUActivation : public Operation {
private:
  int size_;
  float *d_cached_input;

public:
  ReLUActivation(int size) : size_(size), d_cached_input(nullptr) {
    CUDA_CHECK(cudaMalloc(&d_cached_input, size * sizeof(float)));
  }

  ~ReLUActivation() {
    if (d_cached_input)
      cudaFree(d_cached_input);
  }

  int input_size() const override { return size_; }
  int output_size() const override { return size_; }

  void forward(float *d_input, float *d_output) override {
    CUDA_CHECK(cudaMemcpy(d_cached_input, d_input, size_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    int blocks = (size_ + 255) / 256;
    reluKernel<<<blocks, 256>>>(d_input, d_output, size_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int blocks = (size_ + 255) / 256;
    reluBackwardKernel<<<blocks, 256>>>(d_output_grad, d_cached_input,
                                        d_input_grad, size_);
  }
};

class SigmoidActivation : public Operation {
private:
  int size_;
  float *d_cached_output;

public:
  SigmoidActivation(int size) : size_(size), d_cached_output(nullptr) {
    CUDA_CHECK(cudaMalloc(&d_cached_output, size * sizeof(float)));
  }

  ~SigmoidActivation() {
    if (d_cached_output)
      cudaFree(d_cached_output);
  }

  int input_size() const override { return size_; }
  int output_size() const override { return size_; }

  void forward(float *d_input, float *d_output) override {
    int blocks = (size_ + 255) / 256;
    sigmoidKernel<<<blocks, 256>>>(d_input, d_output, size_);
    CUDA_CHECK(cudaMemcpy(d_cached_output, d_output, size_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int blocks = (size_ + 255) / 256;
    sigmoidBackwardKernel<<<blocks, 256>>>(d_output_grad, d_cached_output,
                                           d_input_grad, size_);
  }
};

class TanhActivation : public Operation {
private:
  int size_;
  float *d_cached_output;

public:
  TanhActivation(int size) : size_(size), d_cached_output(nullptr) {
    CUDA_CHECK(cudaMalloc(&d_cached_output, size * sizeof(float)));
  }

  ~TanhActivation() {
    if (d_cached_output)
      cudaFree(d_cached_output);
  }

  int input_size() const override { return size_; }
  int output_size() const override { return size_; }

  void forward(float *d_input, float *d_output) override {
    int blocks = (size_ + 255) / 256;
    tanhActivationKernel<<<blocks, 256>>>(d_input, d_output, size_);
    CUDA_CHECK(cudaMemcpy(d_cached_output, d_output, size_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int blocks = (size_ + 255) / 256;
    tanhBackwardKernel<<<blocks, 256>>>(d_output_grad, d_cached_output,
                                        d_input_grad, size_);
  }
};
