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

__global__ void softmaxForwardGeneralKernel(const float *input, float *output,
                                            int num_rows, int row_width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *in_row = input + row * row_width;
    float *out_row = output + row * row_width;

    float max_val = in_row[0];
    for (int j = 1; j < row_width; j++) {
      max_val = fmaxf(max_val, in_row[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < row_width; j++) {
      out_row[j] = expf(in_row[j] - max_val);
      sum += out_row[j];
    }

    float inv_sum = 1.0f / sum;
    for (int j = 0; j < row_width; j++) {
      out_row[j] *= inv_sum;
    }
  }
}

__global__ void softmaxBackwardGeneralKernel(const float *d_output_grad,
                                             const float *softmax_output,
                                             float *d_input_grad, int num_rows,
                                             int row_width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *dy = d_output_grad + row * row_width;
    const float *s = softmax_output + row * row_width;
    float *dx = d_input_grad + row * row_width;

    float dot = 0.0f;
    for (int j = 0; j < row_width; j++) {
      dot += dy[j] * s[j];
    }

    for (int j = 0; j < row_width; j++) {
      dx[j] = s[j] * (dy[j] - dot);
    }
  }
}

class SoftmaxActivation : public Operation {
private:
  int num_rows_, row_width_;
  float *d_cached_output;

public:
  SoftmaxActivation(int num_rows, int row_width)
      : num_rows_(num_rows), row_width_(row_width), d_cached_output(nullptr) {
    CUDA_CHECK(
        cudaMalloc(&d_cached_output, num_rows * row_width * sizeof(float)));
  }

  ~SoftmaxActivation() {
    if (d_cached_output)
      cudaFree(d_cached_output);
  }

  int input_size() const override { return num_rows_ * row_width_; }
  int output_size() const override { return num_rows_ * row_width_; }

  void forward(float *d_input, float *d_output) override {
    int blocks = (num_rows_ + 255) / 256;
    softmaxForwardGeneralKernel<<<blocks, 256>>>(d_input, d_output, num_rows_,
                                                  row_width_);
    CUDA_CHECK(cudaMemcpy(d_cached_output, d_output,
                           num_rows_ * row_width_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int blocks = (num_rows_ + 255) / 256;
    softmaxBackwardGeneralKernel<<<blocks, 256>>>(
        d_output_grad, d_cached_output, d_input_grad, num_rows_, row_width_);
  }
};

__global__ void geluForwardKernel(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
  }
}

__global__ void geluBackwardKernel(const float *d_output_grad,
                                   const float *d_forward_input,
                                   float *d_input_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = d_forward_input[idx];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float d_inner = 0.7978845608f * (1.0f + 0.134145f * x * x);
    float cdf = 0.5f * (1.0f + tanh_val);
    d_input_grad[idx] = d_output_grad[idx] * (cdf + x * 0.5f * sech2 * d_inner);
  }
}

class GELUActivation : public Operation {
private:
  int size_;
  float *d_cached_input;

public:
  GELUActivation(int size) : size_(size), d_cached_input(nullptr) {
    CUDA_CHECK(cudaMalloc(&d_cached_input, size * sizeof(float)));
  }

  ~GELUActivation() {
    if (d_cached_input)
      cudaFree(d_cached_input);
  }

  int input_size() const override { return size_; }
  int output_size() const override { return size_; }

  void forward(float *d_input, float *d_output) override {
    CUDA_CHECK(cudaMemcpy(d_cached_input, d_input, size_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    int blocks = (size_ + 255) / 256;
    geluForwardKernel<<<blocks, 256>>>(d_input, d_output, size_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int blocks = (size_ + 255) / 256;
    geluBackwardKernel<<<blocks, 256>>>(d_output_grad, d_cached_input,
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
