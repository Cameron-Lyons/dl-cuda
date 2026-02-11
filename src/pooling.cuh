#pragma once

#include "sequential.cuh"

__global__ void maxPool1DForwardKernel(const float *input, float *output,
                                       int *indices, int input_size,
                                       int pool_size, int stride,
                                       int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < output_size) {
    int start = idx * stride;
    float max_val = -1e30f;
    int max_idx = start;
    for (int p = 0; p < pool_size && start + p < input_size; p++) {
      float val = input[start + p];
      if (val > max_val) {
        max_val = val;
        max_idx = start + p;
      }
    }
    output[idx] = max_val;
    indices[idx] = max_idx;
  }
}

__global__ void maxPool1DBackwardKernel(const float *output_grad,
                                        const int *indices,
                                        float *input_grad, int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < output_size) {
    atomicAdd(&input_grad[indices[idx]], output_grad[idx]);
  }
}

class MaxPool1DLayer : public Operation {
private:
  int input_size_, pool_size_, stride_, output_size_;
  int *d_indices_;

public:
  MaxPool1DLayer(int input_size, int pool_size, int stride)
      : input_size_(input_size), pool_size_(pool_size), stride_(stride) {
    output_size_ = (input_size - pool_size) / stride + 1;
    CUDA_CHECK(cudaMalloc(&d_indices_, output_size_ * sizeof(int)));
  }

  ~MaxPool1DLayer() { cudaFree(d_indices_); }

  int input_size() const override { return input_size_; }
  int output_size() const override { return output_size_; }

  void forward(float *d_input, float *d_output) override {
    int blocks = (output_size_ + 255) / 256;
    maxPool1DForwardKernel<<<blocks, 256>>>(d_input, d_output, d_indices_,
                                            input_size_, pool_size_, stride_,
                                            output_size_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    CUDA_CHECK(cudaMemset(d_input_grad, 0, input_size_ * sizeof(float)));
    int blocks = (output_size_ + 255) / 256;
    maxPool1DBackwardKernel<<<blocks, 256>>>(d_output_grad, d_indices_,
                                             d_input_grad, output_size_);
  }
};

__global__ void maxPool2DForwardKernel(const float *input, float *output,
                                       int *indices, int input_w, int input_h,
                                       int pool_w, int pool_h, int stride_x,
                                       int stride_y, int output_w,
                                       int output_h) {
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ox < output_w && oy < output_h) {
    int out_idx = oy * output_w + ox;
    int start_x = ox * stride_x;
    int start_y = oy * stride_y;

    float max_val = -1e30f;
    int max_idx = start_y * input_w + start_x;

    for (int py = 0; py < pool_h && start_y + py < input_h; py++) {
      for (int px = 0; px < pool_w && start_x + px < input_w; px++) {
        int in_idx = (start_y + py) * input_w + (start_x + px);
        float val = input[in_idx];
        if (val > max_val) {
          max_val = val;
          max_idx = in_idx;
        }
      }
    }
    output[out_idx] = max_val;
    indices[out_idx] = max_idx;
  }
}

__global__ void maxPool2DBackwardKernel(const float *output_grad,
                                        const int *indices,
                                        float *input_grad, int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < output_size) {
    atomicAdd(&input_grad[indices[idx]], output_grad[idx]);
  }
}

class MaxPool2DLayer : public Operation {
private:
  int input_w_, input_h_;
  int pool_w_, pool_h_;
  int stride_x_, stride_y_;
  int output_w_, output_h_;
  int *d_indices_;

public:
  MaxPool2DLayer(int input_w, int input_h, int pool_w, int pool_h,
                 int stride_x, int stride_y)
      : input_w_(input_w), input_h_(input_h), pool_w_(pool_w), pool_h_(pool_h),
        stride_x_(stride_x), stride_y_(stride_y) {
    output_w_ = (input_w - pool_w) / stride_x + 1;
    output_h_ = (input_h - pool_h) / stride_y + 1;
    CUDA_CHECK(
        cudaMalloc(&d_indices_, output_w_ * output_h_ * sizeof(int)));
  }

  ~MaxPool2DLayer() { cudaFree(d_indices_); }

  int input_size() const override { return input_w_ * input_h_; }
  int output_size() const override { return output_w_ * output_h_; }

  void forward(float *d_input, float *d_output) override {
    dim3 block_size(16, 16);
    dim3 grid_size((output_w_ + block_size.x - 1) / block_size.x,
                   (output_h_ + block_size.y - 1) / block_size.y);
    maxPool2DForwardKernel<<<grid_size, block_size>>>(
        d_input, d_output, d_indices_, input_w_, input_h_, pool_w_, pool_h_,
        stride_x_, stride_y_, output_w_, output_h_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    CUDA_CHECK(
        cudaMemset(d_input_grad, 0, input_w_ * input_h_ * sizeof(float)));
    int total = output_w_ * output_h_;
    int blocks = (total + 255) / 256;
    maxPool2DBackwardKernel<<<blocks, 256>>>(d_output_grad, d_indices_,
                                             d_input_grad, total);
  }
};
