#pragma once

#include "sequential.cuh"
#include <cmath>
#include <curand_kernel.h>

static const short NUM_THREADS = 256;

__global__ void linearLayerKernel(const float *d_X, const float *d_W,
                                  const float *d_b, float *d_Y, int n,
                                  int in_features, int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < out_features && row < n) {
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
      sum += d_X[row * in_features + i] * d_W[i * out_features + col];
    }
    d_Y[row * out_features + col] = sum + (d_b ? d_b[col] : 0.0f);
  }
}

__global__ void linearBackwardInputKernel(const float *d_output_grad,
                                          const float *d_W,
                                          float *d_input_grad, int n,
                                          int in_features, int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < in_features && row < n) {
    float sum = 0.0f;
    for (int i = 0; i < out_features; ++i) {
      sum += d_output_grad[row * out_features + i] * d_W[col * out_features + i];
    }
    d_input_grad[row * in_features + col] = sum;
  }
}

__global__ void linearBackwardWeightKernel(const float *d_X,
                                           const float *d_output_grad,
                                           float *d_W_grad, int n,
                                           int in_features, int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < in_features && col < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += d_X[i * in_features + row] * d_output_grad[i * out_features + col];
    }
    d_W_grad[row * out_features + col] = sum;
  }
}

__global__ void linearBackwardBiasKernel(const float *d_output_grad,
                                         float *d_b_grad, int n,
                                         int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += d_output_grad[i * out_features + col];
    }
    d_b_grad[col] = sum;
  }
}

__global__ void sgdUpdateKernel(float *params, const float *grads, float lr,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    params[idx] -= lr * grads[idx];
  }
}

__global__ void initWeightsKernel(float *data, int n, float scale,
                                  curandState_t *states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    curand_init(42 + idx, idx, 0, &states[idx]);
    data[idx] = curand_normal(&states[idx]) * scale;
  }
}

class LinearLayer : public Operation {
private:
  float *d_W, *d_b;
  float *d_W_grad, *d_b_grad;
  float *d_cached_input;
  int n_, in_features_, out_features_;

public:
  LinearLayer(int n, int in_features, int out_features)
      : n_(n), in_features_(in_features), out_features_(out_features) {
    CUDA_CHECK(
        cudaMalloc(&d_W, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, out_features * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_W_grad, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_grad, out_features * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_cached_input, n * in_features * sizeof(float)));

    curandState_t *d_states;
    int total_w = in_features * out_features;
    CUDA_CHECK(cudaMalloc(&d_states, total_w * sizeof(curandState_t)));
    float scale = sqrtf(2.0f / in_features);
    int blocks = (total_w + 255) / 256;
    initWeightsKernel<<<blocks, 256>>>(d_W, total_w, scale, d_states);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_states);

    CUDA_CHECK(cudaMemset(d_b, 0, out_features * sizeof(float)));
  }

  ~LinearLayer() {
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_W_grad);
    cudaFree(d_b_grad);
    cudaFree(d_cached_input);
  }

  int input_size() const override { return n_ * in_features_; }
  int output_size() const override { return n_ * out_features_; }

  float *get_weights() { return d_W; }
  float *get_bias() { return d_b; }
  float *get_weight_grad() { return d_W_grad; }
  float *get_bias_grad() { return d_b_grad; }
  int get_in_features() const { return in_features_; }
  int get_out_features() const { return out_features_; }
  int get_n() const { return n_; }

  void forward(float *d_input, float *d_output) override {
    CUDA_CHECK(cudaMemcpy(d_cached_input, d_input,
                           n_ * in_features_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((out_features_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (n_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linearLayerKernel<<<blocks, threadsPerBlock>>>(d_input, d_W, d_b, d_output,
                                                   n_, in_features_,
                                                   out_features_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    dim3 threadsPerBlock(16, 16);

    dim3 blocks_input(
        (in_features_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n_ + threadsPerBlock.y - 1) / threadsPerBlock.y);
    linearBackwardInputKernel<<<blocks_input, threadsPerBlock>>>(
        d_output_grad, d_W, d_input_grad, n_, in_features_, out_features_);

    dim3 blocks_weight(
        (out_features_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (in_features_ + threadsPerBlock.y - 1) / threadsPerBlock.y);
    linearBackwardWeightKernel<<<blocks_weight, threadsPerBlock>>>(
        d_cached_input, d_output_grad, d_W_grad, n_, in_features_,
        out_features_);

    int blocks_bias = (out_features_ + 255) / 256;
    linearBackwardBiasKernel<<<blocks_bias, 256>>>(d_output_grad, d_b_grad, n_,
                                                   out_features_);
  }

  void update_weights(float lr) {
    int w_size = in_features_ * out_features_;
    int blocks_w = (w_size + 255) / 256;
    sgdUpdateKernel<<<blocks_w, 256>>>(d_W, d_W_grad, lr, w_size);

    int blocks_b = (out_features_ + 255) / 256;
    sgdUpdateKernel<<<blocks_b, 256>>>(d_b, d_b_grad, lr, out_features_);
  }
};

__global__ void lstmKernel(const float *x, const float *Wf, const float *Wi,
                           const float *Wc, const float *Wo, const float *bf,
                           const float *bi, const float *bc, const float *bo,
                           float *h, float *c, int hidden_size,
                           int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = idx / hidden_size;
  int hid = idx % hidden_size;

  if (batch < batch_size && hid < hidden_size) {
    int concat_size = 2 * hidden_size;
    int out_idx = batch * hidden_size + hid;

    float fg = 0.0f, ig = 0.0f, cg = 0.0f, og = 0.0f;

    for (int k = 0; k < concat_size; k++) {
      float val;
      if (k < hidden_size) {
        val = x[batch * hidden_size + k];
      } else {
        val = h[batch * hidden_size + (k - hidden_size)];
      }
      fg += Wf[k * hidden_size + hid] * val;
      ig += Wi[k * hidden_size + hid] * val;
      cg += Wc[k * hidden_size + hid] * val;
      og += Wo[k * hidden_size + hid] * val;
    }

    fg = 1.0f / (1.0f + expf(-(fg + bf[hid])));
    ig = 1.0f / (1.0f + expf(-(ig + bi[hid])));
    cg = tanhf(cg + bc[hid]);
    og = 1.0f / (1.0f + expf(-(og + bo[hid])));

    c[out_idx] = fg * c[out_idx] + ig * cg;
    h[out_idx] = og * tanhf(c[out_idx]);
  }
}

class LSTMLayer : public Operation {
private:
  float *Wf, *Wi, *Wc, *Wo, *bf, *bi, *bc, *bo;
  float *h, *c;
  int hidden_size_, batch_size_, sequence_length_;

public:
  LSTMLayer(int hidden_size, int batch_size, int sequence_length)
      : hidden_size_(hidden_size), batch_size_(batch_size),
        sequence_length_(sequence_length) {
    int weight_size = 2 * hidden_size * hidden_size;

    CUDA_CHECK(cudaMalloc(&Wf, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Wi, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Wc, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Wo, weight_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&bf, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bi, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bc, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bo, hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&h, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, batch_size * hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMemset(Wf, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Wi, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Wc, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Wo, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bf, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bi, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bc, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bo, 0, hidden_size * sizeof(float)));
  }

  ~LSTMLayer() {
    cudaFree(Wf);
    cudaFree(Wi);
    cudaFree(Wc);
    cudaFree(Wo);
    cudaFree(bf);
    cudaFree(bi);
    cudaFree(bc);
    cudaFree(bo);
    cudaFree(h);
    cudaFree(c);
  }

  int input_size() const override {
    return sequence_length_ * batch_size_ * hidden_size_;
  }
  int output_size() const override {
    return sequence_length_ * batch_size_ * hidden_size_;
  }

  void forward(float *d_input, float *d_output) override {
    int total = batch_size_ * hidden_size_;
    int num_blocks = (total + NUM_THREADS - 1) / NUM_THREADS;

    CUDA_CHECK(cudaMemset(h, 0, batch_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(c, 0, batch_size_ * hidden_size_ * sizeof(float)));

    for (int t = 0; t < sequence_length_; t++) {
      float *x_t = d_input + t * batch_size_ * hidden_size_;

      lstmKernel<<<num_blocks, NUM_THREADS>>>(x_t, Wf, Wi, Wc, Wo, bf, bi, bc,
                                              bo, h, c, hidden_size_,
                                              batch_size_);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(d_output + t * batch_size_ * hidden_size_, h,
                             batch_size_ * hidden_size_ * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
  }

  void backward(float * /*d_output_grad*/, float *d_input_grad) override {
    CUDA_CHECK(cudaMemset(d_input_grad, 0, input_size() * sizeof(float)));
  }
};

__global__ void elmanRnnKernel(const float *x, const float *h_prev,
                               const float *Wxh, const float *Whh,
                               const float *b_h, const float *Why,
                               const float *b_y, float *h_out, float *y_out,
                               int batch_size, int input_size, int hidden_size,
                               int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    for (int j = 0; j < hidden_size; j++) {
      float hidden_sum = 0.0f;
      for (int i = 0; i < input_size; i++) {
        hidden_sum += Wxh[i * hidden_size + j] * x[idx * input_size + i];
      }
      for (int i = 0; i < hidden_size; i++) {
        hidden_sum += Whh[i * hidden_size + j] * h_prev[idx * hidden_size + i];
      }
      h_out[idx * hidden_size + j] = tanhf(hidden_sum + b_h[j]);
    }

    for (int j = 0; j < output_size; j++) {
      float output_sum = 0.0f;
      for (int i = 0; i < hidden_size; i++) {
        output_sum +=
            Why[i * output_size + j] * h_out[idx * hidden_size + i];
      }
      y_out[idx * output_size + j] = output_sum + b_y[j];
    }
  }
}

class ElmanRNNLayer : public Operation {
private:
  float *Wxh, *Whh, *b_h, *Why, *b_y;
  float *h;
  int input_size_, hidden_size_, output_size_, batch_size_, sequence_length_;

public:
  ElmanRNNLayer(int input_size, int hidden_size, int output_size,
                int batch_size, int sequence_length)
      : input_size_(input_size), hidden_size_(hidden_size),
        output_size_(output_size), batch_size_(batch_size),
        sequence_length_(sequence_length) {
    CUDA_CHECK(
        cudaMalloc(&Wxh, input_size * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&Whh, hidden_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_h, hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&Why, hidden_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_y, output_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&h, batch_size * hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMemset(h, 0, batch_size * hidden_size * sizeof(float)));
  }

  ~ElmanRNNLayer() {
    cudaFree(Wxh);
    cudaFree(Whh);
    cudaFree(b_h);
    cudaFree(Why);
    cudaFree(b_y);
    cudaFree(h);
  }

  int input_size() const override {
    return sequence_length_ * batch_size_ * input_size_;
  }
  int output_size() const override {
    return sequence_length_ * batch_size_ * output_size_;
  }

  void forward(float *d_input, float *d_output) override {
    int num_blocks = (batch_size_ + NUM_THREADS - 1) / NUM_THREADS;

    CUDA_CHECK(
        cudaMemset(h, 0, batch_size_ * hidden_size_ * sizeof(float)));

    for (int t = 0; t < sequence_length_; t++) {
      float *x_t = d_input + t * batch_size_ * input_size_;
      float *y_t = d_output + t * batch_size_ * output_size_;

      elmanRnnKernel<<<num_blocks, NUM_THREADS>>>(
          x_t, h, Wxh, Whh, b_h, Why, b_y, h, y_t, batch_size_, input_size_,
          hidden_size_, output_size_);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  void backward(float * /*d_output_grad*/, float *d_input_grad) override {
    CUDA_CHECK(cudaMemset(d_input_grad, 0, input_size() * sizeof(float)));
  }
};

__global__ void conv1dKernel(const float *input, const float *kernel,
                             float *output, int inputSize, int kernelSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int halfK = kernelSize / 2;

  if (idx < inputSize) {
    float sum = 0.0f;
    for (int i = -halfK; i <= halfK; i++) {
      if (idx + i >= 0 && idx + i < inputSize) {
        sum += input[idx + i] * kernel[halfK + i];
      }
    }
    output[idx] = sum;
  }
}

class Conv1DLayer : public Operation {
private:
  float *d_kernel;
  int inputSize_, kernelSize_;

public:
  Conv1DLayer(int inputSize, int kernelSize)
      : inputSize_(inputSize), kernelSize_(kernelSize) {
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kernel, 0, kernelSize * sizeof(float)));
  }

  ~Conv1DLayer() { cudaFree(d_kernel); }

  int input_size() const override { return inputSize_; }
  int output_size() const override { return inputSize_; }

  void forward(float *d_input, float *d_output) override {
    int blocks = (inputSize_ + 255) / 256;
    conv1dKernel<<<blocks, 256>>>(d_input, d_kernel, d_output, inputSize_,
                                  kernelSize_);
  }

  void backward(float * /*d_output_grad*/, float *d_input_grad) override {
    CUDA_CHECK(cudaMemset(d_input_grad, 0, inputSize_ * sizeof(float)));
  }
};

__global__ void conv2dKernel(const float *input, int inputWidth,
                             int inputHeight, const float *kernel,
                             int kernelWidth, int kernelHeight, float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int halfKernelWidth = kernelWidth / 2;
  int halfKernelHeight = kernelHeight / 2;

  if (x < inputWidth && y < inputHeight) {
    float value = 0.0f;
    for (int ky = -halfKernelHeight; ky <= halfKernelHeight; ky++) {
      for (int kx = -halfKernelWidth; kx <= halfKernelWidth; kx++) {
        int inX = x + kx;
        int inY = y + ky;
        if (inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
          value += input[inY * inputWidth + inX] *
                   kernel[(ky + halfKernelHeight) * kernelWidth +
                          (kx + halfKernelWidth)];
        }
      }
    }
    output[y * inputWidth + x] = value;
  }
}

class Conv2DLayer : public Operation {
private:
  float *d_kernel;
  int inputWidth_, inputHeight_;
  int kernelWidth_, kernelHeight_;

public:
  Conv2DLayer(int inputWidth, int inputHeight, int kernelWidth,
              int kernelHeight)
      : inputWidth_(inputWidth), inputHeight_(inputHeight),
        kernelWidth_(kernelWidth), kernelHeight_(kernelHeight) {
    CUDA_CHECK(
        cudaMalloc(&d_kernel, kernelWidth * kernelHeight * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(d_kernel, 0, kernelWidth * kernelHeight * sizeof(float)));
  }

  ~Conv2DLayer() { cudaFree(d_kernel); }

  int input_size() const override { return inputWidth_ * inputHeight_; }
  int output_size() const override { return inputWidth_ * inputHeight_; }

  void forward(float *d_input, float *d_output) override {
    dim3 block_size(16, 16);
    dim3 grid_size((inputWidth_ + block_size.x - 1) / block_size.x,
                   (inputHeight_ + block_size.y - 1) / block_size.y);

    conv2dKernel<<<grid_size, block_size>>>(d_input, inputWidth_, inputHeight_,
                                            d_kernel, kernelWidth_,
                                            kernelHeight_, d_output);
  }

  void backward(float * /*d_output_grad*/, float *d_input_grad) override {
    CUDA_CHECK(cudaMemset(d_input_grad, 0,
                           inputWidth_ * inputHeight_ * sizeof(float)));
  }
};
