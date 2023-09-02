#include <cuda_runtime.h>

__global__ void linearLayerKernel(float *X, float *W, float *b, float *Y, int n,
                                  int in_features, int out_features) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
      sum += X[row * in_features + i] * W[i * out_features + col];
    }
    Y[row * out_features + col] = sum + b[col];
  }
}

__global__ void lstmKernel(float *x, float *h_prev, float *c_prev, float *Wf,
                           float *Wi, float *Wc, float *Wo, float *bf,
                           float *bi, float *bc, float *bo, float *h, float *c,
                           int sequence_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < sequence_length) {
    // Forget gate
    float ft = sigmoid(dot(Wf, concat(h_prev, x[idx])) + bf);
    // Input gate
    float it = sigmoid(dot(Wi, concat(h_prev, x[idx])) + bi);
    // Cell state
    float c_tilde = tanh(dot(Wc, concat(h_prev, x[idx])) + bc);
    c[idx] = ft * c_prev[idx] + it * c_tilde;
    // Output gate
    float ot = sigmoid(dot(Wo, concat(h_prev, x[idx])) + bo);
    // Hidden state
    h[idx] = ot * tanh(c[idx]);
  }
}

__global__ void elmanRnnKernel(float *x, float *h_prev, float *Wxh, float *Whh,
                               float *b_h, float *Why, float *b_y, float *h,
                               float *y, int sequence_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < sequence_length) {
    float hidden_sum = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      hidden_sum += Wxh[idx * HIDDEN_SIZE + i] * x[idx] + Whh[i] * h_prev[idx];
    }
    h[idx] = tanhf(hidden_sum + b_h[idx]);

    float output_sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      output_sum += Why[idx * OUTPUT_SIZE + i] * h[idx];
    }
    y[idx] = output_sum + b_y[idx];
  }
}

__global__ void conv1dKernel(float *input, float *kernel, float *output,
                             int inputSize, int kernelSize) {
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

__global__ void conv2dKernel(float *input, int inputWidth, int inputHeight,
                             float *kernel, int kernelWidth, int kernelHeight,
                             float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int halfKernelWidth = kernelWidth / 2;
  int halfKernelHeight = kernelHeight / 2;

  float value = 0.0f;

  if (x < inputWidth && y < inputHeight) {
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
