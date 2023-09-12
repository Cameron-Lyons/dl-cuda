#include <cuda_runtime.h>

__global__ void squaredErrorKernel(float *y, float *y_pred, float *error,
                                   int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float diff = y[idx] - y_pred[idx];
    error[idx] = diff * diff;
  }
}

__global__ void absoluteErrorKernel(float *y, float *y_pred, float *error,
                                    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    error[idx] = fabsf(y[idx] - y_pred[idx]);
  }
}

__global__ void binaryCrossEntropyKernel(float *y, float *y_pred, float *error,
                                         int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    error[idx] = -y[idx] * logf(y_pred[idx]) -
                 (1.0f - y[idx]) * logf(1.0f - y_pred[idx]);
  }
}

__global__ void categoricalCrossEntropyKernel(float *y, float *y_pred,
                                              float *error, int n,
                                              int classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float loss = 0.0f;
    for (int c = 0; c < classes; c++) {
      int offset = idx * classes + c;
      loss -= y[offset] *
              logf(y_pred[offset] + 1e-10); // Added epsilon to avoid log(0)
    }
    error[idx] = loss;
  }
}

float computeLoss(float *d_y, float *d_y_pred, int n, bool useMSE = true) {
  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  float *d_error;
  cudaMalloc(&d_error, n * sizeof(float));

  if (useMSE) {
    squaredErrorKernel<<<blocks, threadsPerBlock>>>(d_y, d_y_pred, d_error, n);
  } else {
    absoluteErrorKernel<<<blocks, threadsPerBlock>>>(d_y, d_y_pred, d_error, n);
  }
