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

void computeLoss(float *y, float *y_pred, float *error, int n, int classes,
                 LossType loss_type, dim3 blocks, dim3 threads) {
  switch (loss_type) {
  case SQUARED_ERROR:
    squaredErrorKernel<<<blocks, threads>>>(y, y_pred, error, n);
    break;
  case ABSOLUTE_ERROR:
    absoluteErrorKernel<<<blocks, threads>>>(y, y_pred, error, n);
    break;
  case BINARY_CROSS_ENTROPY:
    binaryCrossEntropyKernel<<<blocks, threads>>>(y, y_pred, error, n);
    break;
  case CATEGORICAL_CROSS_ENTROPY:
    categoricalCrossEntropyKernel<<<blocks, threads>>>(y, y_pred, error, n,
                                                       classes);
    break;
  default:
    printf("Invalid loss type\n");
    break;
  }
  cudaDeviceSynchronize(); // Wait for the kernel to finish
}
