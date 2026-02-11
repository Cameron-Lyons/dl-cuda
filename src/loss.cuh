#pragma once

#include "metrics.cuh"
#include "sequential.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

enum LossType {
  SQUARED_ERROR,
  ABSOLUTE_ERROR,
  BINARY_CROSS_ENTROPY,
  CATEGORICAL_CROSS_ENTROPY
};

static const int LOSS_ELEMENTS_PER_THREAD = 2;

__global__ void squaredErrorKernel(float *y, float *y_pred, float *error,
                                   int n) {
  __shared__ float shared_y[256];
  __shared__ float shared_y_pred[256];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    shared_y[threadIdx.x] = y[idx];
    shared_y_pred[threadIdx.x] = y_pred[idx];
    __syncthreads();

    float diff = shared_y[threadIdx.x] - shared_y_pred[threadIdx.x];
    error[idx] = diff * diff;
  }
}

__global__ void absoluteErrorKernel(float *y, float *y_pred, float *error,
                                    int n, int elements_per_thread) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;

  for (int i = 0; i < elements_per_thread && (idx + i) < n; i++) {
    error[idx + i] = fabsf(y[idx + i] - y_pred[idx + i]);
  }
}

__global__ void binaryCrossEntropyKernelSafe(float *y, float *y_pred,
                                             float *error, int n,
                                             float epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float pred_value = y_pred[idx];
    if (pred_value < epsilon) {
      pred_value = epsilon;
    } else if (pred_value > 1.0f - epsilon) {
      pred_value = 1.0f - epsilon;
    }

    error[idx] =
        -y[idx] * logf(pred_value) - (1.0f - y[idx]) * logf(1.0f - pred_value);
  }
}

__global__ void categoricalCrossEntropyKernel(float *y, float *y_pred,
                                              float *error, int n, int classes,
                                              float epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float loss = 0.0f;
    for (int c = 0; c < classes; c++) {
      int offset = idx * classes + c;
      loss -= y[offset] * logf(y_pred[offset] + epsilon);
    }
    error[idx] = loss;
  }
}

__global__ void squaredErrorBackwardKernel(float *y, float *y_pred,
                                           float *grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    grad[idx] = 2.0f * (y_pred[idx] - y[idx]) / n;
  }
}

__global__ void absoluteErrorBackwardKernel(float *y, float *y_pred,
                                            float *grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float diff = y_pred[idx] - y[idx];
    grad[idx] = ((diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f)) / n;
  }
}

__global__ void binaryCrossEntropyBackwardKernel(float *y, float *y_pred,
                                                 float *grad, int n,
                                                 float epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float p = y_pred[idx];
    if (p < epsilon)
      p = epsilon;
    else if (p > 1.0f - epsilon)
      p = 1.0f - epsilon;
    grad[idx] = (-y[idx] / p + (1.0f - y[idx]) / (1.0f - p)) / n;
  }
}

__global__ void categoricalCrossEntropyBackwardKernel(float *y, float *y_pred,
                                                      float *grad, int n,
                                                      int classes,
                                                      float epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * classes) {
    grad[idx] = -y[idx] / (y_pred[idx] + epsilon) / n;
  }
}

void computeLoss(float *y, float *y_pred, float *error, int n, int classes,
                 LossType loss_type, dim3 blocks, dim3 threads) {
  switch (loss_type) {
  case SQUARED_ERROR:
    squaredErrorKernel<<<blocks, threads>>>(y, y_pred, error, n);
    break;
  case ABSOLUTE_ERROR:
    absoluteErrorKernel<<<blocks, threads>>>(y, y_pred, error, n,
                                             LOSS_ELEMENTS_PER_THREAD);
    break;
  case BINARY_CROSS_ENTROPY:
    binaryCrossEntropyKernelSafe<<<blocks, threads>>>(y, y_pred, error, n,
                                                      1e-10f);
    break;
  case CATEGORICAL_CROSS_ENTROPY:
    categoricalCrossEntropyKernel<<<blocks, threads>>>(y, y_pred, error, n,
                                                       classes, 1e-10f);
    break;
  default:
    printf("Invalid loss type\n");
    break;
  }
  cudaDeviceSynchronize();
}

void computeLossBackward(float *y, float *y_pred, float *grad, int n,
                         int classes, LossType loss_type) {
  switch (loss_type) {
  case SQUARED_ERROR: {
    int blocks = (n + 255) / 256;
    squaredErrorBackwardKernel<<<blocks, 256>>>(y, y_pred, grad, n);
    break;
  }
  case ABSOLUTE_ERROR: {
    int blocks = (n + 255) / 256;
    absoluteErrorBackwardKernel<<<blocks, 256>>>(y, y_pred, grad, n);
    break;
  }
  case BINARY_CROSS_ENTROPY: {
    int blocks = (n + 255) / 256;
    binaryCrossEntropyBackwardKernel<<<blocks, 256>>>(y, y_pred, grad, n,
                                                      1e-10f);
    break;
  }
  case CATEGORICAL_CROSS_ENTROPY: {
    int total = n * classes;
    int blocks = (total + 255) / 256;
    categoricalCrossEntropyBackwardKernel<<<blocks, 256>>>(y, y_pred, grad, n,
                                                           classes, 1e-10f);
    break;
  }
  default:
    printf("Unsupported loss type for backward\n");
    break;
  }
  cudaDeviceSynchronize();
}
