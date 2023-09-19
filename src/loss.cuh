#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

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
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

  for (int i = 0; i < elements_per_thread && (idx + i) < n; i++) {
    error[idx + i] = fabsf(y[idx + i] - y_pred[idx + i]);
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
                                              float *error, int n, int classes,
                                              float epsilon = 1e-10) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float loss = 0.0f;
    for (int c = 0; c < classes; c++) {
      int offset = idx * classes + c;
      loss -= y[offset] *
              logf(y_pred[offset] + epsilon); // Added epsilon to avoid log(0)
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
  cudaDeviceSynchronize();
}

float computeAccuracy(int *y, int *y_pred, int n) {
  int *d_correct_preds;
  cudaMalloc(&d_correct_preds, n * sizeof(int));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  accuracyKernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_correct_preds, n);

  // Sum the correct predictions
  int correct_count =
      thrust::reduce(thrust::device, d_correct_preds, d_correct_preds + n);

  cudaFree(d_correct_preds);

  return float(correct_count) / n;
}

float computeF1Score(int *y, int *y_pred, int n) {
  int *d_TP, *d_FP, *d_FN;
  cudaMalloc(&d_TP, n * sizeof(int));
  cudaMalloc(&d_FP, n * sizeof(int));
  cudaMalloc(&d_FN, n * sizeof(int));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  f1Kernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_TP, d_FP, d_FN, n);

  int sum_TP = thrust::reduce(thrust::device, d_TP, d_TP + n);
  int sum_FP = thrust::reduce(thrust::device, d_FP, d_FP + n);
  int sum_FN = thrust::reduce(thrust::device, d_FN, d_FN + n);

  cudaFree(d_TP);
  cudaFree(d_FP);
  cudaFree(d_FN);

  float precision =
      (sum_TP + sum_FP == 0) ? 0 : float(sum_TP) / (sum_TP + sum_FP);
  float recall = (sum_TP + sum_FN == 0) ? 0 : float(sum_TP) / (sum_TP + sum_FN);

  return (precision + recall == 0)
             ? 0
             : 2.0 * (precision * recall) / (precision + recall);
}

float computeMCC(int *y, int *y_pred, int n) {
  int *d_TP, *d_FP, *d_FN, *d_TN;
  cudaMalloc(&d_TP, n * sizeof(int));
  cudaMalloc(&d_FP, n * sizeof(int));
  cudaMalloc(&d_FN, n * sizeof(int));
  cudaMalloc(&d_TN, n * sizeof(int));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  mccKernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_TP, d_FP, d_FN, d_TN, n);

  int sum_TP = thrust::reduce(thrust::device, d_TP, d_TP + n);
  int sum_FP = thrust::reduce(thrust::device, d_FP, d_FP + n);
  int sum_FN = thrust::reduce(thrust::device, d_FN, d_FN + n);
  int sum_TN = thrust::reduce(thrust::device, d_TN, d_TN + n);

  cudaFree(d_TP);
  cudaFree(d_FP);
  cudaFree(d_FN);
  cudaFree(d_TN);

  float numerator = sum_TP * sum_TN - sum_FP * sum_FN;
  float denominator = sqrtf((sum_TP + sum_FP) * (sum_TP + sum_FN) *
                            (sum_TN + sum_FP) * (sum_TN + sum_FN));

  return (denominator == 0) ? 0 : numerator / denominator;
}
