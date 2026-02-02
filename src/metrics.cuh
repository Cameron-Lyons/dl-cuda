#pragma once

#include <cuda_runtime.h>
#include <cuda/std/functional>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

__global__ void r2Kernel(float *y, float *y_pred, float *numerator,
                         float *denominator, float y_mean, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float diff = y[idx] - y_pred[idx];
    numerator[idx] = diff * diff;

    float diff_mean = y[idx] - y_mean;
    denominator[idx] = diff_mean * diff_mean;
  }
}

__global__ void accuracyKernel(int *y, int *y_pred, int *correct_preds,
                               int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    correct_preds[idx] = (y[idx] == y_pred[idx]) ? 1 : 0;
  }
}

__global__ void computeMetricsKernel(int *y, int *y_pred, int *TP, int *FP,
                                     int *FN, int *TN, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    TP[idx] = (y[idx] == 1 && y_pred[idx] == 1) ? 1 : 0;
    FP[idx] = (y[idx] == 0 && y_pred[idx] == 1) ? 1 : 0;
    FN[idx] = (y[idx] == 1 && y_pred[idx] == 0) ? 1 : 0;
    TN[idx] = (y[idx] == 0 && y_pred[idx] == 0) ? 1 : 0;
  }
}

float r2Loss(float *y, float *y_pred, float y_mean, int n) {
  float *d_numerator, *d_denominator;
  cudaMalloc(&d_numerator, n * sizeof(float));
  cudaMalloc(&d_denominator, n * sizeof(float));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  r2Kernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_numerator, d_denominator,
                                        y_mean, n);

  float numerator = thrust::reduce(thrust::device, d_numerator, d_numerator + n,
                                   0.0f, cuda::std::plus<float>());
  float denominator =
      thrust::reduce(thrust::device, d_denominator, d_denominator + n, 0.0f,
                     cuda::std::plus<float>());

  cudaFree(d_numerator);
  cudaFree(d_denominator);

  if (denominator == 0)
    return 0;
  return 1.0f - (numerator / denominator);
}

float computeAccuracy(int *y, int *y_pred, int n) {
  int *d_correct_preds;
  cudaMalloc(&d_correct_preds, n * sizeof(int));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  accuracyKernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_correct_preds, n);

  int correct_count =
      thrust::reduce(thrust::device, d_correct_preds, d_correct_preds + n);

  cudaFree(d_correct_preds);

  return float(correct_count) / n;
}

float computeF1Score(int *y, int *y_pred, int n) {
  int *d_TP, *d_FP, *d_FN, *d_TN;
  cudaMalloc(&d_TP, n * sizeof(int));
  cudaMalloc(&d_FP, n * sizeof(int));
  cudaMalloc(&d_FN, n * sizeof(int));
  cudaMalloc(&d_TN, n * sizeof(int));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  computeMetricsKernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_TP, d_FP,
                                                    d_FN, d_TN, n);

  int sum_TP = thrust::reduce(thrust::device, d_TP, d_TP + n);
  int sum_FP = thrust::reduce(thrust::device, d_FP, d_FP + n);
  int sum_FN = thrust::reduce(thrust::device, d_FN, d_FN + n);

  cudaFree(d_TP);
  cudaFree(d_FP);
  cudaFree(d_FN);
  cudaFree(d_TN);

  float precision =
      (sum_TP + sum_FP == 0) ? 0.0f : float(sum_TP) / (sum_TP + sum_FP);
  float recall =
      (sum_TP + sum_FN == 0) ? 0.0f : float(sum_TP) / (sum_TP + sum_FN);

  return (precision + recall == 0)
             ? 0.0f
             : 2.0f * (precision * recall) / (precision + recall);
}

float computeMCC(int *y, int *y_pred, int n) {
  int *d_TP, *d_FP, *d_FN, *d_TN;
  cudaMalloc(&d_TP, n * sizeof(int));
  cudaMalloc(&d_FP, n * sizeof(int));
  cudaMalloc(&d_FN, n * sizeof(int));
  cudaMalloc(&d_TN, n * sizeof(int));

  int threadsPerBlock = 256;
  int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  computeMetricsKernel<<<blocks, threadsPerBlock>>>(y, y_pred, d_TP, d_FP,
                                                    d_FN, d_TN, n);

  int sum_TP = thrust::reduce(thrust::device, d_TP, d_TP + n);
  int sum_FP = thrust::reduce(thrust::device, d_FP, d_FP + n);
  int sum_FN = thrust::reduce(thrust::device, d_FN, d_FN + n);
  int sum_TN = thrust::reduce(thrust::device, d_TN, d_TN + n);

  cudaFree(d_TP);
  cudaFree(d_FP);
  cudaFree(d_FN);
  cudaFree(d_TN);

  float numerator = float(sum_TP * sum_TN - sum_FP * sum_FN);
  float denominator = sqrtf(float((sum_TP + sum_FP) * (sum_TP + sum_FN) *
                                  (sum_TN + sum_FP) * (sum_TN + sum_FN)));

  return (denominator == 0) ? 0.0f : numerator / denominator;
}
