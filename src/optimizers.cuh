#pragma once

#include <cmath>
#include <cuda_runtime.h>

__global__ void updateRMSprop(float *d_g, float *d_s, float *d_theta,
                              float learning_rate, float decay_rate,
                              float epsilon, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_s[idx] = decay_rate * d_s[idx] + (1 - decay_rate) * d_g[idx] * d_g[idx];
    d_theta[idx] -= learning_rate * d_g[idx] / (sqrtf(d_s[idx]) + epsilon);
  }
}

__global__ void updateAdam(float *d_g, float *d_m, float *d_v, float *d_theta,
                           float alpha, float beta1, float beta2, float epsilon,
                           float inv_bias_correction1,
                           float inv_bias_correction2, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_m[idx] = beta1 * d_m[idx] + (1.0f - beta1) * d_g[idx];
    d_v[idx] = beta2 * d_v[idx] + (1.0f - beta2) * d_g[idx] * d_g[idx];

    float m_hat = d_m[idx] * inv_bias_correction1;
    float v_hat = d_v[idx] * inv_bias_correction2;

    d_theta[idx] -= alpha * m_hat / (sqrtf(v_hat) + epsilon);
  }
}

__global__ void updateAdamW(float *d_g, float *d_m, float *d_v, float *d_theta,
                            float alpha, float beta1, float beta2,
                            float epsilon, float weight_decay,
                            float inv_bias_correction1,
                            float inv_bias_correction2, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_m[idx] = beta1 * d_m[idx] + (1.0f - beta1) * d_g[idx];
    d_v[idx] = beta2 * d_v[idx] + (1.0f - beta2) * d_g[idx] * d_g[idx];

    float m_hat = d_m[idx] * inv_bias_correction1;
    float v_hat = d_v[idx] * inv_bias_correction2;

    d_theta[idx] -= alpha * (m_hat / (sqrtf(v_hat) + epsilon) +
                              weight_decay * d_theta[idx]);
  }
}

__global__ void clipGradsKernel(float *grads, float clip_coeff, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    grads[idx] *= clip_coeff;
  }
}
