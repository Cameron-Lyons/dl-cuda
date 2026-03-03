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

__global__ void updateAdamWMultiTensor(float *const *d_g, float *const *d_m,
                                       float *const *d_v,
                                       float *const *d_theta,
                                       const int *group_sizes,
                                       int num_groups, float alpha, float beta1,
                                       float beta2, float epsilon,
                                       float weight_decay,
                                       float inv_bias_correction1,
                                       float inv_bias_correction2) {
  int group = blockIdx.x;
  if (group >= num_groups) {
    return;
  }

  int n = group_sizes[group];
  float *g = d_g[group];
  float *m = d_m[group];
  float *v = d_v[group];
  float *theta = d_theta[group];

  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g[idx];
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g[idx] * g[idx];

    float m_hat = m[idx] * inv_bias_correction1;
    float v_hat = v[idx] * inv_bias_correction2;
    theta[idx] -=
        alpha * (m_hat / (sqrtf(v_hat) + epsilon) + weight_decay * theta[idx]);
  }
}

__global__ void accumulateGradNormSqKernel(float *const *grads,
                                           const int *group_sizes,
                                           int num_groups,
                                           float *total_norm_sq) {
  extern __shared__ float sdata[];
  int group = blockIdx.x;
  float local = 0.0f;

  if (group < num_groups) {
    const float *group_grads = grads[group];
    int n = group_sizes[group];
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
      float v = group_grads[idx];
      local += v * v;
    }
  }

  sdata[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && group < num_groups) {
    atomicAdd(total_norm_sq, sdata[0]);
  }
}

__global__ void clipGradsMultiTensorKernel(float *const *grads,
                                           const int *group_sizes,
                                           int num_groups, float clip_coeff) {
  int group = blockIdx.x;
  if (group >= num_groups) {
    return;
  }
  float *group_grads = grads[group];
  int n = group_sizes[group];
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    group_grads[idx] *= clip_coeff;
  }
}

__global__ void clipGradsKernel(float *grads, float clip_coeff, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    grads[idx] *= clip_coeff;
  }
}
