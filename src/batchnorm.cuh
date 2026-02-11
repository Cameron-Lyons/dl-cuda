#pragma once

#include "sequential.cuh"
#include <vector>

__global__ void batchNormForwardKernel(const float *input, const float *gamma,
                                       const float *beta, const float *mean,
                                       const float *var, float *output,
                                       float *x_hat, int batch_size,
                                       int features, float epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * features) {
    int f = idx % features;
    float inv_std = 1.0f / sqrtf(var[f] + epsilon);
    float xh = (input[idx] - mean[f]) * inv_std;
    x_hat[idx] = xh;
    output[idx] = gamma[f] * xh + beta[f];
  }
}

__global__ void batchNormComputeMeanKernel(const float *input, float *mean,
                                           int batch_size, int features) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < features) {
    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
      sum += input[b * features + f];
    }
    mean[f] = sum / batch_size;
  }
}

__global__ void batchNormComputeVarKernel(const float *input, const float *mean,
                                          float *var, int batch_size,
                                          int features) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < features) {
    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
      float diff = input[b * features + f] - mean[f];
      sum += diff * diff;
    }
    var[f] = sum / batch_size;
  }
}

__global__ void updateRunningStatsKernel(float *running, const float *current,
                                         float momentum, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    running[idx] = (1.0f - momentum) * running[idx] + momentum * current[idx];
  }
}

__global__ void batchNormBackwardKernel(const float *d_output, const float *x_hat,
                                        const float *gamma, const float *var,
                                        float *d_input, int batch_size,
                                        int features, float epsilon) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < features) {
    float inv_std = 1.0f / sqrtf(var[f] + epsilon);
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int b = 0; b < batch_size; b++) {
      int idx = b * features + f;
      float dy = d_output[idx] * gamma[f];
      sum_dy += dy;
      sum_dy_xhat += dy * x_hat[idx];
    }
    float inv_n = 1.0f / batch_size;
    for (int b = 0; b < batch_size; b++) {
      int idx = b * features + f;
      float dy = d_output[idx] * gamma[f];
      d_input[idx] = (dy - inv_n * sum_dy - x_hat[idx] * inv_n * sum_dy_xhat) * inv_std;
    }
  }
}

__global__ void batchNormParamGradKernel(const float *d_output,
                                         const float *x_hat, float *d_gamma,
                                         float *d_beta, int batch_size,
                                         int features) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < features) {
    float dg = 0.0f;
    float db = 0.0f;
    for (int b = 0; b < batch_size; b++) {
      int idx = b * features + f;
      dg += d_output[idx] * x_hat[idx];
      db += d_output[idx];
    }
    d_gamma[f] = dg;
    d_beta[f] = db;
  }
}

class BatchNorm1D : public Operation {
private:
  float *d_gamma_, *d_beta_;
  float *d_gamma_grad_, *d_beta_grad_;
  float *d_running_mean_, *d_running_var_;
  float *d_batch_mean_, *d_batch_var_;
  float *d_x_hat_;
  int batch_size_, features_;
  float epsilon_, momentum_;
  bool training_;

public:
  BatchNorm1D(int batch_size, int features, float momentum = 0.1f,
              float epsilon = 1e-5f)
      : batch_size_(batch_size), features_(features), epsilon_(epsilon),
        momentum_(momentum), training_(true) {
    int total = batch_size * features;
    CUDA_CHECK(cudaMalloc(&d_gamma_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma_grad_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_grad_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_running_mean_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_running_var_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_mean_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_var_, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_hat_, total * sizeof(float)));

    std::vector<float> ones(features, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_gamma_, ones.data(), features * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beta_, 0, features * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_running_mean_, 0, features * sizeof(float)));
    std::vector<float> init_var(features, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_running_var_, init_var.data(),
                           features * sizeof(float), cudaMemcpyHostToDevice));
  }

  ~BatchNorm1D() {
    cudaFree(d_gamma_);
    cudaFree(d_beta_);
    cudaFree(d_gamma_grad_);
    cudaFree(d_beta_grad_);
    cudaFree(d_running_mean_);
    cudaFree(d_running_var_);
    cudaFree(d_batch_mean_);
    cudaFree(d_batch_var_);
    cudaFree(d_x_hat_);
  }

  void set_training(bool training) { training_ = training; }

  int input_size() const override { return batch_size_ * features_; }
  int output_size() const override { return batch_size_ * features_; }

  void forward(float *d_input, float *d_output) override {
    int f_blocks = (features_ + 255) / 256;
    int total = batch_size_ * features_;
    int t_blocks = (total + 255) / 256;

    if (training_) {
      batchNormComputeMeanKernel<<<f_blocks, 256>>>(d_input, d_batch_mean_,
                                                     batch_size_, features_);
      CUDA_CHECK(cudaDeviceSynchronize());

      batchNormComputeVarKernel<<<f_blocks, 256>>>(d_input, d_batch_mean_,
                                                    d_batch_var_, batch_size_,
                                                    features_);
      CUDA_CHECK(cudaDeviceSynchronize());

      updateRunningStatsKernel<<<f_blocks, 256>>>(d_running_mean_,
                                                    d_batch_mean_, momentum_,
                                                    features_);
      updateRunningStatsKernel<<<f_blocks, 256>>>(d_running_var_, d_batch_var_,
                                                    momentum_, features_);
      CUDA_CHECK(cudaDeviceSynchronize());

      batchNormForwardKernel<<<t_blocks, 256>>>(d_input, d_gamma_, d_beta_,
                                                 d_batch_mean_, d_batch_var_,
                                                 d_output, d_x_hat_,
                                                 batch_size_, features_,
                                                 epsilon_);
    } else {
      batchNormForwardKernel<<<t_blocks, 256>>>(d_input, d_gamma_, d_beta_,
                                                 d_running_mean_,
                                                 d_running_var_, d_output,
                                                 d_x_hat_, batch_size_,
                                                 features_, epsilon_);
    }
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int f_blocks = (features_ + 255) / 256;

    batchNormBackwardKernel<<<f_blocks, 256>>>(d_output_grad, d_x_hat_,
                                                d_gamma_, d_batch_var_,
                                                d_input_grad, batch_size_,
                                                features_, epsilon_);

    batchNormParamGradKernel<<<f_blocks, 256>>>(d_output_grad, d_x_hat_,
                                                 d_gamma_grad_, d_beta_grad_,
                                                 batch_size_, features_);
  }

  void update_weights(float lr) override {
    int blocks = (features_ + 255) / 256;
    sgdUpdateKernel<<<blocks, 256>>>(d_gamma_, d_gamma_grad_, lr, features_);
    sgdUpdateKernel<<<blocks, 256>>>(d_beta_, d_beta_grad_, lr, features_);
  }

  std::vector<ParamGroup> get_param_groups() override {
    return {{d_gamma_, d_gamma_grad_, features_},
            {d_beta_, d_beta_grad_, features_}};
  }
};
