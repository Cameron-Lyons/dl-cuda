#pragma once

#include "layers.cuh"
#include "sequential.cuh"
#include "transformer.cuh"
#include <vector>

class LayerNorm : public Operation {
private:
  float *d_gamma_, *d_beta_;
  float *d_gamma_grad_, *d_beta_grad_;
  float *d_x_hat_, *d_inv_std_;
  int num_rows_, feature_size_;
  float epsilon_;

public:
  LayerNorm(int num_rows, int feature_size, float epsilon = 1e-5f)
      : num_rows_(num_rows), feature_size_(feature_size), epsilon_(epsilon) {
    int total = num_rows * feature_size;
    CUDA_CHECK(cudaMalloc(&d_gamma_, feature_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_, feature_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma_grad_, feature_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_grad_, feature_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_hat_, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std_, num_rows * sizeof(float)));

    std::vector<float> ones(feature_size, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_gamma_, ones.data(), feature_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beta_, 0, feature_size * sizeof(float)));
  }

  ~LayerNorm() {
    cudaFree(d_gamma_);
    cudaFree(d_beta_);
    cudaFree(d_gamma_grad_);
    cudaFree(d_beta_grad_);
    cudaFree(d_x_hat_);
    cudaFree(d_inv_std_);
  }

  int input_size() const override { return num_rows_ * feature_size_; }
  int output_size() const override { return num_rows_ * feature_size_; }

  void forward(float *d_input, float *d_output) override {
    layerNormForwardCachingKernel<<<(num_rows_ + 255) / 256, 256>>>(
        d_input, d_gamma_, d_beta_, d_output, d_x_hat_, d_inv_std_, num_rows_,
        feature_size_, epsilon_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    layerNormBackwardKernel<<<(num_rows_ + 255) / 256, 256>>>(
        d_output_grad, d_x_hat_, d_inv_std_, d_gamma_, d_input_grad,
        num_rows_, feature_size_);
    layerNormParamGradKernel<<<(feature_size_ + 255) / 256, 256>>>(
        d_output_grad, d_x_hat_, d_gamma_grad_, d_beta_grad_, num_rows_,
        feature_size_);
  }

  void update_weights(float lr) override {
    int blocks = (feature_size_ + 255) / 256;
    sgdUpdateKernel<<<blocks, 256>>>(d_gamma_, d_gamma_grad_, lr,
                                     feature_size_);
    sgdUpdateKernel<<<blocks, 256>>>(d_beta_, d_beta_grad_, lr, feature_size_);
  }

  std::vector<ParamGroup> get_param_groups() override {
    return {{d_gamma_, d_gamma_grad_, feature_size_},
            {d_beta_, d_beta_grad_, feature_size_}};
  }
};
