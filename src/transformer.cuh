#pragma once

#include "activation.cuh"
#include "dropout.cuh"
#include "layers.cuh"
#include "sequential.cuh"
#include <random>

__global__ void attnScoresKernel(const float *Q, const float *K, float *scores,
                                 int seq_len, int d_model, float inv_scale) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < seq_len && j < seq_len) {
    float sum = 0.0f;
    for (int d = 0; d < d_model; d++) {
      sum += Q[i * d_model + d] * K[j * d_model + d];
    }
    scores[i * seq_len + j] = sum * inv_scale;
  }
}

__global__ void softmaxForwardKernel(float *scores, int seq_len) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < seq_len) {
    float *row_ptr = scores + row * seq_len;
    float max_val = row_ptr[0];
    for (int j = 1; j < seq_len; j++) {
      max_val = fmaxf(max_val, row_ptr[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      row_ptr[j] = expf(row_ptr[j] - max_val);
      sum += row_ptr[j];
    }
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < seq_len; j++) {
      row_ptr[j] *= inv_sum;
    }
  }
}

__global__ void attnApplyKernel(const float *scores, const float *V,
                                float *output, int seq_len, int d_model) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < seq_len && d < d_model) {
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      sum += scores[i * seq_len + j] * V[j * d_model + d];
    }
    output[i * d_model + d] = sum;
  }
}

__global__ void softmaxBackwardKernel(float *d_scores, const float *scores,
                                      int seq_len) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < seq_len) {
    float *ds_row = d_scores + row * seq_len;
    const float *s_row = scores + row * seq_len;
    float dot = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      dot += ds_row[j] * s_row[j];
    }
    for (int j = 0; j < seq_len; j++) {
      ds_row[j] = s_row[j] * (ds_row[j] - dot);
    }
  }
}

__global__ void matMulBTKernel(const float *A, const float *B, float *C,
                               int M, int K, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
  }
}

__global__ void matMulATKernel(const float *A, const float *B, float *C,
                               int M, int K, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < K && col < N) {
    float sum = 0.0f;
    for (int m = 0; m < M; m++) {
      sum += A[m * K + row] * B[m * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void layerNormForwardCachingKernel(
    const float *input, const float *gamma, const float *beta, float *output,
    float *x_hat, float *inv_std, int num_rows, int feature_size,
    float epsilon) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *in_row = input + row * feature_size;
    float *out_row = output + row * feature_size;
    float *xh_row = x_hat + row * feature_size;
    float mean = 0.0f;
    for (int d = 0; d < feature_size; d++) {
      mean += in_row[d];
    }
    mean /= feature_size;
    float var = 0.0f;
    for (int d = 0; d < feature_size; d++) {
      float diff = in_row[d] - mean;
      var += diff * diff;
    }
    var /= feature_size;
    float is = 1.0f / sqrtf(var + epsilon);
    inv_std[row] = is;
    for (int d = 0; d < feature_size; d++) {
      float xh = (in_row[d] - mean) * is;
      xh_row[d] = xh;
      out_row[d] = gamma[d] * xh + beta[d];
    }
  }
}

__global__ void layerNormBackwardKernel(const float *dy, const float *x_hat,
                                        const float *inv_std,
                                        const float *gamma, float *dx,
                                        int num_rows, int feature_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *dy_row = dy + row * feature_size;
    const float *xh_row = x_hat + row * feature_size;
    float *dx_row = dx + row * feature_size;
    float is = inv_std[row];
    float c1 = 0.0f;
    float c2 = 0.0f;
    for (int d = 0; d < feature_size; d++) {
      float dxh = dy_row[d] * gamma[d];
      c1 += dxh;
      c2 += dxh * xh_row[d];
    }
    c1 /= feature_size;
    c2 /= feature_size;
    for (int d = 0; d < feature_size; d++) {
      float dxh = dy_row[d] * gamma[d];
      dx_row[d] = (dxh - c1 - xh_row[d] * c2) * is;
    }
  }
}

__global__ void layerNormParamGradKernel(const float *dy, const float *x_hat,
                                         float *d_gamma, float *d_beta,
                                         int num_rows, int feature_size) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < feature_size) {
    float dg = 0.0f;
    float db = 0.0f;
    for (int i = 0; i < num_rows; i++) {
      float dy_val = dy[i * feature_size + d];
      dg += dy_val * x_hat[i * feature_size + d];
      db += dy_val;
    }
    d_gamma[d] = dg;
    d_beta[d] = db;
  }
}

__global__ void scaleBufferKernel(float *data, float scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

__global__ void transformerResidualAddKernel(const float *a, const float *b,
                                             float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

class TransformerLayer : public Operation {
private:
  int batch_size_, sequence_length_, d_model_, num_heads_;
  float dropout_prob_;

  float *d_q_weights_, *d_k_weights_, *d_v_weights_;
  float *d_ff_weights_, *d_ff_bias_;
  float *d_gamma1_, *d_beta1_, *d_gamma2_, *d_beta2_;

  float *d_input_cached_, *d_queries_, *d_keys_, *d_values_;
  float *d_attn_scores_, *d_attn_output_;
  float *d_x_hat1_, *d_inv_std1_, *d_ln1_output_;
  float *d_ff_pre_relu_, *d_ff_output_;
  float *d_x_hat2_, *d_inv_std2_;
  float *d_residual_buf_;

  float *d_q_weights_grad_, *d_k_weights_grad_, *d_v_weights_grad_;
  float *d_ff_weights_grad_, *d_ff_bias_grad_;
  float *d_gamma1_grad_, *d_beta1_grad_, *d_gamma2_grad_, *d_beta2_grad_;

  float *d_grad_buf_, *d_score_grad_;
  float *d_dQ_, *d_dK_, *d_dV_;

public:
  TransformerLayer(int batch_size, int sequence_length, int d_model,
                   int num_heads, float dropout_prob)
      : batch_size_(batch_size), sequence_length_(sequence_length),
        d_model_(d_model), num_heads_(num_heads), dropout_prob_(dropout_prob) {
    int ST = batch_size * sequence_length;
    int D = d_model;
    int w_size = D * D;
    int buf_size = ST * D;
    int score_size = batch_size * sequence_length * sequence_length;

    CUDA_CHECK(cudaMalloc(&d_q_weights_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_weights_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_weights_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_weights_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_bias_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta1_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta2_, D * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_input_cached_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keys_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_scores_, score_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_output_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_hat1_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std1_, ST * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_output_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_pre_relu_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_output_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_hat2_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std2_, ST * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_buf_, buf_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_q_weights_grad_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_weights_grad_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_weights_grad_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_weights_grad_, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_bias_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta1_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta2_grad_, D * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_grad_buf_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score_grad_, score_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV_, buf_size * sizeof(float)));

    std::mt19937 rng(42);
    float scale = sqrtf(2.0f / D);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> h_w(w_size);
    auto init_w = [&](float *d_ptr) {
      for (int i = 0; i < w_size; i++)
        h_w[i] = dist(rng);
      CUDA_CHECK(cudaMemcpy(d_ptr, h_w.data(), w_size * sizeof(float),
                             cudaMemcpyHostToDevice));
    };
    init_w(d_q_weights_);
    init_w(d_k_weights_);
    init_w(d_v_weights_);
    init_w(d_ff_weights_);

    std::vector<float> h_ones(D, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_gamma1_, h_ones.data(), D * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma2_, h_ones.data(), D * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beta1_, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta2_, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ff_bias_, 0, D * sizeof(float)));
  }

  ~TransformerLayer() {
    cudaFree(d_q_weights_);
    cudaFree(d_k_weights_);
    cudaFree(d_v_weights_);
    cudaFree(d_ff_weights_);
    cudaFree(d_ff_bias_);
    cudaFree(d_gamma1_);
    cudaFree(d_beta1_);
    cudaFree(d_gamma2_);
    cudaFree(d_beta2_);
    cudaFree(d_input_cached_);
    cudaFree(d_queries_);
    cudaFree(d_keys_);
    cudaFree(d_values_);
    cudaFree(d_attn_scores_);
    cudaFree(d_attn_output_);
    cudaFree(d_x_hat1_);
    cudaFree(d_inv_std1_);
    cudaFree(d_ln1_output_);
    cudaFree(d_ff_pre_relu_);
    cudaFree(d_ff_output_);
    cudaFree(d_x_hat2_);
    cudaFree(d_inv_std2_);
    cudaFree(d_residual_buf_);
    cudaFree(d_q_weights_grad_);
    cudaFree(d_k_weights_grad_);
    cudaFree(d_v_weights_grad_);
    cudaFree(d_ff_weights_grad_);
    cudaFree(d_ff_bias_grad_);
    cudaFree(d_gamma1_grad_);
    cudaFree(d_beta1_grad_);
    cudaFree(d_gamma2_grad_);
    cudaFree(d_beta2_grad_);
    cudaFree(d_grad_buf_);
    cudaFree(d_score_grad_);
    cudaFree(d_dQ_);
    cudaFree(d_dK_);
    cudaFree(d_dV_);
  }

  int input_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }
  int output_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }

  void forward(float *input, float *output) override {
    int ST = batch_size_ * sequence_length_;
    int D = d_model_;
    int S = sequence_length_;
    int total = ST * D;
    float inv_scale = 1.0f / sqrtf(static_cast<float>(D));

    CUDA_CHECK(cudaMemcpy(d_input_cached_, input, total * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    dim3 threads(16, 16);
    dim3 proj_blocks((D + 15) / 16, (ST + 15) / 16);

    linearLayerKernel<<<proj_blocks, threads>>>(input, d_q_weights_, nullptr,
                                                d_queries_, ST, D, D);
    linearLayerKernel<<<proj_blocks, threads>>>(input, d_k_weights_, nullptr,
                                                d_keys_, ST, D, D);
    linearLayerKernel<<<proj_blocks, threads>>>(input, d_v_weights_, nullptr,
                                                d_values_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int b = 0; b < batch_size_; b++) {
      int off_SD = b * S * D;
      int off_SS = b * S * S;

      dim3 score_blocks((S + 15) / 16, (S + 15) / 16);
      attnScoresKernel<<<score_blocks, threads>>>(
          d_queries_ + off_SD, d_keys_ + off_SD, d_attn_scores_ + off_SS, S,
          D, inv_scale);
      CUDA_CHECK(cudaDeviceSynchronize());

      softmaxForwardKernel<<<(S + 255) / 256, 256>>>(
          d_attn_scores_ + off_SS, S);
      CUDA_CHECK(cudaDeviceSynchronize());

      dim3 apply_blocks((D + 15) / 16, (S + 15) / 16);
      attnApplyKernel<<<apply_blocks, threads>>>(
          d_attn_scores_ + off_SS, d_values_ + off_SD,
          d_attn_output_ + off_SD, S, D);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    int elem_blocks = (total + 255) / 256;
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        input, d_attn_output_, d_residual_buf_, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    layerNormForwardCachingKernel<<<(ST + 255) / 256, 256>>>(
        d_residual_buf_, d_gamma1_, d_beta1_, d_ln1_output_, d_x_hat1_,
        d_inv_std1_, ST, D, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    linearLayerKernel<<<proj_blocks, threads>>>(
        d_ln1_output_, d_ff_weights_, d_ff_bias_, d_ff_pre_relu_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    reluKernel<<<elem_blocks, 256>>>(d_ff_pre_relu_, d_ff_output_, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_ln1_output_, d_ff_output_, d_residual_buf_, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    layerNormForwardCachingKernel<<<(ST + 255) / 256, 256>>>(
        d_residual_buf_, d_gamma2_, d_beta2_, output, d_x_hat2_, d_inv_std2_,
        ST, D, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int ST = batch_size_ * sequence_length_;
    int D = d_model_;
    int S = sequence_length_;
    int total = ST * D;
    float inv_scale = 1.0f / sqrtf(static_cast<float>(D));

    dim3 threads(16, 16);
    dim3 proj_blocks((D + 15) / 16, (ST + 15) / 16);
    dim3 w_blocks((D + 15) / 16, (D + 15) / 16);
    int elem_blocks = (total + 255) / 256;

    layerNormBackwardKernel<<<(ST + 255) / 256, 256>>>(
        d_output_grad, d_x_hat2_, d_inv_std2_, d_gamma2_, d_grad_buf_, ST, D);
    layerNormParamGradKernel<<<(D + 255) / 256, 256>>>(
        d_output_grad, d_x_hat2_, d_gamma2_grad_, d_beta2_grad_, ST, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    reluBackwardKernel<<<elem_blocks, 256>>>(d_grad_buf_, d_ff_pre_relu_,
                                             d_residual_buf_, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    linearBackwardInputKernel<<<proj_blocks, threads>>>(
        d_residual_buf_, d_ff_weights_, d_ff_output_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_grad_buf_, d_ff_output_, d_grad_buf_, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    linearBackwardWeightKernel<<<w_blocks, threads>>>(
        d_ln1_output_, d_residual_buf_, d_ff_weights_grad_, ST, D, D);
    linearBackwardBiasKernel<<<(D + 255) / 256, 256>>>(
        d_residual_buf_, d_ff_bias_grad_, ST, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    layerNormBackwardKernel<<<(ST + 255) / 256, 256>>>(
        d_grad_buf_, d_x_hat1_, d_inv_std1_, d_gamma1_, d_residual_buf_, ST,
        D);
    layerNormParamGradKernel<<<(D + 255) / 256, 256>>>(
        d_grad_buf_, d_x_hat1_, d_gamma1_grad_, d_beta1_grad_, ST, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_input_grad, d_residual_buf_,
                           total * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < batch_size_; b++) {
      int off_SD = b * S * D;
      int off_SS = b * S * S;

      dim3 score_blocks((S + 15) / 16, (S + 15) / 16);
      dim3 sd_blocks((D + 15) / 16, (S + 15) / 16);

      matMulBTKernel<<<score_blocks, threads>>>(
          d_residual_buf_ + off_SD, d_values_ + off_SD,
          d_score_grad_ + off_SS, S, D, S);
      CUDA_CHECK(cudaDeviceSynchronize());

      matMulATKernel<<<sd_blocks, threads>>>(
          d_attn_scores_ + off_SS, d_residual_buf_ + off_SD,
          d_dV_ + off_SD, S, S, D);
      CUDA_CHECK(cudaDeviceSynchronize());

      softmaxBackwardKernel<<<(S + 255) / 256, 256>>>(
          d_score_grad_ + off_SS, d_attn_scores_ + off_SS, S);
      CUDA_CHECK(cudaDeviceSynchronize());

      scaleBufferKernel<<<(S * S + 255) / 256, 256>>>(
          d_score_grad_ + off_SS, inv_scale, S * S);
      CUDA_CHECK(cudaDeviceSynchronize());

      linearLayerKernel<<<sd_blocks, threads>>>(
          d_score_grad_ + off_SS, d_keys_ + off_SD, nullptr,
          d_dQ_ + off_SD, S, S, D);
      CUDA_CHECK(cudaDeviceSynchronize());

      matMulATKernel<<<sd_blocks, threads>>>(
          d_score_grad_ + off_SS, d_queries_ + off_SD,
          d_dK_ + off_SD, S, S, D);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    linearBackwardInputKernel<<<proj_blocks, threads>>>(
        d_dQ_, d_q_weights_, d_grad_buf_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_input_grad, d_grad_buf_, d_input_grad, total);

    linearBackwardInputKernel<<<proj_blocks, threads>>>(
        d_dK_, d_k_weights_, d_grad_buf_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_input_grad, d_grad_buf_, d_input_grad, total);

    linearBackwardInputKernel<<<proj_blocks, threads>>>(
        d_dV_, d_v_weights_, d_grad_buf_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_input_grad, d_grad_buf_, d_input_grad, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    linearBackwardWeightKernel<<<w_blocks, threads>>>(
        d_input_cached_, d_dQ_, d_q_weights_grad_, ST, D, D);
    linearBackwardWeightKernel<<<w_blocks, threads>>>(
        d_input_cached_, d_dK_, d_k_weights_grad_, ST, D, D);
    linearBackwardWeightKernel<<<w_blocks, threads>>>(
        d_input_cached_, d_dV_, d_v_weights_grad_, ST, D, D);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void update_weights(float lr) override {
    int D = d_model_;
    int w_size = D * D;
    int w_blocks = (w_size + 255) / 256;
    int d_blocks = (D + 255) / 256;

    sgdUpdateKernel<<<w_blocks, 256>>>(d_q_weights_, d_q_weights_grad_, lr,
                                       w_size);
    sgdUpdateKernel<<<w_blocks, 256>>>(d_k_weights_, d_k_weights_grad_, lr,
                                       w_size);
    sgdUpdateKernel<<<w_blocks, 256>>>(d_v_weights_, d_v_weights_grad_, lr,
                                       w_size);
    sgdUpdateKernel<<<w_blocks, 256>>>(d_ff_weights_, d_ff_weights_grad_, lr,
                                       w_size);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_ff_bias_, d_ff_bias_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_gamma1_, d_gamma1_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_beta1_, d_beta1_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_gamma2_, d_gamma2_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_beta2_, d_beta2_grad_, lr, D);
  }
};
