#pragma once

#include "activation.cuh"
#include "layers.cuh"
#include "sequential.cuh"
#include <cfloat>
#include <random>

__global__ void maskedScaledSoftmaxFromQKBatchedKernel(
    const float *Q, const float *K, float *scores, int seq_len, int d_k,
    float inv_scale, int num_batches, int causal) {
  int batch = blockIdx.x;
  int row = blockIdx.y;
  int tid = threadIdx.x;
  if (batch >= num_batches || row >= seq_len) {
    return;
  }

  int q_row_offset = (batch * seq_len + row) * d_k;
  int score_row_offset = (batch * seq_len + row) * seq_len;

  float local_max = -FLT_MAX;
  for (int j = tid; j < seq_len; j += blockDim.x) {
    int k_row_offset = (batch * seq_len + j) * d_k;
    float sum = 0.0f;
    for (int d = 0; d < d_k; d++) {
      sum += Q[q_row_offset + d] * K[k_row_offset + d];
    }
    float val = sum * inv_scale;
    if (causal && j > row) {
      val = -1e9f;
    }
    scores[score_row_offset + j] = val;
    local_max = fmaxf(local_max, val);
  }

  extern __shared__ float sdata[];
  sdata[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }
  float row_max = sdata[0];

  float local_sum = 0.0f;
  for (int j = tid; j < seq_len; j += blockDim.x) {
    float e = expf(scores[score_row_offset + j] - row_max);
    scores[score_row_offset + j] = e;
    local_sum += e;
  }
  sdata[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  float inv_sum = 1.0f / (sdata[0] + 1e-20f);

  for (int j = tid; j < seq_len; j += blockDim.x) {
    scores[score_row_offset + j] *= inv_sum;
  }
}

__global__ void attnApplyBatchedKernel(const float *scores, const float *V,
                                       float *output, int seq_len, int d_v,
                                       int num_batches) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z;
  if (batch < num_batches && i < seq_len && d < d_v) {
    int score_offset = batch * seq_len * seq_len;
    int v_offset = batch * seq_len * d_v;
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      sum += scores[score_offset + i * seq_len + j] *
             V[v_offset + j * d_v + d];
    }
    output[v_offset + i * d_v + d] = sum;
  }
}

__global__ void softmaxScaleMaskBackwardBatchedKernel(
    float *d_scores, const float *scores, int seq_len, int num_batches,
    float inv_scale, int causal) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int total_rows = num_batches * seq_len;
  if (row < total_rows) {
    int row_in_seq = row % seq_len;
    float *ds_row = d_scores + row * seq_len;
    const float *s_row = scores + row * seq_len;
    float dot = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      float g = ds_row[j];
      if (causal && j > row_in_seq) {
        g = 0.0f;
      }
      ds_row[j] = g;
      dot += g * s_row[j];
    }
    for (int j = 0; j < seq_len; j++) {
      float out = s_row[j] * (ds_row[j] - dot) * inv_scale;
      if (causal && j > row_in_seq) {
        out = 0.0f;
      }
      ds_row[j] = out;
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

__global__ void matMulBTBatchedKernel(const float *A, const float *B, float *C,
                                      int M, int K, int N, int num_batches) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z;
  if (batch < num_batches && row < M && col < N) {
    int a_offset = batch * M * K;
    int b_offset = batch * N * K;
    int c_offset = batch * M * N;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[a_offset + row * K + k] * B[b_offset + col * K + k];
    }
    C[c_offset + row * N + col] = sum;
  }
}

__global__ void matMulATBatchedKernel(const float *A, const float *B, float *C,
                                      int M, int K, int N, int num_batches) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z;
  if (batch < num_batches && row < K && col < N) {
    int a_offset = batch * M * K;
    int b_offset = batch * M * N;
    int c_offset = batch * K * N;
    float sum = 0.0f;
    for (int m = 0; m < M; m++) {
      sum += A[a_offset + m * K + row] * B[b_offset + m * N + col];
    }
    C[c_offset + row * N + col] = sum;
  }
}

__global__ void reshapeToHeadsKernel(const float *input, float *output,
                                     int batch_size, int seq_len,
                                     int num_heads, int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * num_heads * seq_len * head_dim;
  if (idx < total) {
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    int d_model = num_heads * head_dim;
    int in_idx = (b * seq_len + s) * d_model + h * head_dim + d;
    output[idx] = input[in_idx];
  }
}

__global__ void reshapeFromHeadsKernel(const float *input, float *output,
                                       int batch_size, int seq_len,
                                       int num_heads, int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * num_heads * seq_len * head_dim;
  if (idx < total) {
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    int d_model = num_heads * head_dim;
    int out_idx = (b * seq_len + s) * d_model + h * head_dim + d;
    output[out_idx] = input[idx];
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

__global__ void transformerResidualAddKernel(const float *a, const float *b,
                                             float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

class TransformerLayer : public Operation {
private:
  int batch_size_, sequence_length_, d_model_, d_ff_, num_heads_, head_dim_;
  float dropout_prob_;
  bool causal_;

  float *d_q_weights_, *d_k_weights_, *d_v_weights_, *d_wo_weights_;
  float *d_ff1_weights_, *d_ff1_bias_;
  float *d_ff2_weights_, *d_ff2_bias_;
  float *d_gamma1_, *d_beta1_, *d_gamma2_, *d_beta2_;

  float *d_input_cached_, *d_queries_, *d_keys_, *d_values_;
  float *d_q_heads_, *d_k_heads_, *d_v_heads_;
  float *d_attn_scores_, *d_attn_heads_out_, *d_attn_concat_;
  float *d_attn_output_;
  float *d_x_hat1_, *d_inv_std1_, *d_ln1_output_;
  float *d_ff_pre_act_, *d_ff_post_act_, *d_ff_output_;
  float *d_x_hat2_, *d_inv_std2_;
  float *d_residual_buf_;

  float *d_q_weights_grad_, *d_k_weights_grad_, *d_v_weights_grad_;
  float *d_wo_weights_grad_;
  float *d_ff1_weights_grad_, *d_ff1_bias_grad_;
  float *d_ff2_weights_grad_, *d_ff2_bias_grad_;
  float *d_gamma1_grad_, *d_beta1_grad_, *d_gamma2_grad_, *d_beta2_grad_;

  float *d_grad_buf_, *d_ff_grad_buf_, *d_score_grad_;
  float *d_dQ_, *d_dK_, *d_dV_;
  float *d_dQ_heads_, *d_dK_heads_, *d_dV_heads_;
  float *d_d_concat_;

public:
  TransformerLayer(int batch_size, int sequence_length, int d_model,
                   int num_heads, float dropout_prob, int d_ff = 0)
      : batch_size_(batch_size), sequence_length_(sequence_length),
        d_model_(d_model), d_ff_(d_ff > 0 ? d_ff : 4 * d_model),
        num_heads_(num_heads), head_dim_(d_model / num_heads),
        dropout_prob_(dropout_prob), causal_(false) {
    int ST = batch_size * sequence_length;
    int D = d_model;
    int F = d_ff_;
    int H = num_heads;
    int hd = head_dim_;
    int attn_w_size = D * D;
    int buf_size = ST * D;
    int ff_buf_size = ST * F;
    int head_buf_size = batch_size * H * sequence_length * hd;
    int score_size = batch_size * H * sequence_length * sequence_length;

    CUDA_CHECK(cudaMalloc(&d_q_weights_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_weights_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_weights_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wo_weights_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff1_weights_, D * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff1_bias_, F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff2_weights_, F * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff2_bias_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta1_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta2_, D * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_input_cached_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keys_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_heads_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_heads_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_heads_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_scores_, score_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_heads_out_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_concat_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_output_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_hat1_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std1_, ST * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_output_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_pre_act_, ff_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_post_act_, ff_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_output_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_hat2_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std2_, ST * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_buf_, buf_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_q_weights_grad_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_weights_grad_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_weights_grad_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wo_weights_grad_, attn_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff1_weights_grad_, D * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff1_bias_grad_, F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff2_weights_grad_, F * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff2_bias_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta1_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2_grad_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta2_grad_, D * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_grad_buf_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_grad_buf_, ff_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score_grad_, score_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV_, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ_heads_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK_heads_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV_heads_, head_buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_concat_, buf_size * sizeof(float)));

    std::mt19937 rng(42);
    auto init_w = [&](float *d_ptr, int size, float scale) {
      std::normal_distribution<float> dist(0.0f, scale);
      std::vector<float> h_w(size);
      for (int i = 0; i < size; i++)
        h_w[i] = dist(rng);
      CUDA_CHECK(cudaMemcpy(d_ptr, h_w.data(), size * sizeof(float),
                             cudaMemcpyHostToDevice));
    };
    float attn_scale = sqrtf(2.0f / D);
    init_w(d_q_weights_, attn_w_size, attn_scale);
    init_w(d_k_weights_, attn_w_size, attn_scale);
    init_w(d_v_weights_, attn_w_size, attn_scale);
    init_w(d_wo_weights_, attn_w_size, attn_scale);
    init_w(d_ff1_weights_, D * F, sqrtf(2.0f / D));
    init_w(d_ff2_weights_, F * D, sqrtf(2.0f / F));

    std::vector<float> h_ones(D, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_gamma1_, h_ones.data(), D * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma2_, h_ones.data(), D * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beta1_, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta2_, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ff1_bias_, 0, F * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ff2_bias_, 0, D * sizeof(float)));
  }

  ~TransformerLayer() {
    cudaFree(d_q_weights_);
    cudaFree(d_k_weights_);
    cudaFree(d_v_weights_);
    cudaFree(d_wo_weights_);
    cudaFree(d_ff1_weights_);
    cudaFree(d_ff1_bias_);
    cudaFree(d_ff2_weights_);
    cudaFree(d_ff2_bias_);
    cudaFree(d_gamma1_);
    cudaFree(d_beta1_);
    cudaFree(d_gamma2_);
    cudaFree(d_beta2_);
    cudaFree(d_input_cached_);
    cudaFree(d_queries_);
    cudaFree(d_keys_);
    cudaFree(d_values_);
    cudaFree(d_q_heads_);
    cudaFree(d_k_heads_);
    cudaFree(d_v_heads_);
    cudaFree(d_attn_scores_);
    cudaFree(d_attn_heads_out_);
    cudaFree(d_attn_concat_);
    cudaFree(d_attn_output_);
    cudaFree(d_x_hat1_);
    cudaFree(d_inv_std1_);
    cudaFree(d_ln1_output_);
    cudaFree(d_ff_pre_act_);
    cudaFree(d_ff_post_act_);
    cudaFree(d_ff_output_);
    cudaFree(d_x_hat2_);
    cudaFree(d_inv_std2_);
    cudaFree(d_residual_buf_);
    cudaFree(d_q_weights_grad_);
    cudaFree(d_k_weights_grad_);
    cudaFree(d_v_weights_grad_);
    cudaFree(d_wo_weights_grad_);
    cudaFree(d_ff1_weights_grad_);
    cudaFree(d_ff1_bias_grad_);
    cudaFree(d_ff2_weights_grad_);
    cudaFree(d_ff2_bias_grad_);
    cudaFree(d_gamma1_grad_);
    cudaFree(d_beta1_grad_);
    cudaFree(d_gamma2_grad_);
    cudaFree(d_beta2_grad_);
    cudaFree(d_grad_buf_);
    cudaFree(d_ff_grad_buf_);
    cudaFree(d_score_grad_);
    cudaFree(d_dQ_);
    cudaFree(d_dK_);
    cudaFree(d_dV_);
    cudaFree(d_dQ_heads_);
    cudaFree(d_dK_heads_);
    cudaFree(d_dV_heads_);
    cudaFree(d_d_concat_);
  }

  void set_causal(bool causal) { causal_ = causal; }

  int input_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }
  int output_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }

  void forward(float *input, float *output) override {
    int B = batch_size_;
    int S = sequence_length_;
    int D = d_model_;
    int F = d_ff_;
    int H = num_heads_;
    int hd = head_dim_;
    int ST = B * S;
    int BH = B * H;
    int total = ST * D;
    int ff_total = ST * F;
    int head_total = BH * S * hd;
    float inv_scale = 1.0f / sqrtf(static_cast<float>(hd));

    CUDA_CHECK(cudaMemcpyAsync(d_input_cached_, input, total * sizeof(float),
                               cudaMemcpyDeviceToDevice, 0));

    dim3 threads(16, 16);

    linearForward(input, d_q_weights_, nullptr, d_queries_, ST, D, D);
    linearForward(input, d_k_weights_, nullptr, d_keys_, ST, D, D);
    linearForward(input, d_v_weights_, nullptr, d_values_, ST, D, D);
    CUDA_CHECK(cudaGetLastError());

    int reshape_blocks = (head_total + 255) / 256;
    reshapeToHeadsKernel<<<reshape_blocks, 256>>>(d_queries_, d_q_heads_, B, S,
                                                   H, hd);
    reshapeToHeadsKernel<<<reshape_blocks, 256>>>(d_keys_, d_k_heads_, B, S, H,
                                                   hd);
    reshapeToHeadsKernel<<<reshape_blocks, 256>>>(d_values_, d_v_heads_, B, S,
                                                   H, hd);
    CUDA_CHECK(cudaGetLastError());

    dim3 softmax_blocks(BH, S);
    maskedScaledSoftmaxFromQKBatchedKernel<<<softmax_blocks, 128,
                                             128 * sizeof(float)>>>(
        d_q_heads_, d_k_heads_, d_attn_scores_, S, hd, inv_scale, BH,
        causal_ ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    dim3 apply_blocks((hd + 15) / 16, (S + 15) / 16, BH);
    attnApplyBatchedKernel<<<apply_blocks, threads>>>(
        d_attn_scores_, d_v_heads_, d_attn_heads_out_, S, hd, BH);
    CUDA_CHECK(cudaGetLastError());

    reshapeFromHeadsKernel<<<reshape_blocks, 256>>>(d_attn_heads_out_,
                                                     d_attn_concat_, B, S, H,
                                                     hd);
    CUDA_CHECK(cudaGetLastError());

    linearForward(d_attn_concat_, d_wo_weights_, nullptr, d_attn_output_, ST, D,
                  D);
    CUDA_CHECK(cudaGetLastError());

    int elem_blocks = (total + 255) / 256;
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        input, d_attn_output_, d_residual_buf_, total);
    CUDA_CHECK(cudaGetLastError());

    layerNormForwardCachingKernel<<<(ST + 255) / 256, 256>>>(
        d_residual_buf_, d_gamma1_, d_beta1_, d_ln1_output_, d_x_hat1_,
        d_inv_std1_, ST, D, 1e-5f);
    CUDA_CHECK(cudaGetLastError());

    linearForward(d_ln1_output_, d_ff1_weights_, d_ff1_bias_, d_ff_pre_act_, ST,
                  D, F);
    CUDA_CHECK(cudaGetLastError());

    int ff_elem_blocks = (ff_total + 255) / 256;
    geluForwardKernel<<<ff_elem_blocks, 256>>>(d_ff_pre_act_, d_ff_post_act_,
                                                ff_total);
    CUDA_CHECK(cudaGetLastError());

    linearForward(d_ff_post_act_, d_ff2_weights_, d_ff2_bias_, d_ff_output_, ST,
                  F, D);
    CUDA_CHECK(cudaGetLastError());

    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_ln1_output_, d_ff_output_, d_residual_buf_, total);
    CUDA_CHECK(cudaGetLastError());

    layerNormForwardCachingKernel<<<(ST + 255) / 256, 256>>>(
        d_residual_buf_, d_gamma2_, d_beta2_, output, d_x_hat2_, d_inv_std2_,
        ST, D, 1e-5f);
    CUDA_CHECK(cudaGetLastError());
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int B = batch_size_;
    int S = sequence_length_;
    int D = d_model_;
    int F = d_ff_;
    int H = num_heads_;
    int hd = head_dim_;
    int ST = B * S;
    int BH = B * H;
    int total = ST * D;
    int ff_total = ST * F;
    int head_total = BH * S * hd;
    float inv_scale = 1.0f / sqrtf(static_cast<float>(hd));

    dim3 threads(16, 16);
    int elem_blocks = (total + 255) / 256;
    int ff_elem_blocks = (ff_total + 255) / 256;
    int reshape_blocks = (head_total + 255) / 256;

    layerNormBackwardKernel<<<(ST + 255) / 256, 256>>>(
        d_output_grad, d_x_hat2_, d_inv_std2_, d_gamma2_, d_grad_buf_, ST, D);
    layerNormParamGradKernel<<<(D + 255) / 256, 256>>>(
        d_output_grad, d_x_hat2_, d_gamma2_grad_, d_beta2_grad_, ST, D);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardInput(d_grad_buf_, d_ff2_weights_, d_ff_grad_buf_, ST, F, D);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardWeight(d_ff_post_act_, d_grad_buf_, d_ff2_weights_grad_, ST,
                         F, D);
    linearBackwardBiasKernel<<<(D + 255) / 256, 256>>>(
        d_grad_buf_, d_ff2_bias_grad_, ST, D);
    CUDA_CHECK(cudaGetLastError());

    geluBackwardKernel<<<ff_elem_blocks, 256>>>(d_ff_grad_buf_, d_ff_pre_act_,
                                                 d_ff_grad_buf_, ff_total);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardInput(d_ff_grad_buf_, d_ff1_weights_, d_ff_output_, ST, D, F);
    CUDA_CHECK(cudaGetLastError());

    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_grad_buf_, d_ff_output_, d_grad_buf_, total);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardWeight(d_ln1_output_, d_ff_grad_buf_, d_ff1_weights_grad_, ST,
                         D, F);
    linearBackwardBiasKernel<<<(F + 255) / 256, 256>>>(
        d_ff_grad_buf_, d_ff1_bias_grad_, ST, F);
    CUDA_CHECK(cudaGetLastError());

    layerNormBackwardKernel<<<(ST + 255) / 256, 256>>>(
        d_grad_buf_, d_x_hat1_, d_inv_std1_, d_gamma1_, d_residual_buf_, ST,
        D);
    layerNormParamGradKernel<<<(D + 255) / 256, 256>>>(
        d_grad_buf_, d_x_hat1_, d_gamma1_grad_, d_beta1_grad_, ST, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(d_input_grad, d_residual_buf_,
                               total * sizeof(float),
                               cudaMemcpyDeviceToDevice, 0));

    linearBackwardInput(d_residual_buf_, d_wo_weights_, d_d_concat_, ST, D, D);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardWeight(d_attn_concat_, d_residual_buf_, d_wo_weights_grad_, ST,
                         D, D);
    CUDA_CHECK(cudaGetLastError());

    reshapeToHeadsKernel<<<reshape_blocks, 256>>>(d_d_concat_, d_dQ_heads_, B,
                                                   S, H, hd);
    CUDA_CHECK(cudaGetLastError());

    dim3 score_blocks((S + 15) / 16, (S + 15) / 16, BH);
    dim3 sd_blocks((hd + 15) / 16, (S + 15) / 16, BH);

    matMulBTBatchedKernel<<<score_blocks, threads>>>(
        d_dQ_heads_, d_v_heads_, d_score_grad_, S, hd, S, BH);
    CUDA_CHECK(cudaGetLastError());

    matMulATBatchedKernel<<<sd_blocks, threads>>>(
        d_attn_scores_, d_dQ_heads_, d_dV_heads_, S, S, hd, BH);
    CUDA_CHECK(cudaGetLastError());

    softmaxScaleMaskBackwardBatchedKernel<<<(BH * S + 255) / 256, 256>>>(
        d_score_grad_, d_attn_scores_, S, BH, inv_scale, causal_ ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    attnApplyBatchedKernel<<<sd_blocks, threads>>>(
        d_score_grad_, d_k_heads_, d_dQ_heads_, S, hd, BH);
    CUDA_CHECK(cudaGetLastError());

    matMulATBatchedKernel<<<sd_blocks, threads>>>(
        d_score_grad_, d_q_heads_, d_dK_heads_, S, S, hd, BH);
    CUDA_CHECK(cudaGetLastError());

    reshapeFromHeadsKernel<<<reshape_blocks, 256>>>(d_dQ_heads_, d_dQ_, B, S,
                                                     H, hd);
    reshapeFromHeadsKernel<<<reshape_blocks, 256>>>(d_dK_heads_, d_dK_, B, S,
                                                     H, hd);
    reshapeFromHeadsKernel<<<reshape_blocks, 256>>>(d_dV_heads_, d_dV_, B, S,
                                                     H, hd);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardInput(d_dQ_, d_q_weights_, d_grad_buf_, ST, D, D);
    CUDA_CHECK(cudaGetLastError());
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_input_grad, d_grad_buf_, d_input_grad, total);

    linearBackwardInput(d_dK_, d_k_weights_, d_grad_buf_, ST, D, D);
    CUDA_CHECK(cudaGetLastError());
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_input_grad, d_grad_buf_, d_input_grad, total);

    linearBackwardInput(d_dV_, d_v_weights_, d_grad_buf_, ST, D, D);
    CUDA_CHECK(cudaGetLastError());
    transformerResidualAddKernel<<<elem_blocks, 256>>>(
        d_input_grad, d_grad_buf_, d_input_grad, total);
    CUDA_CHECK(cudaGetLastError());

    linearBackwardWeight(d_input_cached_, d_dQ_, d_q_weights_grad_, ST, D, D);
    linearBackwardWeight(d_input_cached_, d_dK_, d_k_weights_grad_, ST, D, D);
    linearBackwardWeight(d_input_cached_, d_dV_, d_v_weights_grad_, ST, D, D);
    CUDA_CHECK(cudaGetLastError());
  }

  void update_weights(float lr) override {
    int D = d_model_;
    int F = d_ff_;
    int attn_w_size = D * D;
    int aw_blocks = (attn_w_size + 255) / 256;
    int ff1_blocks = (D * F + 255) / 256;
    int ff2_blocks = (F * D + 255) / 256;
    int d_blocks = (D + 255) / 256;
    int f_blocks = (F + 255) / 256;

    sgdUpdateKernel<<<aw_blocks, 256>>>(d_q_weights_, d_q_weights_grad_, lr,
                                        attn_w_size);
    sgdUpdateKernel<<<aw_blocks, 256>>>(d_k_weights_, d_k_weights_grad_, lr,
                                        attn_w_size);
    sgdUpdateKernel<<<aw_blocks, 256>>>(d_v_weights_, d_v_weights_grad_, lr,
                                        attn_w_size);
    sgdUpdateKernel<<<aw_blocks, 256>>>(d_wo_weights_, d_wo_weights_grad_, lr,
                                        attn_w_size);
    sgdUpdateKernel<<<ff1_blocks, 256>>>(d_ff1_weights_, d_ff1_weights_grad_,
                                         lr, D * F);
    sgdUpdateKernel<<<f_blocks, 256>>>(d_ff1_bias_, d_ff1_bias_grad_, lr, F);
    sgdUpdateKernel<<<ff2_blocks, 256>>>(d_ff2_weights_, d_ff2_weights_grad_,
                                         lr, F * D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_ff2_bias_, d_ff2_bias_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_gamma1_, d_gamma1_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_beta1_, d_beta1_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_gamma2_, d_gamma2_grad_, lr, D);
    sgdUpdateKernel<<<d_blocks, 256>>>(d_beta2_, d_beta2_grad_, lr, D);
  }

  std::vector<ParamGroup> get_param_groups() override {
    int D = d_model_;
    int F = d_ff_;
    int attn_w_size = D * D;
    return {{d_q_weights_, d_q_weights_grad_, attn_w_size},
            {d_k_weights_, d_k_weights_grad_, attn_w_size},
            {d_v_weights_, d_v_weights_grad_, attn_w_size},
            {d_wo_weights_, d_wo_weights_grad_, attn_w_size},
            {d_ff1_weights_, d_ff1_weights_grad_, D * F},
            {d_ff1_bias_, d_ff1_bias_grad_, F},
            {d_ff2_weights_, d_ff2_weights_grad_, F * D},
            {d_ff2_bias_, d_ff2_bias_grad_, D},
            {d_gamma1_, d_gamma1_grad_, D},
            {d_beta1_, d_beta1_grad_, D},
            {d_gamma2_, d_gamma2_grad_, D},
            {d_beta2_, d_beta2_grad_, D}};
  }
};

class TransformerStack : public Operation {
private:
  std::vector<TransformerLayer *> layers_;
  std::vector<float *> intermediates_;
  std::vector<float *> backward_intermediates_;
  int batch_size_, sequence_length_, d_model_;
  int num_layers_;

public:
  TransformerStack(int num_layers, int batch_size, int sequence_length,
                   int d_model, int num_heads, float dropout_prob,
                   int d_ff = 0, bool causal = false)
      : batch_size_(batch_size), sequence_length_(sequence_length),
        d_model_(d_model), num_layers_(num_layers) {
    int buf_size = batch_size * sequence_length * d_model;
    for (int i = 0; i < num_layers; i++) {
      auto *layer = new TransformerLayer(batch_size, sequence_length, d_model,
                                         num_heads, dropout_prob, d_ff);
      if (causal)
        layer->set_causal(true);
      layers_.push_back(layer);
    }
    for (int i = 0; i < num_layers - 1; i++) {
      float *forward_buf;
      float *backward_buf;
      CUDA_CHECK(cudaMalloc(&forward_buf, buf_size * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&backward_buf, buf_size * sizeof(float)));
      intermediates_.push_back(forward_buf);
      backward_intermediates_.push_back(backward_buf);
    }
  }

  ~TransformerStack() {
    for (auto *layer : layers_)
      delete layer;
    for (auto *buf : intermediates_)
      cudaFree(buf);
    for (auto *buf : backward_intermediates_)
      cudaFree(buf);
  }

  int input_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }
  int output_size() const override {
    return batch_size_ * sequence_length_ * d_model_;
  }

  void forward(float *input, float *output) override {
    if (num_layers_ == 1) {
      layers_[0]->forward(input, output);
      return;
    }
    layers_[0]->forward(input, intermediates_[0]);
    for (int i = 1; i < num_layers_ - 1; i++) {
      layers_[i]->forward(intermediates_[i - 1], intermediates_[i]);
    }
    layers_[num_layers_ - 1]->forward(intermediates_[num_layers_ - 2], output);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    if (num_layers_ == 1) {
      layers_[0]->backward(d_output_grad, d_input_grad);
      return;
    }
    layers_[num_layers_ - 1]->backward(d_output_grad,
                                        backward_intermediates_[num_layers_ - 2]);
    for (int i = num_layers_ - 2; i >= 1; i--) {
      layers_[i]->backward(backward_intermediates_[i],
                           backward_intermediates_[i - 1]);
    }
    layers_[0]->backward(backward_intermediates_[0], d_input_grad);
  }

  void update_weights(float lr) override {
    for (auto *layer : layers_)
      layer->update_weights(lr);
  }

  std::vector<ParamGroup> get_param_groups() override {
    std::vector<ParamGroup> all;
    for (auto *layer : layers_) {
      auto groups = layer->get_param_groups();
      all.insert(all.end(), groups.begin(), groups.end());
    }
    return all;
  }
};
