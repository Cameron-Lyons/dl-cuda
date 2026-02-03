#pragma once

#include "sequential.cuh"
#include <cmath>
#include <curand_kernel.h>

static const short NUM_THREADS = 256;

__global__ void linearLayerKernel(const float *d_X, const float *d_W,
                                  const float *d_b, float *d_Y, int n,
                                  int in_features, int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < out_features && row < n) {
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
      sum += d_X[row * in_features + i] * d_W[i * out_features + col];
    }
    d_Y[row * out_features + col] = sum + (d_b ? d_b[col] : 0.0f);
  }
}

__global__ void linearBackwardInputKernel(const float *d_output_grad,
                                          const float *d_W,
                                          float *d_input_grad, int n,
                                          int in_features, int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < in_features && row < n) {
    float sum = 0.0f;
    for (int i = 0; i < out_features; ++i) {
      sum += d_output_grad[row * out_features + i] * d_W[col * out_features + i];
    }
    d_input_grad[row * in_features + col] = sum;
  }
}

__global__ void linearBackwardWeightKernel(const float *d_X,
                                           const float *d_output_grad,
                                           float *d_W_grad, int n,
                                           int in_features, int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < in_features && col < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += d_X[i * in_features + row] * d_output_grad[i * out_features + col];
    }
    d_W_grad[row * out_features + col] = sum;
  }
}

__global__ void linearBackwardBiasKernel(const float *d_output_grad,
                                         float *d_b_grad, int n,
                                         int out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += d_output_grad[i * out_features + col];
    }
    d_b_grad[col] = sum;
  }
}

__global__ void sgdUpdateKernel(float *params, const float *grads, float lr,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    params[idx] -= lr * grads[idx];
  }
}

__global__ void initWeightsKernel(float *data, int n, float scale,
                                  curandState_t *states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    curand_init(42 + idx, idx, 0, &states[idx]);
    data[idx] = curand_normal(&states[idx]) * scale;
  }
}

class LinearLayer : public Operation {
private:
  float *d_W, *d_b;
  float *d_W_grad, *d_b_grad;
  float *d_cached_input;
  int n_, in_features_, out_features_;

public:
  LinearLayer(int n, int in_features, int out_features)
      : n_(n), in_features_(in_features), out_features_(out_features) {
    CUDA_CHECK(
        cudaMalloc(&d_W, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, out_features * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_W_grad, in_features * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_grad, out_features * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_cached_input, n * in_features * sizeof(float)));

    curandState_t *d_states;
    int total_w = in_features * out_features;
    CUDA_CHECK(cudaMalloc(&d_states, total_w * sizeof(curandState_t)));
    float scale = sqrtf(2.0f / in_features);
    int blocks = (total_w + 255) / 256;
    initWeightsKernel<<<blocks, 256>>>(d_W, total_w, scale, d_states);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_states);

    CUDA_CHECK(cudaMemset(d_b, 0, out_features * sizeof(float)));
  }

  ~LinearLayer() {
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_W_grad);
    cudaFree(d_b_grad);
    cudaFree(d_cached_input);
  }

  int input_size() const override { return n_ * in_features_; }
  int output_size() const override { return n_ * out_features_; }

  float *get_weights() { return d_W; }
  float *get_bias() { return d_b; }
  float *get_weight_grad() { return d_W_grad; }
  float *get_bias_grad() { return d_b_grad; }
  int get_in_features() const { return in_features_; }
  int get_out_features() const { return out_features_; }
  int get_n() const { return n_; }

  void forward(float *d_input, float *d_output) override {
    CUDA_CHECK(cudaMemcpy(d_cached_input, d_input,
                           n_ * in_features_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((out_features_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (n_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linearLayerKernel<<<blocks, threadsPerBlock>>>(d_input, d_W, d_b, d_output,
                                                   n_, in_features_,
                                                   out_features_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    dim3 threadsPerBlock(16, 16);

    dim3 blocks_input(
        (in_features_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n_ + threadsPerBlock.y - 1) / threadsPerBlock.y);
    linearBackwardInputKernel<<<blocks_input, threadsPerBlock>>>(
        d_output_grad, d_W, d_input_grad, n_, in_features_, out_features_);

    dim3 blocks_weight(
        (out_features_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (in_features_ + threadsPerBlock.y - 1) / threadsPerBlock.y);
    linearBackwardWeightKernel<<<blocks_weight, threadsPerBlock>>>(
        d_cached_input, d_output_grad, d_W_grad, n_, in_features_,
        out_features_);

    int blocks_bias = (out_features_ + 255) / 256;
    linearBackwardBiasKernel<<<blocks_bias, 256>>>(d_output_grad, d_b_grad, n_,
                                                   out_features_);
  }

  void update_weights(float lr) override {
    int w_size = in_features_ * out_features_;
    int blocks_w = (w_size + 255) / 256;
    sgdUpdateKernel<<<blocks_w, 256>>>(d_W, d_W_grad, lr, w_size);

    int blocks_b = (out_features_ + 255) / 256;
    sgdUpdateKernel<<<blocks_b, 256>>>(d_b, d_b_grad, lr, out_features_);
  }
};

__global__ void transposeMatVecKernel(const float *d_out, const float *W,
                                     float *d_in, int batch_size, int in_size,
                                     int out_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * in_size)
    return;
  int batch = idx / in_size;
  int i = idx % in_size;

  float sum = 0.0f;
  for (int j = 0; j < out_size; j++) {
    sum += W[i * out_size + j] * d_out[batch * out_size + j];
  }
  d_in[idx] = sum;
}

__global__ void weightGradAccumKernel(const float *d_out, const float *input,
                                      float *dW, int batch_size, int in_size,
                                      int out_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_size * out_size)
    return;
  int i = idx / out_size;
  int j = idx % out_size;

  float sum = 0.0f;
  for (int b = 0; b < batch_size; b++) {
    sum += d_out[b * out_size + j] * input[b * in_size + i];
  }
  dW[idx] += sum;
}

__global__ void biasGradAccumKernel(const float *d, float *db, int batch_size,
                                    int size) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= size)
    return;

  float sum = 0.0f;
  for (int b = 0; b < batch_size; b++) {
    sum += d[b * size + j];
  }
  db[j] += sum;
}

__global__ void tanhBackwardElementKernel(const float *dh, const float *h,
                                          float *d_pre, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float hv = h[idx];
    d_pre[idx] = dh[idx] * (1.0f - hv * hv);
  }
}

__global__ void addVectorsKernel(const float *a, const float *b, float *out,
                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void lstmForwardCachingKernel(
    const float *x, const float *Wf, const float *Wi, const float *Wc,
    const float *Wo, const float *bf, const float *bi, const float *bc,
    const float *bo, float *h, float *c, float *cache_f, float *cache_i,
    float *cache_c_hat, float *cache_o, int hidden_size, int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = idx / hidden_size;
  int hid = idx % hidden_size;

  if (batch < batch_size && hid < hidden_size) {
    int concat_size = 2 * hidden_size;
    int out_idx = batch * hidden_size + hid;

    float fg = 0.0f, ig = 0.0f, cg = 0.0f, og = 0.0f;

    for (int k = 0; k < concat_size; k++) {
      float val;
      if (k < hidden_size) {
        val = x[batch * hidden_size + k];
      } else {
        val = h[batch * hidden_size + (k - hidden_size)];
      }
      fg += Wf[k * hidden_size + hid] * val;
      ig += Wi[k * hidden_size + hid] * val;
      cg += Wc[k * hidden_size + hid] * val;
      og += Wo[k * hidden_size + hid] * val;
    }

    fg = 1.0f / (1.0f + expf(-(fg + bf[hid])));
    ig = 1.0f / (1.0f + expf(-(ig + bi[hid])));
    cg = tanhf(cg + bc[hid]);
    og = 1.0f / (1.0f + expf(-(og + bo[hid])));

    cache_f[out_idx] = fg;
    cache_i[out_idx] = ig;
    cache_c_hat[out_idx] = cg;
    cache_o[out_idx] = og;

    c[out_idx] = fg * c[out_idx] + ig * cg;
    h[out_idx] = og * tanhf(c[out_idx]);
  }
}

__global__ void lstmBackwardGatesKernel(
    const float *dh_t, const float *dh_next, const float *dc_next,
    const float *c_prev, const float *c_cur, const float *gate_f,
    const float *gate_i, const float *gate_c_hat, const float *gate_o,
    float *dz_f, float *dz_i, float *dz_c, float *dz_o, float *dc_prev_out,
    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float dh = dh_t[idx] + dh_next[idx];
  float tanh_c = tanhf(c_cur[idx]);
  float o = gate_o[idx];
  float f = gate_f[idx];
  float i_g = gate_i[idx];
  float c_hat = gate_c_hat[idx];

  float do_val = dh * tanh_c;
  float dc = dh * o * (1.0f - tanh_c * tanh_c) + dc_next[idx];

  dz_f[idx] = dc * c_prev[idx] * f * (1.0f - f);
  dz_i[idx] = dc * c_hat * i_g * (1.0f - i_g);
  dz_c[idx] = dc * i_g * (1.0f - c_hat * c_hat);
  dz_o[idx] = do_val * o * (1.0f - o);

  dc_prev_out[idx] = dc * f;
}

__global__ void lstmBackwardConcatKernel(
    const float *dz_f, const float *dz_i, const float *dz_c,
    const float *dz_o, const float *Wf, const float *Wi, const float *Wc,
    const float *Wo, float *dx, float *dh_prev, int batch_size,
    int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * 2 * hidden_size;
  if (idx >= total)
    return;

  int batch = idx / (2 * hidden_size);
  int k = idx % (2 * hidden_size);

  float sum = 0.0f;
  for (int h = 0; h < hidden_size; h++) {
    int w_idx = k * hidden_size + h;
    int bh = batch * hidden_size + h;
    sum += Wf[w_idx] * dz_f[bh] + Wi[w_idx] * dz_i[bh] +
           Wc[w_idx] * dz_c[bh] + Wo[w_idx] * dz_o[bh];
  }

  if (k < hidden_size) {
    dx[batch * hidden_size + k] = sum;
  } else {
    dh_prev[batch * hidden_size + (k - hidden_size)] = sum;
  }
}

__global__ void lstmWeightGradKernel(const float *dz, const float *x,
                                     const float *h_prev, float *dW,
                                     int batch_size, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int concat_size = 2 * hidden_size;
  int total = concat_size * hidden_size;
  if (idx >= total)
    return;

  int k = idx / hidden_size;
  int h = idx % hidden_size;

  float sum = 0.0f;
  for (int b = 0; b < batch_size; b++) {
    float concat_val;
    if (k < hidden_size) {
      concat_val = x[b * hidden_size + k];
    } else {
      concat_val = h_prev[b * hidden_size + (k - hidden_size)];
    }
    sum += dz[b * hidden_size + h] * concat_val;
  }
  dW[idx] += sum;
}

class LSTMLayer : public Operation {
private:
  float *Wf, *Wi, *Wc, *Wo, *bf, *bi, *bc, *bo;
  float *h, *c;
  float *h_all_, *c_all_;
  float *cache_f_, *cache_i_, *cache_c_hat_, *cache_o_;
  float *cached_input_;
  float *dWf_, *dWi_, *dWc_, *dWo_, *dbf_, *dbi_, *dbc_, *dbo_;
  float *dh_next_, *dc_next_;
  float *dz_f_, *dz_i_, *dz_c_, *dz_o_;
  int hidden_size_, batch_size_, sequence_length_;

public:
  LSTMLayer(int hidden_size, int batch_size, int sequence_length)
      : hidden_size_(hidden_size), batch_size_(batch_size),
        sequence_length_(sequence_length) {
    int weight_size = 2 * hidden_size * hidden_size;
    int bh = batch_size * hidden_size;

    CUDA_CHECK(cudaMalloc(&Wf, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Wi, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Wc, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Wo, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bf, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bi, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bc, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bo, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, bh * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&h_all_, (sequence_length + 1) * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&c_all_, (sequence_length + 1) * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&cache_f_, sequence_length * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&cache_i_, sequence_length * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&cache_c_hat_, sequence_length * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&cache_o_, sequence_length * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&cached_input_, sequence_length * bh * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dWf_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWi_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWc_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWo_, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbf_, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbi_, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbc_, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbo_, hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dh_next_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc_next_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz_f_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz_i_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz_c_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz_o_, bh * sizeof(float)));

    CUDA_CHECK(cudaMemset(Wf, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Wi, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Wc, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Wo, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bf, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bi, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bc, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bo, 0, hidden_size * sizeof(float)));
  }

  ~LSTMLayer() {
    cudaFree(Wf);
    cudaFree(Wi);
    cudaFree(Wc);
    cudaFree(Wo);
    cudaFree(bf);
    cudaFree(bi);
    cudaFree(bc);
    cudaFree(bo);
    cudaFree(h);
    cudaFree(c);
    cudaFree(h_all_);
    cudaFree(c_all_);
    cudaFree(cache_f_);
    cudaFree(cache_i_);
    cudaFree(cache_c_hat_);
    cudaFree(cache_o_);
    cudaFree(cached_input_);
    cudaFree(dWf_);
    cudaFree(dWi_);
    cudaFree(dWc_);
    cudaFree(dWo_);
    cudaFree(dbf_);
    cudaFree(dbi_);
    cudaFree(dbc_);
    cudaFree(dbo_);
    cudaFree(dh_next_);
    cudaFree(dc_next_);
    cudaFree(dz_f_);
    cudaFree(dz_i_);
    cudaFree(dz_c_);
    cudaFree(dz_o_);
  }

  int input_size() const override {
    return sequence_length_ * batch_size_ * hidden_size_;
  }
  int output_size() const override {
    return sequence_length_ * batch_size_ * hidden_size_;
  }

  void forward(float *d_input, float *d_output) override {
    int bh = batch_size_ * hidden_size_;
    int num_blocks = (bh + NUM_THREADS - 1) / NUM_THREADS;

    CUDA_CHECK(cudaMemset(h, 0, bh * sizeof(float)));
    CUDA_CHECK(cudaMemset(c, 0, bh * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(h_all_, h, bh * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(c_all_, c, bh * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int t = 0; t < sequence_length_; t++) {
      float *x_t = d_input + t * bh;

      CUDA_CHECK(cudaMemcpy(cached_input_ + t * bh, x_t,
                             bh * sizeof(float), cudaMemcpyDeviceToDevice));

      lstmForwardCachingKernel<<<num_blocks, NUM_THREADS>>>(
          x_t, Wf, Wi, Wc, Wo, bf, bi, bc, bo, h, c, cache_f_ + t * bh,
          cache_i_ + t * bh, cache_c_hat_ + t * bh, cache_o_ + t * bh,
          hidden_size_, batch_size_);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(h_all_ + (t + 1) * bh, h, bh * sizeof(float),
                             cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(c_all_ + (t + 1) * bh, c, bh * sizeof(float),
                             cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(d_output + t * bh, h, bh * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int bh = batch_size_ * hidden_size_;
    int num_blocks = (bh + NUM_THREADS - 1) / NUM_THREADS;
    int weight_size = 2 * hidden_size_ * hidden_size_;

    CUDA_CHECK(cudaMemset(dWf_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(dWi_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(dWc_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(dWo_, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(dbf_, 0, hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(dbi_, 0, hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(dbc_, 0, hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(dbo_, 0, hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(dh_next_, 0, bh * sizeof(float)));
    CUDA_CHECK(cudaMemset(dc_next_, 0, bh * sizeof(float)));

    for (int t = sequence_length_ - 1; t >= 0; t--) {
      float *dh_t = d_output_grad + t * bh;
      float *dx_t = d_input_grad + t * bh;
      float *x_t = cached_input_ + t * bh;
      float *h_prev = h_all_ + t * bh;
      float *c_prev = c_all_ + t * bh;
      float *c_cur = c_all_ + (t + 1) * bh;

      lstmBackwardGatesKernel<<<num_blocks, NUM_THREADS>>>(
          dh_t, dh_next_, dc_next_, c_prev, c_cur, cache_f_ + t * bh,
          cache_i_ + t * bh, cache_c_hat_ + t * bh, cache_o_ + t * bh,
          dz_f_, dz_i_, dz_c_, dz_o_, dc_next_, bh);
      CUDA_CHECK(cudaDeviceSynchronize());

      int concat_total = batch_size_ * 2 * hidden_size_;
      int concat_blocks = (concat_total + NUM_THREADS - 1) / NUM_THREADS;
      lstmBackwardConcatKernel<<<concat_blocks, NUM_THREADS>>>(
          dz_f_, dz_i_, dz_c_, dz_o_, Wf, Wi, Wc, Wo, dx_t, dh_next_,
          batch_size_, hidden_size_);
      CUDA_CHECK(cudaDeviceSynchronize());

      int w_blocks = (weight_size + NUM_THREADS - 1) / NUM_THREADS;
      lstmWeightGradKernel<<<w_blocks, NUM_THREADS>>>(
          dz_f_, x_t, h_prev, dWf_, batch_size_, hidden_size_);
      lstmWeightGradKernel<<<w_blocks, NUM_THREADS>>>(
          dz_i_, x_t, h_prev, dWi_, batch_size_, hidden_size_);
      lstmWeightGradKernel<<<w_blocks, NUM_THREADS>>>(
          dz_c_, x_t, h_prev, dWc_, batch_size_, hidden_size_);
      lstmWeightGradKernel<<<w_blocks, NUM_THREADS>>>(
          dz_o_, x_t, h_prev, dWo_, batch_size_, hidden_size_);

      int b_blocks = (hidden_size_ + NUM_THREADS - 1) / NUM_THREADS;
      biasGradAccumKernel<<<b_blocks, NUM_THREADS>>>(
          dz_f_, dbf_, batch_size_, hidden_size_);
      biasGradAccumKernel<<<b_blocks, NUM_THREADS>>>(
          dz_i_, dbi_, batch_size_, hidden_size_);
      biasGradAccumKernel<<<b_blocks, NUM_THREADS>>>(
          dz_c_, dbc_, batch_size_, hidden_size_);
      biasGradAccumKernel<<<b_blocks, NUM_THREADS>>>(
          dz_o_, dbo_, batch_size_, hidden_size_);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  void update_weights(float lr) override {
    int weight_size = 2 * hidden_size_ * hidden_size_;
    int w_blocks = (weight_size + NUM_THREADS - 1) / NUM_THREADS;
    sgdUpdateKernel<<<w_blocks, NUM_THREADS>>>(Wf, dWf_, lr, weight_size);
    sgdUpdateKernel<<<w_blocks, NUM_THREADS>>>(Wi, dWi_, lr, weight_size);
    sgdUpdateKernel<<<w_blocks, NUM_THREADS>>>(Wc, dWc_, lr, weight_size);
    sgdUpdateKernel<<<w_blocks, NUM_THREADS>>>(Wo, dWo_, lr, weight_size);

    int b_blocks = (hidden_size_ + NUM_THREADS - 1) / NUM_THREADS;
    sgdUpdateKernel<<<b_blocks, NUM_THREADS>>>(bf, dbf_, lr, hidden_size_);
    sgdUpdateKernel<<<b_blocks, NUM_THREADS>>>(bi, dbi_, lr, hidden_size_);
    sgdUpdateKernel<<<b_blocks, NUM_THREADS>>>(bc, dbc_, lr, hidden_size_);
    sgdUpdateKernel<<<b_blocks, NUM_THREADS>>>(bo, dbo_, lr, hidden_size_);
  }
};

__global__ void elmanRnnKernel(const float *x, const float *h_prev,
                               const float *Wxh, const float *Whh,
                               const float *b_h, const float *Why,
                               const float *b_y, float *h_out, float *y_out,
                               int batch_size, int input_size, int hidden_size,
                               int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    for (int j = 0; j < hidden_size; j++) {
      float hidden_sum = 0.0f;
      for (int i = 0; i < input_size; i++) {
        hidden_sum += Wxh[i * hidden_size + j] * x[idx * input_size + i];
      }
      for (int i = 0; i < hidden_size; i++) {
        hidden_sum += Whh[i * hidden_size + j] * h_prev[idx * hidden_size + i];
      }
      h_out[idx * hidden_size + j] = tanhf(hidden_sum + b_h[j]);
    }

    for (int j = 0; j < output_size; j++) {
      float output_sum = 0.0f;
      for (int i = 0; i < hidden_size; i++) {
        output_sum +=
            Why[i * output_size + j] * h_out[idx * hidden_size + i];
      }
      y_out[idx * output_size + j] = output_sum + b_y[j];
    }
  }
}

class ElmanRNNLayer : public Operation {
private:
  float *Wxh, *Whh, *b_h, *Why, *b_y;
  float *h;
  float *h_all_;
  float *cached_input_;
  float *dWxh_, *dWhh_, *db_h_, *dWhy_, *db_y_;
  float *dh_next_, *dh_buf_, *d_pre_h_;
  int input_size_, hidden_size_, output_size_, batch_size_, sequence_length_;

public:
  ElmanRNNLayer(int input_size, int hidden_size, int output_size,
                int batch_size, int sequence_length)
      : input_size_(input_size), hidden_size_(hidden_size),
        output_size_(output_size), batch_size_(batch_size),
        sequence_length_(sequence_length) {
    int bh = batch_size * hidden_size;
    int bi_size = batch_size * input_size;

    CUDA_CHECK(cudaMalloc(&Wxh, input_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Whh, hidden_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_h, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Why, hidden_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_y, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h, bh * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&h_all_, (sequence_length + 1) * bh * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&cached_input_, sequence_length * bi_size * sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(&dWxh_, input_size * hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&dWhh_, hidden_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db_h_, hidden_size * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&dWhy_, hidden_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db_y_, output_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dh_next_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh_buf_, bh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pre_h_, bh * sizeof(float)));

    CUDA_CHECK(cudaMemset(Wxh, 0, input_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Whh, 0, hidden_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(b_h, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(Why, 0, hidden_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(b_y, 0, output_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(h, 0, bh * sizeof(float)));
  }

  ~ElmanRNNLayer() {
    cudaFree(Wxh);
    cudaFree(Whh);
    cudaFree(b_h);
    cudaFree(Why);
    cudaFree(b_y);
    cudaFree(h);
    cudaFree(h_all_);
    cudaFree(cached_input_);
    cudaFree(dWxh_);
    cudaFree(dWhh_);
    cudaFree(db_h_);
    cudaFree(dWhy_);
    cudaFree(db_y_);
    cudaFree(dh_next_);
    cudaFree(dh_buf_);
    cudaFree(d_pre_h_);
  }

  int input_size() const override {
    return sequence_length_ * batch_size_ * input_size_;
  }
  int output_size() const override {
    return sequence_length_ * batch_size_ * output_size_;
  }

  void forward(float *d_input, float *d_output) override {
    int num_blocks = (batch_size_ + NUM_THREADS - 1) / NUM_THREADS;
    int bh = batch_size_ * hidden_size_;
    int bi_size = batch_size_ * input_size_;

    CUDA_CHECK(cudaMemset(h, 0, bh * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(h_all_, h, bh * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int t = 0; t < sequence_length_; t++) {
      float *x_t = d_input + t * bi_size;
      float *y_t = d_output + t * batch_size_ * output_size_;

      CUDA_CHECK(cudaMemcpy(cached_input_ + t * bi_size, x_t,
                             bi_size * sizeof(float),
                             cudaMemcpyDeviceToDevice));

      elmanRnnKernel<<<num_blocks, NUM_THREADS>>>(
          x_t, h, Wxh, Whh, b_h, Why, b_y, h, y_t, batch_size_, input_size_,
          hidden_size_, output_size_);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(h_all_ + (t + 1) * bh, h, bh * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int bh = batch_size_ * hidden_size_;
    int bo = batch_size_ * output_size_;
    int bi_size = batch_size_ * input_size_;
    int bh_blocks = (bh + NUM_THREADS - 1) / NUM_THREADS;

    CUDA_CHECK(
        cudaMemset(dWxh_, 0, input_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(dWhh_, 0, hidden_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(db_h_, 0, hidden_size_ * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(dWhy_, 0, hidden_size_ * output_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(db_y_, 0, output_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(dh_next_, 0, bh * sizeof(float)));

    for (int t = sequence_length_ - 1; t >= 0; t--) {
      float *dy_t = d_output_grad + t * bo;
      float *dx_t = d_input_grad + t * bi_size;
      float *x_t = cached_input_ + t * bi_size;
      float *h_t = h_all_ + (t + 1) * bh;
      float *h_prev = h_all_ + t * bh;

      transposeMatVecKernel<<<bh_blocks, NUM_THREADS>>>(
          dy_t, Why, dh_buf_, batch_size_, hidden_size_, output_size_);
      CUDA_CHECK(cudaDeviceSynchronize());

      addVectorsKernel<<<bh_blocks, NUM_THREADS>>>(
          dh_buf_, dh_next_, dh_buf_, bh);
      CUDA_CHECK(cudaDeviceSynchronize());

      tanhBackwardElementKernel<<<bh_blocks, NUM_THREADS>>>(
          dh_buf_, h_t, d_pre_h_, bh);
      CUDA_CHECK(cudaDeviceSynchronize());

      int bi_blocks = (bi_size + NUM_THREADS - 1) / NUM_THREADS;
      transposeMatVecKernel<<<bi_blocks, NUM_THREADS>>>(
          d_pre_h_, Wxh, dx_t, batch_size_, input_size_, hidden_size_);

      transposeMatVecKernel<<<bh_blocks, NUM_THREADS>>>(
          d_pre_h_, Whh, dh_next_, batch_size_, hidden_size_, hidden_size_);
      CUDA_CHECK(cudaDeviceSynchronize());

      int wxh_size = input_size_ * hidden_size_;
      int wxh_blocks = (wxh_size + NUM_THREADS - 1) / NUM_THREADS;
      weightGradAccumKernel<<<wxh_blocks, NUM_THREADS>>>(
          d_pre_h_, x_t, dWxh_, batch_size_, input_size_, hidden_size_);

      int whh_size = hidden_size_ * hidden_size_;
      int whh_blocks = (whh_size + NUM_THREADS - 1) / NUM_THREADS;
      weightGradAccumKernel<<<whh_blocks, NUM_THREADS>>>(
          d_pre_h_, h_prev, dWhh_, batch_size_, hidden_size_, hidden_size_);

      int bh_grad_blocks = (hidden_size_ + NUM_THREADS - 1) / NUM_THREADS;
      biasGradAccumKernel<<<bh_grad_blocks, NUM_THREADS>>>(
          d_pre_h_, db_h_, batch_size_, hidden_size_);

      int why_size = hidden_size_ * output_size_;
      int why_blocks = (why_size + NUM_THREADS - 1) / NUM_THREADS;
      weightGradAccumKernel<<<why_blocks, NUM_THREADS>>>(
          dy_t, h_t, dWhy_, batch_size_, hidden_size_, output_size_);

      int by_blocks = (output_size_ + NUM_THREADS - 1) / NUM_THREADS;
      biasGradAccumKernel<<<by_blocks, NUM_THREADS>>>(
          dy_t, db_y_, batch_size_, output_size_);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  void update_weights(float lr) override {
    int wxh_size = input_size_ * hidden_size_;
    int whh_size = hidden_size_ * hidden_size_;
    int why_size = hidden_size_ * output_size_;

    sgdUpdateKernel<<<(wxh_size + 255) / 256, 256>>>(Wxh, dWxh_, lr,
                                                      wxh_size);
    sgdUpdateKernel<<<(whh_size + 255) / 256, 256>>>(Whh, dWhh_, lr,
                                                      whh_size);
    sgdUpdateKernel<<<(hidden_size_ + 255) / 256, 256>>>(b_h, db_h_, lr,
                                                          hidden_size_);
    sgdUpdateKernel<<<(why_size + 255) / 256, 256>>>(Why, dWhy_, lr,
                                                      why_size);
    sgdUpdateKernel<<<(output_size_ + 255) / 256, 256>>>(b_y, db_y_, lr,
                                                          output_size_);
  }
};

__global__ void conv1dKernel(const float *input, const float *kernel,
                             float *output, int inputSize, int kernelSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int halfK = kernelSize / 2;

  if (idx < inputSize) {
    float sum = 0.0f;
    for (int i = -halfK; i <= halfK; i++) {
      if (idx + i >= 0 && idx + i < inputSize) {
        sum += input[idx + i] * kernel[halfK + i];
      }
    }
    output[idx] = sum;
  }
}

__global__ void conv1dBackwardInputKernel(const float *d_output_grad,
                                          const float *kernel,
                                          float *d_input_grad, int inputSize,
                                          int kernelSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int halfK = kernelSize / 2;

  if (idx < inputSize) {
    float sum = 0.0f;
    for (int k = -halfK; k <= halfK; k++) {
      int j = idx - k;
      if (j >= 0 && j < inputSize) {
        sum += d_output_grad[j] * kernel[halfK + k];
      }
    }
    d_input_grad[idx] = sum;
  }
}

__global__ void conv1dBackwardKernelGradKernel(const float *d_output_grad,
                                               const float *input,
                                               float *d_kernel_grad,
                                               int inputSize, int kernelSize) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int halfK = kernelSize / 2;

  if (j < kernelSize) {
    float sum = 0.0f;
    int offset = j - halfK;
    for (int i = 0; i < inputSize; i++) {
      int input_idx = i + offset;
      if (input_idx >= 0 && input_idx < inputSize) {
        sum += d_output_grad[i] * input[input_idx];
      }
    }
    d_kernel_grad[j] = sum;
  }
}

class Conv1DLayer : public Operation {
private:
  float *d_kernel, *d_kernel_grad, *d_cached_input;
  int inputSize_, kernelSize_;

public:
  Conv1DLayer(int inputSize, int kernelSize)
      : inputSize_(inputSize), kernelSize_(kernelSize) {
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kernel, 0, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel_grad, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cached_input, inputSize * sizeof(float)));
  }

  ~Conv1DLayer() {
    cudaFree(d_kernel);
    cudaFree(d_kernel_grad);
    cudaFree(d_cached_input);
  }

  int input_size() const override { return inputSize_; }
  int output_size() const override { return inputSize_; }

  void forward(float *d_input, float *d_output) override {
    CUDA_CHECK(cudaMemcpy(d_cached_input, d_input, inputSize_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    int blocks = (inputSize_ + 255) / 256;
    conv1dKernel<<<blocks, 256>>>(d_input, d_kernel, d_output, inputSize_,
                                  kernelSize_);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    int blocks = (inputSize_ + 255) / 256;
    conv1dBackwardInputKernel<<<blocks, 256>>>(d_output_grad, d_kernel,
                                               d_input_grad, inputSize_,
                                               kernelSize_);

    int k_blocks = (kernelSize_ + 255) / 256;
    conv1dBackwardKernelGradKernel<<<k_blocks, 256>>>(
        d_output_grad, d_cached_input, d_kernel_grad, inputSize_, kernelSize_);
  }

  void update_weights(float lr) override {
    int blocks = (kernelSize_ + 255) / 256;
    sgdUpdateKernel<<<blocks, 256>>>(d_kernel, d_kernel_grad, lr, kernelSize_);
  }
};

__global__ void conv2dKernel(const float *input, int inputWidth,
                             int inputHeight, const float *kernel,
                             int kernelWidth, int kernelHeight, float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int halfKernelWidth = kernelWidth / 2;
  int halfKernelHeight = kernelHeight / 2;

  if (x < inputWidth && y < inputHeight) {
    float value = 0.0f;
    for (int ky = -halfKernelHeight; ky <= halfKernelHeight; ky++) {
      for (int kx = -halfKernelWidth; kx <= halfKernelWidth; kx++) {
        int inX = x + kx;
        int inY = y + ky;
        if (inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
          value += input[inY * inputWidth + inX] *
                   kernel[(ky + halfKernelHeight) * kernelWidth +
                          (kx + halfKernelWidth)];
        }
      }
    }
    output[y * inputWidth + x] = value;
  }
}

__global__ void conv2dBackwardInputKernel(const float *d_output_grad,
                                          int inputWidth, int inputHeight,
                                          const float *kernel, int kernelWidth,
                                          int kernelHeight,
                                          float *d_input_grad) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int halfKW = kernelWidth / 2;
  int halfKH = kernelHeight / 2;

  if (x < inputWidth && y < inputHeight) {
    float sum = 0.0f;
    for (int ky = -halfKH; ky <= halfKH; ky++) {
      for (int kx = -halfKW; kx <= halfKW; kx++) {
        int oy = y - ky;
        int ox = x - kx;
        if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth) {
          sum += d_output_grad[oy * inputWidth + ox] *
                 kernel[(ky + halfKH) * kernelWidth + (kx + halfKW)];
        }
      }
    }
    d_input_grad[y * inputWidth + x] = sum;
  }
}

__global__ void conv2dBackwardKernelGradKernel(const float *d_output_grad,
                                               const float *input,
                                               int inputWidth, int inputHeight,
                                               float *d_kernel_grad,
                                               int kernelWidth,
                                               int kernelHeight) {
  int kx = blockIdx.x * blockDim.x + threadIdx.x;
  int ky = blockIdx.y * blockDim.y + threadIdx.y;
  int halfKW = kernelWidth / 2;
  int halfKH = kernelHeight / 2;

  if (kx < kernelWidth && ky < kernelHeight) {
    float sum = 0.0f;
    int offY = ky - halfKH;
    int offX = kx - halfKW;
    for (int y = 0; y < inputHeight; y++) {
      for (int x = 0; x < inputWidth; x++) {
        int iy = y + offY;
        int ix = x + offX;
        if (iy >= 0 && iy < inputHeight && ix >= 0 && ix < inputWidth) {
          sum += d_output_grad[y * inputWidth + x] *
                 input[iy * inputWidth + ix];
        }
      }
    }
    d_kernel_grad[ky * kernelWidth + kx] = sum;
  }
}

class Conv2DLayer : public Operation {
private:
  float *d_kernel, *d_kernel_grad, *d_cached_input;
  int inputWidth_, inputHeight_;
  int kernelWidth_, kernelHeight_;

public:
  Conv2DLayer(int inputWidth, int inputHeight, int kernelWidth,
              int kernelHeight)
      : inputWidth_(inputWidth), inputHeight_(inputHeight),
        kernelWidth_(kernelWidth), kernelHeight_(kernelHeight) {
    int k_size = kernelWidth * kernelHeight;
    int i_size = inputWidth * inputHeight;
    CUDA_CHECK(cudaMalloc(&d_kernel, k_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kernel, 0, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel_grad, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cached_input, i_size * sizeof(float)));
  }

  ~Conv2DLayer() {
    cudaFree(d_kernel);
    cudaFree(d_kernel_grad);
    cudaFree(d_cached_input);
  }

  int input_size() const override { return inputWidth_ * inputHeight_; }
  int output_size() const override { return inputWidth_ * inputHeight_; }

  void forward(float *d_input, float *d_output) override {
    CUDA_CHECK(cudaMemcpy(d_cached_input, d_input,
                           inputWidth_ * inputHeight_ * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    dim3 block_size(16, 16);
    dim3 grid_size((inputWidth_ + block_size.x - 1) / block_size.x,
                   (inputHeight_ + block_size.y - 1) / block_size.y);

    conv2dKernel<<<grid_size, block_size>>>(d_input, inputWidth_, inputHeight_,
                                            d_kernel, kernelWidth_,
                                            kernelHeight_, d_output);
  }

  void backward(float *d_output_grad, float *d_input_grad) override {
    dim3 block_size(16, 16);
    dim3 grid_size((inputWidth_ + block_size.x - 1) / block_size.x,
                   (inputHeight_ + block_size.y - 1) / block_size.y);

    conv2dBackwardInputKernel<<<grid_size, block_size>>>(
        d_output_grad, inputWidth_, inputHeight_, d_kernel, kernelWidth_,
        kernelHeight_, d_input_grad);

    dim3 k_grid((kernelWidth_ + block_size.x - 1) / block_size.x,
                (kernelHeight_ + block_size.y - 1) / block_size.y);
    conv2dBackwardKernelGradKernel<<<k_grid, block_size>>>(
        d_output_grad, d_cached_input, inputWidth_, inputHeight_,
        d_kernel_grad, kernelWidth_, kernelHeight_);
  }

  void update_weights(float lr) override {
    int k_size = kernelWidth_ * kernelHeight_;
    int blocks = (k_size + 255) / 256;
    sgdUpdateKernel<<<blocks, 256>>>(d_kernel, d_kernel_grad, lr, k_size);
  }
};
