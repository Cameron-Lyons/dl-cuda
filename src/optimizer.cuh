#pragma once

#include "layers.cuh"
#include "optimizers.cuh"
#include <cmath>

class Optimizer {
public:
  virtual void step(float lr) = 0;
  virtual void register_params(const std::vector<ParamGroup> &groups) = 0;
  virtual ~Optimizer() = default;
};

class SGDOptimizer : public Optimizer {
private:
  std::vector<ParamGroup> param_groups_;

public:
  void register_params(const std::vector<ParamGroup> &groups) override {
    param_groups_ = groups;
  }

  void step(float lr) override {
    for (auto &pg : param_groups_) {
      int blocks = (pg.size + 255) / 256;
      sgdUpdateKernel<<<blocks, 256>>>(pg.params, pg.grads, lr, pg.size);
    }
  }
};

class AdamOptimizer : public Optimizer {
private:
  std::vector<ParamGroup> param_groups_;
  std::vector<float *> m_buffers_;
  std::vector<float *> v_buffers_;
  float beta1_, beta2_, epsilon_;
  int t_ = 0;

public:
  AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f,
                float epsilon = 1e-8f)
      : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

  ~AdamOptimizer() {
    for (auto *p : m_buffers_)
      cudaFree(p);
    for (auto *p : v_buffers_)
      cudaFree(p);
  }

  void register_params(const std::vector<ParamGroup> &groups) override {
    for (auto *p : m_buffers_)
      cudaFree(p);
    for (auto *p : v_buffers_)
      cudaFree(p);
    m_buffers_.clear();
    v_buffers_.clear();

    param_groups_ = groups;
    for (auto &pg : param_groups_) {
      float *m, *v;
      CUDA_CHECK(cudaMalloc(&m, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&v, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMemset(m, 0, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMemset(v, 0, pg.size * sizeof(float)));
      m_buffers_.push_back(m);
      v_buffers_.push_back(v);
    }
    t_ = 0;
  }

  void step(float lr) override {
    t_++;
    float inv_bias_correction1 =
        1.0f / (1.0f - powf(beta1_, static_cast<float>(t_)));
    float inv_bias_correction2 =
        1.0f / (1.0f - powf(beta2_, static_cast<float>(t_)));
    for (size_t i = 0; i < param_groups_.size(); i++) {
      auto &pg = param_groups_[i];
      int blocks = (pg.size + 255) / 256;
      updateAdam<<<blocks, 256>>>(pg.grads, m_buffers_[i], v_buffers_[i],
                                  pg.params, lr, beta1_, beta2_, epsilon_,
                                  inv_bias_correction1, inv_bias_correction2,
                                  pg.size);
    }
  }
};

class AdamWOptimizer : public Optimizer {
private:
  std::vector<ParamGroup> param_groups_;
  std::vector<float *> m_buffers_;
  std::vector<float *> v_buffers_;
  float **d_param_ptrs_ = nullptr;
  float **d_grad_ptrs_ = nullptr;
  float **d_m_ptrs_ = nullptr;
  float **d_v_ptrs_ = nullptr;
  int *d_group_sizes_ = nullptr;
  int num_groups_ = 0;
  float beta1_, beta2_, epsilon_, weight_decay_;
  int t_ = 0;

  void clear_moment_buffers() {
    for (auto *p : m_buffers_) {
      cudaFree(p);
    }
    for (auto *p : v_buffers_) {
      cudaFree(p);
    }
    m_buffers_.clear();
    v_buffers_.clear();
  }

  void clear_device_metadata() {
    if (d_param_ptrs_) {
      cudaFree(d_param_ptrs_);
      d_param_ptrs_ = nullptr;
    }
    if (d_grad_ptrs_) {
      cudaFree(d_grad_ptrs_);
      d_grad_ptrs_ = nullptr;
    }
    if (d_m_ptrs_) {
      cudaFree(d_m_ptrs_);
      d_m_ptrs_ = nullptr;
    }
    if (d_v_ptrs_) {
      cudaFree(d_v_ptrs_);
      d_v_ptrs_ = nullptr;
    }
    if (d_group_sizes_) {
      cudaFree(d_group_sizes_);
      d_group_sizes_ = nullptr;
    }
    num_groups_ = 0;
  }

  void rebuild_device_metadata() {
    clear_device_metadata();
    num_groups_ = static_cast<int>(param_groups_.size());
    if (num_groups_ == 0) {
      return;
    }

    std::vector<float *> h_param_ptrs(static_cast<size_t>(num_groups_));
    std::vector<float *> h_grad_ptrs(static_cast<size_t>(num_groups_));
    std::vector<float *> h_m_ptrs(static_cast<size_t>(num_groups_));
    std::vector<float *> h_v_ptrs(static_cast<size_t>(num_groups_));
    std::vector<int> h_sizes(static_cast<size_t>(num_groups_));
    for (int i = 0; i < num_groups_; i++) {
      h_param_ptrs[static_cast<size_t>(i)] = param_groups_[static_cast<size_t>(i)].params;
      h_grad_ptrs[static_cast<size_t>(i)] = param_groups_[static_cast<size_t>(i)].grads;
      h_m_ptrs[static_cast<size_t>(i)] = m_buffers_[static_cast<size_t>(i)];
      h_v_ptrs[static_cast<size_t>(i)] = v_buffers_[static_cast<size_t>(i)];
      h_sizes[static_cast<size_t>(i)] = param_groups_[static_cast<size_t>(i)].size;
    }

    CUDA_CHECK(
        cudaMalloc(&d_param_ptrs_, static_cast<size_t>(num_groups_) * sizeof(float *)));
    CUDA_CHECK(
        cudaMalloc(&d_grad_ptrs_, static_cast<size_t>(num_groups_) * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_m_ptrs_, static_cast<size_t>(num_groups_) * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_v_ptrs_, static_cast<size_t>(num_groups_) * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_group_sizes_, static_cast<size_t>(num_groups_) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_param_ptrs_, h_param_ptrs.data(),
                          static_cast<size_t>(num_groups_) * sizeof(float *),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_ptrs_, h_grad_ptrs.data(),
                          static_cast<size_t>(num_groups_) * sizeof(float *),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m_ptrs_, h_m_ptrs.data(),
                          static_cast<size_t>(num_groups_) * sizeof(float *),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_ptrs_, h_v_ptrs.data(),
                          static_cast<size_t>(num_groups_) * sizeof(float *),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_group_sizes_, h_sizes.data(),
                          static_cast<size_t>(num_groups_) * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

public:
  AdamWOptimizer(float beta1 = 0.9f, float beta2 = 0.999f,
                 float epsilon = 1e-8f, float weight_decay = 0.01f)
      : beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
        weight_decay_(weight_decay) {}

  ~AdamWOptimizer() {
    clear_moment_buffers();
    clear_device_metadata();
  }

  void register_params(const std::vector<ParamGroup> &groups) override {
    clear_moment_buffers();
    clear_device_metadata();

    param_groups_ = groups;
    for (auto &pg : param_groups_) {
      float *m, *v;
      CUDA_CHECK(cudaMalloc(&m, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&v, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMemset(m, 0, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMemset(v, 0, pg.size * sizeof(float)));
      m_buffers_.push_back(m);
      v_buffers_.push_back(v);
    }
    rebuild_device_metadata();
    t_ = 0;
  }

  void step(float lr) override {
    if (num_groups_ == 0) {
      return;
    }
    t_++;
    float inv_bias_correction1 =
        1.0f / (1.0f - powf(beta1_, static_cast<float>(t_)));
    float inv_bias_correction2 =
        1.0f / (1.0f - powf(beta2_, static_cast<float>(t_)));
    updateAdamWMultiTensor<<<num_groups_, 256>>>(
        d_grad_ptrs_, d_m_ptrs_, d_v_ptrs_, d_param_ptrs_, d_group_sizes_,
        num_groups_, lr, beta1_, beta2_, epsilon_, weight_decay_,
        inv_bias_correction1, inv_bias_correction2);
    CUDA_CHECK(cudaGetLastError());
  }
};

class RMSpropOptimizer : public Optimizer {
private:
  std::vector<ParamGroup> param_groups_;
  std::vector<float *> s_buffers_;
  float decay_rate_, epsilon_;

public:
  RMSpropOptimizer(float decay_rate = 0.9f, float epsilon = 1e-8f)
      : decay_rate_(decay_rate), epsilon_(epsilon) {}

  ~RMSpropOptimizer() {
    for (auto *p : s_buffers_)
      cudaFree(p);
  }

  void register_params(const std::vector<ParamGroup> &groups) override {
    for (auto *p : s_buffers_)
      cudaFree(p);
    s_buffers_.clear();

    param_groups_ = groups;
    for (auto &pg : param_groups_) {
      float *s;
      CUDA_CHECK(cudaMalloc(&s, pg.size * sizeof(float)));
      CUDA_CHECK(cudaMemset(s, 0, pg.size * sizeof(float)));
      s_buffers_.push_back(s);
    }
  }

  void step(float lr) override {
    for (size_t i = 0; i < param_groups_.size(); i++) {
      auto &pg = param_groups_[i];
      int blocks = (pg.size + 255) / 256;
      updateRMSprop<<<blocks, 256>>>(pg.grads, s_buffers_[i], pg.params, lr,
                                     decay_rate_, epsilon_, pg.size);
    }
  }
};

inline void Sequential::set_optimizer(Optimizer *opt) {
  optimizer_ = opt;
  if (param_groups_.empty() && !operations.empty()) {
    rebuild_param_group_cache();
  }
  if (optimizer_) {
    optimizer_->register_params(param_groups_);
  }
}

inline float Sequential::clip_grad_norm(float max_norm) {
  if (param_groups_.empty() && !operations.empty()) {
    rebuild_param_group_cache();
  }
  if (param_groups_.empty()) {
    return 0.0f;
  }

  CUDA_CHECK(cudaMemsetAsync(d_total_grad_norm_sq_, 0, sizeof(float), 0));
  accumulateGradNormSqKernel<<<num_param_groups_, 256, 256 * sizeof(float)>>>(
      d_grad_ptrs_, d_group_sizes_, num_param_groups_, d_total_grad_norm_sq_);
  CUDA_CHECK(cudaGetLastError());

  float total_norm_sq = 0.0f;
  CUDA_CHECK(cudaMemcpy(&total_norm_sq, d_total_grad_norm_sq_, sizeof(float),
                        cudaMemcpyDeviceToHost));
  float total_norm = sqrtf(total_norm_sq);

  if (total_norm > max_norm) {
    float clip_coeff = max_norm / (total_norm + 1e-6f);
    clipGradsMultiTensorKernel<<<num_param_groups_, 256>>>(
        d_grad_ptrs_, d_group_sizes_, num_param_groups_, clip_coeff);
    CUDA_CHECK(cudaGetLastError());
  }

  return total_norm;
}

inline void Sequential::update_weights(float lr) {
  if (optimizer_) {
    optimizer_->step(lr);
  } else {
    for (auto *op : operations) {
      op->update_weights(lr);
    }
  }
}
