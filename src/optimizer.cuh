#pragma once

#include "layers.cuh"
#include "optimizers.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

struct SquareOp {
  __host__ __device__ float operator()(float x) const { return x * x; }
};

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
    for (size_t i = 0; i < param_groups_.size(); i++) {
      auto &pg = param_groups_[i];
      int blocks = (pg.size + 255) / 256;
      updateAdam<<<blocks, 256>>>(pg.grads, m_buffers_[i], v_buffers_[i],
                                  pg.params, lr, beta1_, beta2_, epsilon_, t_,
                                  pg.size);
    }
  }
};

class AdamWOptimizer : public Optimizer {
private:
  std::vector<ParamGroup> param_groups_;
  std::vector<float *> m_buffers_;
  std::vector<float *> v_buffers_;
  float beta1_, beta2_, epsilon_, weight_decay_;
  int t_ = 0;

public:
  AdamWOptimizer(float beta1 = 0.9f, float beta2 = 0.999f,
                 float epsilon = 1e-8f, float weight_decay = 0.01f)
      : beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
        weight_decay_(weight_decay) {}

  ~AdamWOptimizer() {
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
    for (size_t i = 0; i < param_groups_.size(); i++) {
      auto &pg = param_groups_[i];
      int blocks = (pg.size + 255) / 256;
      updateAdamW<<<blocks, 256>>>(pg.grads, m_buffers_[i], v_buffers_[i],
                                   pg.params, lr, beta1_, beta2_, epsilon_,
                                   weight_decay_, t_, pg.size);
    }
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
  std::vector<ParamGroup> all_groups;
  for (auto *op : operations) {
    auto groups = op->get_param_groups();
    all_groups.insert(all_groups.end(), groups.begin(), groups.end());
  }
  optimizer_->register_params(all_groups);
}

inline float Sequential::clip_grad_norm(float max_norm) {
  std::vector<ParamGroup> all_groups;
  for (auto *op : operations) {
    auto groups = op->get_param_groups();
    all_groups.insert(all_groups.end(), groups.begin(), groups.end());
  }
  if (all_groups.empty())
    return 0.0f;

  float total_norm_sq = 0.0f;
  for (auto &pg : all_groups) {
    thrust::device_ptr<float> begin(pg.grads);
    thrust::device_ptr<float> end = begin + pg.size;
    total_norm_sq +=
        thrust::transform_reduce(thrust::device, begin, end, SquareOp(), 0.0f,
                                 thrust::plus<float>());
  }
  float total_norm = sqrtf(total_norm_sq);

  if (total_norm > max_norm) {
    float clip_coeff = max_norm / (total_norm + 1e-6f);
    for (auto &pg : all_groups) {
      int blocks = (pg.size + 255) / 256;
      clipGradsKernel<<<blocks, 256>>>(pg.grads, clip_coeff, pg.size);
    }
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
