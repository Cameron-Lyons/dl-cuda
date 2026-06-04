#pragma once

#include "dl_cuda/nn.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"

#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dlcuda {

struct OptimizerParamGroup {
  std::vector<std::string> parameter_names;
  float lr = 1e-3f;
  float weight_decay = 0.0f;
};

class LearningRateScheduler {
public:
  virtual ~LearningRateScheduler() = default;

  virtual Result<float> LearningRate(int64_t step_index, float base_lr) const = 0;
};

class ConstantLRScheduler : public LearningRateScheduler {
public:
  Result<float> LearningRate(int64_t step_index, float base_lr) const override;
};

class StepLRScheduler : public LearningRateScheduler {
public:
  StepLRScheduler(int64_t step_size, float gamma) : step_size_(step_size), gamma_(gamma) {}

  Result<float> LearningRate(int64_t step_index, float base_lr) const override;

private:
  int64_t step_size_ = 1;
  float gamma_ = 1.0f;
};

class ExponentialLRScheduler : public LearningRateScheduler {
public:
  explicit ExponentialLRScheduler(float gamma) : gamma_(gamma) {}

  Result<float> LearningRate(int64_t step_index, float base_lr) const override;

private:
  float gamma_ = 1.0f;
};

class CosineAnnealingLRScheduler : public LearningRateScheduler {
public:
  CosineAnnealingLRScheduler(int64_t max_steps, float min_lr = 0.0f)
      : max_steps_(max_steps), min_lr_(min_lr) {}

  Result<float> LearningRate(int64_t step_index, float base_lr) const override;

private:
  int64_t max_steps_ = 1;
  float min_lr_ = 0.0f;
};

struct ResolvedOptimizerParam {
  const ParameterRef *param = nullptr;
  float lr = 1e-3f;
  float weight_decay = 0.0f;
};

class Optimizer {
public:
  explicit Optimizer(float lr = 1e-3f, float weight_decay = 0.0f);
  explicit Optimizer(std::vector<OptimizerParamGroup> param_groups);
  virtual ~Optimizer() = default;

  [[nodiscard]] virtual const char *Name() const = 0;

  Status ZeroGrad(RuntimeContext &ctx, const std::vector<ParameterRef> &params);
  Status Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params);
  Status Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float lr);
  Status Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params,
              const LearningRateScheduler &scheduler);

  Status SetParameterGroups(std::vector<OptimizerParamGroup> param_groups);
  [[nodiscard]] const std::vector<OptimizerParamGroup> &parameter_groups() const {
    return param_groups_;
  }

  [[nodiscard]] int64_t step_count() const {
    return step_count_;
  }

  Status SaveCheckpoint(RuntimeContext &ctx, const std::string &path,
                        const std::vector<ParameterRef> &params);
  Status SaveCheckpoint(RuntimeContext &ctx, FILE *file, const std::vector<ParameterRef> &params);
  Status LoadCheckpoint(RuntimeContext &ctx, const std::string &path,
                        const std::vector<ParameterRef> &params);
  Status LoadCheckpoint(RuntimeContext &ctx, FILE *file, const std::vector<ParameterRef> &params);

  struct StateTensorRef {
    std::string name;
    Tensor *tensor = nullptr;
  };

  struct Hyperparameter {
    std::string name;
    float value = 0.0f;
  };

protected:
  virtual Status ValidateHyperparameters() const = 0;
  virtual void CollectHyperparameters(std::vector<Hyperparameter> *out) const = 0;
  virtual Status EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) = 0;
  virtual Status StepImpl(RuntimeContext &ctx, const std::vector<ResolvedOptimizerParam> &params,
                          int64_t step_index) = 0;
  virtual Status CollectStateTensors(const std::vector<ParameterRef> &params,
                                     std::vector<StateTensorRef> *out) = 0;

private:
  Result<std::vector<ResolvedOptimizerParam>>
  ResolveParameterGroups(const std::vector<ParameterRef> &params, const float *lr_override,
                         const LearningRateScheduler *scheduler) const;

  std::vector<OptimizerParamGroup> param_groups_;
  int64_t step_count_ = 0;
};

class SGDOptimizer : public Optimizer {
public:
  SGDOptimizer(float lr = 1e-2f, float momentum = 0.0f, float weight_decay = 0.0f,
               float dampening = 0.0f, bool nesterov = false);
  SGDOptimizer(std::vector<OptimizerParamGroup> param_groups, float momentum = 0.0f,
               float dampening = 0.0f, bool nesterov = false);

  [[nodiscard]] const char *Name() const override {
    return "SGD";
  }

protected:
  Status ValidateHyperparameters() const override;
  void CollectHyperparameters(std::vector<Hyperparameter> *out) const override;
  Status EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) override;
  Status StepImpl(RuntimeContext &ctx, const std::vector<ResolvedOptimizerParam> &params,
                  int64_t step_index) override;
  Status CollectStateTensors(const std::vector<ParameterRef> &params,
                             std::vector<StateTensorRef> *out) override;

private:
  float momentum_ = 0.0f;
  float dampening_ = 0.0f;
  bool nesterov_ = false;
  std::unordered_map<const Tensor *, Tensor> momentum_state_;
};

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
  AdamOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1 = 0.9f,
                float beta2 = 0.999f, float epsilon = 1e-8f);

  [[nodiscard]] const char *Name() const override {
    return decoupled_weight_decay_ ? "AdamW" : "Adam";
  }

protected:
  AdamOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1, float beta2,
                float epsilon, bool decoupled_weight_decay);

  Status ValidateHyperparameters() const override;
  void CollectHyperparameters(std::vector<Hyperparameter> *out) const override;
  Status EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) override;
  Status StepImpl(RuntimeContext &ctx, const std::vector<ResolvedOptimizerParam> &params,
                  int64_t step_index) override;
  Status CollectStateTensors(const std::vector<ParameterRef> &params,
                             std::vector<StateTensorRef> *out) override;

private:
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float epsilon_ = 1e-8f;
  bool decoupled_weight_decay_ = false;
  std::unordered_map<const Tensor *, Tensor> m_state_;
  std::unordered_map<const Tensor *, Tensor> v_state_;
};

class AdamWOptimizer : public AdamOptimizer {
public:
  AdamWOptimizer(float lr = 1e-3f, float weight_decay = 1e-2f, float beta1 = 0.9f,
                 float beta2 = 0.999f, float epsilon = 1e-8f);
  AdamWOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1 = 0.9f,
                 float beta2 = 0.999f, float epsilon = 1e-8f);
};

class RMSPropOptimizer : public Optimizer {
public:
  RMSPropOptimizer(float lr = 1e-2f, float alpha = 0.99f, float epsilon = 1e-8f,
                   float momentum = 0.0f, float weight_decay = 0.0f, bool centered = false);
  RMSPropOptimizer(std::vector<OptimizerParamGroup> param_groups, float alpha = 0.99f,
                   float epsilon = 1e-8f, float momentum = 0.0f, bool centered = false);

  [[nodiscard]] const char *Name() const override {
    return "RMSProp";
  }

protected:
  Status ValidateHyperparameters() const override;
  void CollectHyperparameters(std::vector<Hyperparameter> *out) const override;
  Status EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) override;
  Status StepImpl(RuntimeContext &ctx, const std::vector<ResolvedOptimizerParam> &params,
                  int64_t step_index) override;
  Status CollectStateTensors(const std::vector<ParameterRef> &params,
                             std::vector<StateTensorRef> *out) override;

private:
  float alpha_ = 0.99f;
  float epsilon_ = 1e-8f;
  float momentum_ = 0.0f;
  bool centered_ = false;
  std::unordered_map<const Tensor *, Tensor> square_avg_state_;
  std::unordered_map<const Tensor *, Tensor> momentum_state_;
  std::unordered_map<const Tensor *, Tensor> grad_avg_state_;
};

Status ClipGradNorm(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float max_norm,
                    float *total_norm = nullptr);

} // namespace dlcuda
