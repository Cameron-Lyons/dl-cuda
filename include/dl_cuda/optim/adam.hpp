#pragma once

#include "dl_cuda/optim/optimizer.hpp"

#include <unordered_map>

namespace dlcuda {

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

} // namespace dlcuda
