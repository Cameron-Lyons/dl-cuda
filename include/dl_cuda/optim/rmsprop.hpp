#pragma once

#include "dl_cuda/optim/optimizer.hpp"

#include <unordered_map>

namespace dlcuda {

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

} // namespace dlcuda
