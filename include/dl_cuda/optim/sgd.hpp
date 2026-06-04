#pragma once

#include "dl_cuda/optim/optimizer.hpp"

#include <unordered_map>

namespace dlcuda {

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

} // namespace dlcuda
