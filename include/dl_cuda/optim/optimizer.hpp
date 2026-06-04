#pragma once

#include "dl_cuda/nn/module.hpp"
#include "dl_cuda/optim/schedulers.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace dlcuda {

struct OptimizerParamGroup {
  std::vector<std::string> parameter_names;
  float lr = 1e-3f;
  float weight_decay = 0.0f;
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

} // namespace dlcuda
