#pragma once

#include "dl_cuda/nn/module.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <vector>

namespace dlcuda {

class LearningRateScheduler;
class Optimizer;
class RuntimeContext;

struct TrainStepOptions {
  float learning_rate = 1e-3f;
  float max_grad_norm = 0.0f;
  bool compute_metrics = true;
  bool use_optimizer_learning_rates = false;
  const LearningRateScheduler *scheduler = nullptr;
};

struct TrainStepResult {
  float loss = 0.0f;
  float accuracy = 0.0f;
  bool has_accuracy = false;
  float grad_norm = 0.0f;
};

class SupervisedTrainer {
public:
  SupervisedTrainer(RuntimeContext &ctx, Module &model, Optimizer &optimizer,
                    std::vector<ParameterRef> params);

  Status TrainBinaryClassificationStep(const Tensor &inputs, const Tensor &targets,
                                       const TrainStepOptions &options = TrainStepOptions(),
                                       TrainStepResult *result = nullptr);

  Status TrainCategoricalClassificationStep(const Tensor &inputs, const Tensor &target_ids,
                                            const TrainStepOptions &options = TrainStepOptions(),
                                            TrainStepResult *result = nullptr);

  [[nodiscard]] const std::vector<ParameterRef> &parameters() const {
    return params_;
  }

private:
  enum class ClassificationKind {
    kBinary,
    kCategorical,
  };

  Status ValidateOptions(const TrainStepOptions &options) const;
  Status TrainClassificationStep(const Tensor &inputs, const Tensor &labels,
                                 const TrainStepOptions &options, TrainStepResult *result,
                                 ClassificationKind kind);
  Status FinishStep(const TrainStepOptions &options, TrainStepResult *result);

  RuntimeContext *ctx_ = nullptr;
  Module *model_ = nullptr;
  Optimizer *optimizer_ = nullptr;
  std::vector<ParameterRef> params_;
  Tensor outputs_;
  Tensor loss_grad_;
};

} // namespace dlcuda
