#include "dl_cuda/trainer.hpp"

#include "dl_cuda/loss.hpp"
#include "dl_cuda/optim/clip_grad.hpp"
#include "dl_cuda/optim/optimizer.hpp"

#include <cmath>
#include <utility>

namespace dlcuda {

SupervisedTrainer::SupervisedTrainer(RuntimeContext &ctx, Module &model, Optimizer &optimizer,
                                     std::vector<ParameterRef> params)
    : ctx_(&ctx), model_(&model), optimizer_(&optimizer), params_(std::move(params)) {}

Status SupervisedTrainer::ValidateOptions(const TrainStepOptions &options) const {
  if (ctx_ == nullptr || model_ == nullptr || optimizer_ == nullptr) {
    return Status::InvalidArgument("SupervisedTrainer is not initialized");
  }
  if (params_.empty()) {
    return Status::InvalidArgument("SupervisedTrainer requires at least one parameter");
  }
  if (options.scheduler == nullptr && !options.use_optimizer_learning_rates &&
      (!std::isfinite(options.learning_rate) || !(options.learning_rate > 0.0f))) {
    return Status::InvalidArgument("TrainStepOptions learning_rate must be finite and > 0");
  }
  if (!std::isfinite(options.max_grad_norm) || options.max_grad_norm < 0.0f) {
    return Status::InvalidArgument("TrainStepOptions max_grad_norm must be finite and >= 0");
  }
  return Status::Ok();
}

Status SupervisedTrainer::FinishStep(const TrainStepOptions &options, TrainStepResult *result) {
  if (options.max_grad_norm > 0.0f) {
    float grad_norm = 0.0f;
    DLCUDA_RETURN_IF_ERROR(ClipGradNorm(*ctx_, params_, options.max_grad_norm, &grad_norm));
    if (result != nullptr) {
      result->grad_norm = grad_norm;
    }
  }

  if (options.scheduler != nullptr) {
    return optimizer_->Step(*ctx_, params_, *options.scheduler);
  }
  if (options.use_optimizer_learning_rates) {
    return optimizer_->Step(*ctx_, params_);
  }
  return optimizer_->Step(*ctx_, params_, options.learning_rate);
}

Status SupervisedTrainer::TrainBinaryClassificationStep(const Tensor &inputs, const Tensor &targets,
                                                        const TrainStepOptions &options,
                                                        TrainStepResult *result) {
  DLCUDA_RETURN_IF_ERROR(ValidateOptions(options));
  if (result != nullptr) {
    *result = TrainStepResult();
  }

  DLCUDA_RETURN_IF_ERROR(optimizer_->ZeroGrad(*ctx_, params_));
  DLCUDA_RETURN_IF_ERROR(model_->Forward(*ctx_, inputs, &outputs_));

  if (result != nullptr && options.compute_metrics) {
    auto loss = BinaryCrossEntropyLoss(*ctx_, targets, outputs_);
    if (!loss.ok()) {
      return loss.status();
    }
    result->loss = loss.value();
  }

  DLCUDA_RETURN_IF_ERROR(BinaryCrossEntropyBackward(*ctx_, targets, outputs_, &loss_grad_));
  DLCUDA_RETURN_IF_ERROR(model_->Backward(*ctx_, loss_grad_, nullptr));
  return FinishStep(options, result);
}

Status SupervisedTrainer::TrainCategoricalClassificationStep(const Tensor &inputs,
                                                             const Tensor &target_ids,
                                                             const TrainStepOptions &options,
                                                             TrainStepResult *result) {
  DLCUDA_RETURN_IF_ERROR(ValidateOptions(options));
  if (result != nullptr) {
    *result = TrainStepResult();
  }

  DLCUDA_RETURN_IF_ERROR(optimizer_->ZeroGrad(*ctx_, params_));
  DLCUDA_RETURN_IF_ERROR(model_->Forward(*ctx_, inputs, &outputs_));

  if (result != nullptr && options.compute_metrics) {
    auto metrics = CategoricalCrossEntropyMetricsFromLogits(*ctx_, target_ids, outputs_);
    if (!metrics.ok()) {
      return metrics.status();
    }
    result->loss = metrics.value().loss;
    result->accuracy = metrics.value().accuracy;
    result->has_accuracy = true;
  }

  DLCUDA_RETURN_IF_ERROR(
      CategoricalCrossEntropyBackwardFromLogits(*ctx_, target_ids, outputs_, &loss_grad_));
  DLCUDA_RETURN_IF_ERROR(model_->Backward(*ctx_, loss_grad_, nullptr));
  return FinishStep(options, result);
}

} // namespace dlcuda
