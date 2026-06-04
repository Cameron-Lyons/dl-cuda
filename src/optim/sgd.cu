#include "detail/sgd_kernels.cuh"

namespace dlcuda {

SGDOptimizer::SGDOptimizer(float lr, float momentum, float weight_decay, float dampening,
                           bool nesterov)
    : Optimizer(lr, weight_decay), momentum_(momentum), dampening_(dampening), nesterov_(nesterov) {
}

SGDOptimizer::SGDOptimizer(std::vector<OptimizerParamGroup> param_groups, float momentum,
                           float dampening, bool nesterov)
    : Optimizer(std::move(param_groups)), momentum_(momentum), dampening_(dampening),
      nesterov_(nesterov) {}

Status SGDOptimizer::ValidateHyperparameters() const {
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(momentum_, "SGD momentum"));
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(dampening_, "SGD dampening"));
  if (nesterov_ && (momentum_ <= 0.0f || dampening_ != 0.0f)) {
    return Status::InvalidArgument("SGD nesterov requires momentum > 0 and dampening == 0");
  }
  return Status::Ok();
}

void SGDOptimizer::CollectHyperparameters(std::vector<Hyperparameter> *out) const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->push_back(Hyperparameter{"momentum", momentum_});
  out->push_back(Hyperparameter{"dampening", dampening_});
  out->push_back(Hyperparameter{"nesterov", nesterov_ ? 1.0f : 0.0f});
}

Status SGDOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  if (momentum_ == 0.0f) {
    ClearStateMap(&momentum_state_);
    for (const auto &param : params) {
      DLCUDA_RETURN_IF_ERROR(ValidateParameterOnly(param, "SGD"));
    }
    return Status::Ok();
  }
  return EnsureStateMap(ctx, params, &momentum_state_);
}

Status SGDOptimizer::StepImpl(RuntimeContext &ctx,
                              const std::vector<ResolvedOptimizerParam> &params,
                              int64_t step_index) {
  (void)step_index;
  bool has_momentum = momentum_ != 0.0f;
  for (const auto &resolved : params) {
    const ParameterRef &param = *resolved.param;
    Tensor *momentum_buffer = has_momentum ? &momentum_state_.at(param.value) : nullptr;
    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchSGDUpdate(ctx, param, momentum_buffer, has_momentum, resolved.lr,
                                             momentum_, resolved.weight_decay, dampening_,
                                             nesterov_, blocks.value()));
    }
  }
  return Status::Ok();
}

Status SGDOptimizer::CollectStateTensors(const std::vector<ParameterRef> &params,
                                         std::vector<StateTensorRef> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("SGD state destination is null");
  }
  out->clear();
  if (momentum_ == 0.0f) {
    return Status::Ok();
  }
  out->reserve(params.size());
  for (const auto &param : params) {
    out->push_back(StateTensorRef{StateName(param, "momentum"), &momentum_state_.at(param.value)});
  }
  return Status::Ok();
}

} // namespace dlcuda
