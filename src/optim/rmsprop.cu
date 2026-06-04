#include "detail/rmsprop_kernels.cuh"

namespace dlcuda {

RMSPropOptimizer::RMSPropOptimizer(float lr, float alpha, float epsilon, float momentum,
                                   float weight_decay, bool centered)
    : Optimizer(lr, weight_decay), alpha_(alpha), epsilon_(epsilon), momentum_(momentum),
      centered_(centered) {}

RMSPropOptimizer::RMSPropOptimizer(std::vector<OptimizerParamGroup> param_groups, float alpha,
                                   float epsilon, float momentum, bool centered)
    : Optimizer(std::move(param_groups)), alpha_(alpha), epsilon_(epsilon), momentum_(momentum),
      centered_(centered) {}

Status RMSPropOptimizer::ValidateHyperparameters() const {
  DLCUDA_RETURN_IF_ERROR(ValidateRate(alpha_, "RMSProp alpha"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(epsilon_, "RMSProp epsilon"));
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(momentum_, "RMSProp momentum"));
  return Status::Ok();
}

void RMSPropOptimizer::CollectHyperparameters(std::vector<Hyperparameter> *out) const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->push_back(Hyperparameter{"alpha", alpha_});
  out->push_back(Hyperparameter{"epsilon", epsilon_});
  out->push_back(Hyperparameter{"momentum", momentum_});
  out->push_back(Hyperparameter{"centered", centered_ ? 1.0f : 0.0f});
}

Status RMSPropOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &square_avg_state_));
  if (momentum_ != 0.0f) {
    DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &momentum_state_));
  } else {
    ClearStateMap(&momentum_state_);
  }
  if (centered_) {
    DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &grad_avg_state_));
  } else {
    ClearStateMap(&grad_avg_state_);
  }
  return Status::Ok();
}

Status RMSPropOptimizer::StepImpl(RuntimeContext &ctx,
                                  const std::vector<ResolvedOptimizerParam> &params,
                                  int64_t step_index) {
  (void)step_index;
  bool has_momentum = momentum_ != 0.0f;
  for (const auto &resolved : params) {
    const ParameterRef &param = *resolved.param;
    Tensor &square_avg = square_avg_state_.at(param.value);
    Tensor *momentum_buffer = has_momentum ? &momentum_state_.at(param.value) : nullptr;
    Tensor *grad_avg = centered_ ? &grad_avg_state_.at(param.value) : nullptr;

    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchRMSPropUpdate(
          ctx, param, &square_avg, momentum_buffer, grad_avg, has_momentum, centered_, resolved.lr,
          alpha_, epsilon_, momentum_, resolved.weight_decay, blocks.value()));
    }
  }
  return Status::Ok();
}

Status RMSPropOptimizer::CollectStateTensors(const std::vector<ParameterRef> &params,
                                             std::vector<StateTensorRef> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("RMSProp state destination is null");
  }
  out->clear();
  size_t states_per_param = 1 + (momentum_ != 0.0f ? 1 : 0) + (centered_ ? 1 : 0);
  out->reserve(params.size() * states_per_param);
  for (const auto &param : params) {
    out->push_back(
        StateTensorRef{StateName(param, "square_avg"), &square_avg_state_.at(param.value)});
    if (momentum_ != 0.0f) {
      out->push_back(
          StateTensorRef{StateName(param, "momentum"), &momentum_state_.at(param.value)});
    }
    if (centered_) {
      out->push_back(
          StateTensorRef{StateName(param, "grad_avg"), &grad_avg_state_.at(param.value)});
    }
  }
  return Status::Ok();
}

} // namespace dlcuda
