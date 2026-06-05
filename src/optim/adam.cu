#include "detail/adam_kernels.cuh"

namespace dlcuda {
namespace {

constexpr int kAdamMultiTensorMaxBlocksPerParam = 256;

} // namespace

AdamOptimizer::AdamOptimizer(float beta1, float beta2, float epsilon)
    : Optimizer(), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

AdamOptimizer::AdamOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1,
                             float beta2, float epsilon)
    : AdamOptimizer(std::move(param_groups), beta1, beta2, epsilon, false) {}

AdamOptimizer::AdamOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1,
                             float beta2, float epsilon, bool decoupled_weight_decay)
    : Optimizer(std::move(param_groups)), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      decoupled_weight_decay_(decoupled_weight_decay) {}

Status AdamOptimizer::ValidateHyperparameters() const {
  DLCUDA_RETURN_IF_ERROR(ValidateRate(beta1_, "Adam beta1"));
  DLCUDA_RETURN_IF_ERROR(ValidateRate(beta2_, "Adam beta2"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(epsilon_, "Adam epsilon"));
  return Status::Ok();
}

void AdamOptimizer::CollectHyperparameters(std::vector<Hyperparameter> *out) const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->push_back(Hyperparameter{"beta1", beta1_});
  out->push_back(Hyperparameter{"beta2", beta2_});
  out->push_back(Hyperparameter{"epsilon", epsilon_});
}

Status AdamOptimizer::EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params) {
  DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &m_state_));
  DLCUDA_RETURN_IF_ERROR(EnsureStateMap(ctx, params, &v_state_));
  return Status::Ok();
}

Status AdamOptimizer::StepImpl(RuntimeContext &ctx,
                               const std::vector<ResolvedOptimizerParam> &params,
                               int64_t step_index) {
  float beta1_power = std::pow(beta1_, static_cast<float>(step_index));
  float beta2_power = std::pow(beta2_, static_cast<float>(step_index));
  float inv_bias_correction1 = 1.0f / (1.0f - beta1_power);
  float inv_bias_correction2 = 1.0f / (1.0f - beta2_power);

  if (params.size() == 1) {
    const ParameterRef &param = *params.front().param;
    Tensor &m = m_state_.at(param.value);
    Tensor &v = v_state_.at(param.value);

    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchAdamUpdate(ctx, param, &m, &v, params.front().lr, beta1_, beta2_,
                                              epsilon_, inv_bias_correction1, inv_bias_correction2,
                                              params.front().weight_decay, decoupled_weight_decay_,
                                              blocks.value()));
    }
    return Status::Ok();
  }

  std::vector<AdamUpdateBlock> update_blocks;
  for (const auto &resolved : params) {
    const ParameterRef &param = *resolved.param;
    Tensor &m = m_state_.at(param.value);
    Tensor &v = v_state_.at(param.value);

    auto blocks = detail::BlocksForElements(param.value->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > kAdamMultiTensorMaxBlocksPerParam) {
      DLCUDA_RETURN_IF_ERROR(LaunchAdamUpdate(
          ctx, param, &m, &v, resolved.lr, beta1_, beta2_, epsilon_, inv_bias_correction1,
          inv_bias_correction2, resolved.weight_decay, decoupled_weight_decay_, blocks.value()));
      continue;
    }

    int64_t n = param.value->numel();
    for (int block = 0; block < blocks.value(); ++block) {
      int64_t start = static_cast<int64_t>(block) * kOptimizerThreads;
      update_blocks.push_back(AdamUpdateBlock{
          param.value->data(), param.grad->data(), m.data_as<float>(), v.data_as<float>(), start, n,
          param.value->dtype(), param.grad->dtype(), resolved.lr, resolved.weight_decay});
    }
  }

  if (update_blocks.empty()) {
    return Status::Ok();
  }

  size_t descriptor_bytes = update_blocks.size() * sizeof(AdamUpdateBlock);
  auto descriptor_tensor = ScratchTensorForBytes(ctx, "optim.adam.update_blocks", descriptor_bytes);
  if (!descriptor_tensor.ok()) {
    return descriptor_tensor.status();
  }
  Tensor descriptor_buffer = descriptor_tensor.value();
  DLCUDA_RETURN_IF_ERROR(
      descriptor_buffer.CopyFromHost(update_blocks.data(), descriptor_bytes, ctx.stream()));

  auto rows = detail::RowsForGrid(static_cast<int64_t>(update_blocks.size()), "Adam update");
  if (!rows.ok()) {
    return rows.status();
  }
  return LaunchAdamUpdateBlocks(ctx, &descriptor_buffer, rows.value(), beta1_, beta2_, epsilon_,
                                inv_bias_correction1, inv_bias_correction2,
                                decoupled_weight_decay_);
}

Status AdamOptimizer::CollectStateTensors(const std::vector<ParameterRef> &params,
                                          std::vector<StateTensorRef> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("Adam state destination is null");
  }
  out->clear();
  out->reserve(params.size() * 2);
  for (const auto &param : params) {
    out->push_back(StateTensorRef{StateName(param, "m"), &m_state_.at(param.value)});
    out->push_back(StateTensorRef{StateName(param, "v"), &v_state_.at(param.value)});
  }
  return Status::Ok();
}

AdamWOptimizer::AdamWOptimizer(float lr, float weight_decay, float beta1, float beta2,
                               float epsilon)
    : AdamOptimizer(std::vector<OptimizerParamGroup>{{{}, lr, weight_decay}}, beta1, beta2, epsilon,
                    true) {}

AdamWOptimizer::AdamWOptimizer(std::vector<OptimizerParamGroup> param_groups, float beta1,
                               float beta2, float epsilon)
    : AdamOptimizer(std::move(param_groups), beta1, beta2, epsilon, true) {}

} // namespace dlcuda
