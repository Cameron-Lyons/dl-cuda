#pragma once

#include "dl_cuda/nn.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace dlcuda {

class Optimizer {
public:
  virtual ~Optimizer() = default;

  virtual Status ZeroGrad(RuntimeContext &ctx, const std::vector<ParameterRef> &params) = 0;
  virtual Status Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float lr) = 0;
};

class SGDOptimizer : public Optimizer {
public:
  Status ZeroGrad(RuntimeContext &ctx, const std::vector<ParameterRef> &params) override;
  Status Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float lr) override;
};

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
      : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

  Status ZeroGrad(RuntimeContext &ctx, const std::vector<ParameterRef> &params) override;
  Status Step(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float lr) override;

private:
  Status EnsureState(RuntimeContext &ctx, const std::vector<ParameterRef> &params);

  float beta1_;
  float beta2_;
  float epsilon_;
  float beta1_power_ = 1.0f;
  float beta2_power_ = 1.0f;

  std::unordered_map<const Tensor *, Tensor> m_state_;
  std::unordered_map<const Tensor *, Tensor> v_state_;
};

} // namespace dlcuda
