#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dlcuda {

struct ParameterRef {
  std::string name;
  Tensor *value = nullptr;
  Tensor *grad = nullptr;
};

// Modules keep explicit kernels and parameter ownership. For graph-based automatic
// differentiation, wrap modules with GradientTape::ApplyModule from autograd.hpp.
class Module {
public:
  virtual ~Module() = default;

  virtual Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) = 0;
  // grad_input may be null when the caller does not need gradients with respect to module input.
  virtual Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) = 0;
  virtual void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) = 0;
};

class Sequential : public Module {
public:
  Sequential() = default;

  Status Add(std::unique_ptr<Module> module);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;
  [[nodiscard]] const std::vector<ParameterRef> &parameters() const {
    return parameter_cache_;
  }

private:
  void RebuildParameterCache();

  std::vector<std::unique_ptr<Module>> modules_;
  std::vector<ParameterRef> parameter_cache_;
};

class Residual : public Module {
public:
  explicit Residual(std::unique_ptr<Module> branch);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  std::unique_ptr<Module> branch_;
  Tensor branch_output_;
  Tensor forward_output_;
  Tensor branch_grad_;
  Tensor backward_output_;
};

} // namespace dlcuda
