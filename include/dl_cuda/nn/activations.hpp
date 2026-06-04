#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class ReLU : public Module {
public:
  ReLU() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
};

class GELU : public Module {
public:
  GELU() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
};

class Sigmoid : public Module {
public:
  Sigmoid() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Tensor cached_output_;
  Tensor backward_output_;
};

class Softmax : public Module {
public:
  Softmax() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  int64_t num_rows_ = 0;
  int64_t row_width_ = 0;
  Tensor cached_output_;
  Tensor backward_output_;
};

} // namespace dlcuda
