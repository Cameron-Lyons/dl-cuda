#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class Linear : public Module {
public:
  Linear(int64_t in_features, int64_t out_features, RuntimeContext &ctx,
         DType dtype = DType::kFloat32);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;
  int64_t last_batch_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor weight_;
  Tensor bias_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
};

} // namespace dlcuda
