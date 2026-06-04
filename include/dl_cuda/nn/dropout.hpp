#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class Dropout : public Module {
public:
  explicit Dropout(float probability = 0.5f, uint64_t seed = 0ULL);

  void SetTraining(bool training);
  [[nodiscard]] bool training() const {
    return training_;
  }
  [[nodiscard]] float probability() const {
    return probability_;
  }

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  float probability_ = 0.5f;
  bool training_ = true;
  bool last_training_ = false;
  uint64_t seed_ = 0ULL;
  uint64_t call_index_ = 0ULL;
  Tensor mask_;
  Tensor forward_output_;
  Tensor backward_output_;
};

} // namespace dlcuda
