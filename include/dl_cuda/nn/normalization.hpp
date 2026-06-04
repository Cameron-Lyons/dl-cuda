#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class LayerNorm : public Module {
public:
  LayerNorm(int64_t normalized_size, RuntimeContext &ctx, float eps = 1e-5f,
            DType dtype = DType::kFloat32);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t normalized_size_ = 0;
  int64_t last_rows_ = 0;
  float eps_ = 1e-5f;
  DType dtype_ = DType::kFloat32;
  Tensor gamma_;
  Tensor beta_;
  Tensor grad_gamma_;
  Tensor grad_beta_;
  Tensor cached_x_hat_;
  Tensor inv_std_;
  Tensor forward_output_;
  Tensor backward_output_;
};

class BatchNorm1d : public Module {
public:
  BatchNorm1d(int64_t features, RuntimeContext &ctx, float eps = 1e-5f, float momentum = 0.1f,
              DType dtype = DType::kFloat32);

  void SetTraining(bool training);
  [[nodiscard]] bool training() const {
    return training_;
  }

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t features_ = 0;
  int64_t last_batch_ = 0;
  float eps_ = 1e-5f;
  float momentum_ = 0.1f;
  bool training_ = true;
  bool last_training_ = true;
  DType dtype_ = DType::kFloat32;
  Tensor gamma_;
  Tensor beta_;
  Tensor grad_gamma_;
  Tensor grad_beta_;
  Tensor running_mean_;
  Tensor running_var_;
  Tensor cached_x_hat_;
  Tensor inv_std_;
  Tensor forward_output_;
  Tensor backward_output_;
};

} // namespace dlcuda
