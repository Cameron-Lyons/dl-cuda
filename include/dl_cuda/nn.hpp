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

class Conv2d : public Module {
public:
  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, RuntimeContext &ctx,
         DType dtype = DType::kFloat32);
  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
         RuntimeContext &ctx, int64_t stride_h = 1, int64_t stride_w = 1, int64_t padding_h = 0,
         int64_t padding_w = 0, DType dtype = DType::kFloat32);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t in_channels_ = 0;
  int64_t out_channels_ = 0;
  int64_t kernel_h_ = 0;
  int64_t kernel_w_ = 0;
  int64_t stride_h_ = 1;
  int64_t stride_w_ = 1;
  int64_t padding_h_ = 0;
  int64_t padding_w_ = 0;
  int64_t last_batch_ = 0;
  int64_t last_input_h_ = 0;
  int64_t last_input_w_ = 0;
  int64_t last_output_h_ = 0;
  int64_t last_output_w_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor weight_;
  Tensor bias_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
};

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

class MaxPool2d : public Module {
public:
  explicit MaxPool2d(int64_t kernel_size, int64_t stride = 0);
  MaxPool2d(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
            int64_t padding_h = 0, int64_t padding_w = 0);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t kernel_h_ = 0;
  int64_t kernel_w_ = 0;
  int64_t stride_h_ = 0;
  int64_t stride_w_ = 0;
  int64_t padding_h_ = 0;
  int64_t padding_w_ = 0;
  int64_t last_batch_ = 0;
  int64_t last_channels_ = 0;
  int64_t last_input_h_ = 0;
  int64_t last_input_w_ = 0;
  int64_t last_output_h_ = 0;
  int64_t last_output_w_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor argmax_indices_;
  Tensor forward_output_;
  Tensor backward_output_;
};

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

class Embedding : public Module {
public:
  Embedding(int64_t vocab_size, int64_t embedding_dim, RuntimeContext &ctx,
            DType dtype = DType::kFloat32);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t vocab_size_ = 0;
  int64_t embedding_dim_ = 0;
  int64_t last_num_tokens_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor table_;
  Tensor grad_table_;
  Tensor cached_token_ids_;
  Tensor forward_output_;
};

} // namespace dlcuda
