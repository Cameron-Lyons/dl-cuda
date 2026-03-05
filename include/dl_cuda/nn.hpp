#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dlcuda {

struct ParameterRef {
  std::string name;
  Tensor *value = nullptr;
  Tensor *grad = nullptr;
};

class Module {
public:
  virtual ~Module() = default;

  virtual Status Forward(RuntimeContext &ctx, const Tensor &input,
                         Tensor *output) = 0;
  virtual Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                          Tensor *grad_input) = 0;
  virtual std::vector<ParameterRef> Parameters() = 0;
  virtual std::string Name() const = 0;
};

class Sequential : public Module {
public:
  Sequential() = default;

  Status Add(std::unique_ptr<Module> module);

  Status Forward(RuntimeContext &ctx, const Tensor &input,
                 Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                  Tensor *grad_input) override;

  std::vector<ParameterRef> Parameters() override;
  std::string Name() const override { return "Sequential"; }

  size_t size() const { return modules_.size(); }

private:
  std::vector<std::unique_ptr<Module>> modules_;
};

class Linear : public Module {
public:
  Linear(int64_t in_features, int64_t out_features, RuntimeContext &ctx);

  Status Forward(RuntimeContext &ctx, const Tensor &input,
                 Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                  Tensor *grad_input) override;

  std::vector<ParameterRef> Parameters() override;
  std::string Name() const override { return "Linear"; }

private:
  Status init_status_;
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;
  int64_t last_batch_ = 0;
  Tensor weight_;
  Tensor bias_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  Tensor cached_input_;
};

class ReLU : public Module {
public:
  ReLU() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input,
                 Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                  Tensor *grad_input) override;

  std::vector<ParameterRef> Parameters() override { return {}; }
  std::string Name() const override { return "ReLU"; }

private:
  Tensor cached_input_;
};

class Sigmoid : public Module {
public:
  Sigmoid() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input,
                 Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                  Tensor *grad_input) override;

  std::vector<ParameterRef> Parameters() override { return {}; }
  std::string Name() const override { return "Sigmoid"; }

private:
  Tensor cached_output_;
};

class Softmax : public Module {
public:
  Softmax() = default;

  Status Forward(RuntimeContext &ctx, const Tensor &input,
                 Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                  Tensor *grad_input) override;

  std::vector<ParameterRef> Parameters() override { return {}; }
  std::string Name() const override { return "Softmax"; }

private:
  int64_t num_rows_ = 0;
  int64_t row_width_ = 0;
  Tensor cached_output_;
};

class Embedding : public Module {
public:
  Embedding(int64_t vocab_size, int64_t embedding_dim, RuntimeContext &ctx);

  Status Forward(RuntimeContext &ctx, const Tensor &input,
                 Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output,
                  Tensor *grad_input) override;

  std::vector<ParameterRef> Parameters() override;
  std::string Name() const override { return "Embedding"; }

private:
  Status init_status_;
  int64_t vocab_size_ = 0;
  int64_t embedding_dim_ = 0;
  int64_t last_num_tokens_ = 0;
  Tensor table_;
  Tensor grad_table_;
  Tensor cached_token_ids_;
};

} // namespace dlcuda
