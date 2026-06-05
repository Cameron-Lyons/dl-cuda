#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dlcuda {

class Module;
class GradientTape;

class AutoTensor {
public:
  AutoTensor() = default;

  [[nodiscard]] bool defined() const;
  [[nodiscard]] int64_t id() const {
    return id_;
  }
  [[nodiscard]] bool requires_grad() const {
    return requires_grad_;
  }
  [[nodiscard]] const Tensor &value() const {
    return value_;
  }

  Status Backward(RuntimeContext &ctx) const;
  Status Backward(RuntimeContext &ctx, const Tensor &initial_gradient) const;
  Result<Tensor> grad() const;

private:
  friend class GradientTape;

  AutoTensor(GradientTape *tape, int64_t id, Tensor value, bool requires_grad);

  GradientTape *tape_ = nullptr;
  int64_t id_ = -1;
  Tensor value_;
  bool requires_grad_ = false;
};

using CustomAutogradForward =
    std::function<Status(RuntimeContext &ctx, const std::vector<Tensor> &inputs, Tensor *output)>;
using CustomAutogradBackward = std::function<Status(
    RuntimeContext &ctx, const Tensor &output_grad, const std::vector<Tensor> &inputs,
    const Tensor &output, std::vector<Tensor> *input_grads)>;

class GradientTape {
public:
  GradientTape() = default;
  GradientTape(const GradientTape &) = delete;
  GradientTape &operator=(const GradientTape &) = delete;

  AutoTensor Variable(const Tensor &value, bool requires_grad = true);
  AutoTensor Constant(const Tensor &value);

  Result<AutoTensor> Add(RuntimeContext &ctx, const AutoTensor &lhs, const AutoTensor &rhs);
  Result<AutoTensor> Subtract(RuntimeContext &ctx, const AutoTensor &lhs, const AutoTensor &rhs);
  Result<AutoTensor> Multiply(RuntimeContext &ctx, const AutoTensor &lhs, const AutoTensor &rhs);
  Result<AutoTensor> Divide(RuntimeContext &ctx, const AutoTensor &lhs, const AutoTensor &rhs);
  Result<AutoTensor> MatMul(RuntimeContext &ctx, const AutoTensor &lhs, const AutoTensor &rhs);
  Result<AutoTensor> ReduceSum(RuntimeContext &ctx, const AutoTensor &input);
  Result<AutoTensor> Relu(RuntimeContext &ctx, const AutoTensor &input);
  Result<AutoTensor> Sigmoid(RuntimeContext &ctx, const AutoTensor &input);

  Result<AutoTensor> ApplyModule(RuntimeContext &ctx, Module &module, const AutoTensor &input);

  Status RegisterCustomOp(std::string name, CustomAutogradForward forward,
                          CustomAutogradBackward backward);
  Result<AutoTensor> ApplyCustomOp(RuntimeContext &ctx, const std::string &name,
                                   const std::vector<AutoTensor> &inputs);

  Status Backward(RuntimeContext &ctx, const AutoTensor &target);
  Status Backward(RuntimeContext &ctx, const AutoTensor &target, const Tensor &initial_gradient);
  Result<Tensor> Gradient(const AutoTensor &tensor) const;

  void ClearGradients();
  void Reset();

  [[nodiscard]] size_t node_count() const {
    return nodes_.size();
  }

private:
  struct Node {
    int64_t output_id = -1;
    std::string op_name;
    std::function<Status(RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad)>
        backward;
  };

  struct CustomOp {
    CustomAutogradForward forward;
    CustomAutogradBackward backward;
  };

  AutoTensor CreateTensor(Tensor value, bool requires_grad);
  Status ValidateTensor(const AutoTensor &tensor, const char *name) const;
  Status AccumulateReducedGradient(RuntimeContext &ctx, int64_t id, const Tensor &grad,
                                   const std::vector<int64_t> &input_shape, float scale = 1.0f);
  Status AccumulateProductGradient(RuntimeContext &ctx, int64_t id, const Tensor &output_grad,
                                   const Tensor &factor, const std::vector<int64_t> &input_shape);
  Status AccumulateQuotientGradient(RuntimeContext &ctx, int64_t id, const Tensor &output_grad,
                                    const Tensor &divisor, const std::vector<int64_t> &input_shape);
  Status AccumulateDivideRhsGradient(RuntimeContext &ctx, int64_t id, const Tensor &output_grad,
                                     const Tensor &lhs, const Tensor &rhs,
                                     const std::vector<int64_t> &rhs_shape);
  Status AccumulateGradient(RuntimeContext &ctx, int64_t id, const Tensor &grad,
                            float scale = 1.0f);
  void RecordNode(Node node);

  int64_t next_id_ = 1;
  std::vector<Node> nodes_;
  std::unordered_map<int64_t, Tensor> gradients_;
  std::unordered_set<int64_t> owned_gradient_ids_;
  std::unordered_map<std::string, CustomOp> custom_ops_;
};

} // namespace dlcuda
