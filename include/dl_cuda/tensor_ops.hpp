#pragma once

#include "dl_cuda/runtime.hpp"
#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

namespace dlcuda {

enum class TensorBinaryOp {
  kAdd = 0,
  kSubtract = 1,
  kMultiply = 2,
  kDivide = 3,
};

Status TensorAdd(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output);
Status TensorSubtract(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output);
Status TensorMultiply(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output);
Status TensorDivide(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output);

Status TensorMatMul(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output);

Status TensorReduceSum(RuntimeContext &ctx, const Tensor &input, Tensor *output);
Result<float> TensorSum(RuntimeContext &ctx, const Tensor &input);

} // namespace dlcuda
