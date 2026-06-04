#include "dl_cuda/autograd.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/nn.hpp"
#include "dl_cuda/tensor_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dlcuda {
namespace {

constexpr int kAutogradThreads = 256;
constexpr int kMatMulTile = 16;
constexpr int kMaxBroadcastRank = 8;

struct BroadcastGradientDescriptor {
  int rank = 0;
  int64_t input_strides[kMaxBroadcastRank] = {};
  int64_t output_strides[kMaxBroadcastRank] = {};
};

Status ValidateFloat32Tensor(const Tensor &tensor, const char *name) {
  if (!tensor.defined()) {
    return Status::InvalidArgument(std::string(name) + " is undefined");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Status::Unsupported(std::string(name) +
                               " built-in autograd currently supports float32 tensors");
  }
  return Status::Ok();
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= shape[i - 1];
  }
  return strides;
}

Result<BroadcastGradientDescriptor>
BuildBroadcastGradientDescriptor(const std::vector<int64_t> &input_shape,
                                 const std::vector<int64_t> &output_shape, const char *op_name) {
  if (output_shape.size() > kMaxBroadcastRank) {
    std::ostringstream oss;
    oss << op_name << " supports broadcast rank up to " << kMaxBroadcastRank;
    return Status::InvalidArgument(oss.str());
  }
  if (input_shape.size() > output_shape.size()) {
    return Status::InvalidArgument(std::string(op_name) + " input rank exceeds output rank");
  }

  BroadcastGradientDescriptor desc;
  desc.rank = static_cast<int>(output_shape.size());

  std::vector<int64_t> input_strides = ContiguousStrides(input_shape);
  std::vector<int64_t> output_strides = ContiguousStrides(output_shape);
  for (size_t output_axis = 0; output_axis < output_shape.size(); ++output_axis) {
    int input_axis =
        static_cast<int>(output_axis) - static_cast<int>(output_shape.size() - input_shape.size());
    int64_t input_dim = input_axis >= 0 ? input_shape[static_cast<size_t>(input_axis)] : 1;
    int64_t output_dim = output_shape[output_axis];
    if (input_dim != output_dim && input_dim != 1) {
      std::ostringstream oss;
      oss << op_name << " cannot reduce broadcast gradient for dimension " << output_axis
          << ": input " << input_dim << " vs output " << output_dim;
      return Status::InvalidArgument(oss.str());
    }
    desc.output_strides[output_axis] = output_strides[output_axis];
    desc.input_strides[output_axis] =
        (input_axis >= 0 && input_dim != 1) ? input_strides[static_cast<size_t>(input_axis)] : 0;
  }

  return desc;
}

__global__ void FillFloatKernel(float *output, float value, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    output[idx] = value;
  }
}

__global__ void NegateFloatKernel(const float *input, float *output, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    output[idx] = -input[idx];
  }
}

__global__ void BroadcastReduceGradientFloatKernel(const float *output_grad, float *input_grad,
                                                   BroadcastGradientDescriptor desc,
                                                   int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t remaining = idx;
  int64_t input_offset = 0;
  for (int axis = 0; axis < desc.rank; ++axis) {
    int64_t output_stride = desc.output_strides[axis];
    int64_t coord = output_stride == 0 ? 0 : remaining / output_stride;
    remaining = output_stride == 0 ? 0 : remaining % output_stride;
    input_offset += coord * desc.input_strides[axis];
  }

  atomicAdd(input_grad + input_offset, output_grad[idx]);
}

__global__ void ReduceSumBackwardFloatKernel(const float *output_grad, float *input_grad,
                                             int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    input_grad[idx] = output_grad[0];
  }
}

__global__ void ReluForwardFloatKernel(const float *input, float *output, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    float value = input[idx];
    output[idx] = value > 0.0f ? value : 0.0f;
  }
}

__global__ void ReluBackwardFloatKernel(const float *output_grad, const float *input,
                                        float *input_grad, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    input_grad[idx] = input[idx] > 0.0f ? output_grad[idx] : 0.0f;
  }
}

__global__ void SigmoidForwardFloatKernel(const float *input, float *output, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

__global__ void SigmoidBackwardFloatKernel(const float *output_grad, const float *output,
                                           float *input_grad, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    float sigmoid = output[idx];
    input_grad[idx] = output_grad[idx] * sigmoid * (1.0f - sigmoid);
  }
}

__global__ void MatMulBackwardLeftFloatKernel(const float *output_grad, const float *rhs,
                                              float *lhs_grad, int64_t rows, int64_t inner,
                                              int64_t cols) {
  int64_t row = static_cast<int64_t>(blockIdx.y) * kMatMulTile + threadIdx.y;
  int64_t col = static_cast<int64_t>(blockIdx.x) * kMatMulTile + threadIdx.x;
  if (row >= rows || col >= inner) {
    return;
  }

  float sum = 0.0f;
  for (int64_t k = 0; k < cols; ++k) {
    sum += output_grad[row * cols + k] * rhs[col * cols + k];
  }
  lhs_grad[row * inner + col] = sum;
}

__global__ void MatMulBackwardRightFloatKernel(const float *lhs, const float *output_grad,
                                               float *rhs_grad, int64_t rows, int64_t inner,
                                               int64_t cols) {
  int64_t row = static_cast<int64_t>(blockIdx.y) * kMatMulTile + threadIdx.y;
  int64_t col = static_cast<int64_t>(blockIdx.x) * kMatMulTile + threadIdx.x;
  if (row >= inner || col >= cols) {
    return;
  }

  float sum = 0.0f;
  for (int64_t k = 0; k < rows; ++k) {
    sum += lhs[k * inner + row] * output_grad[k * cols + col];
  }
  rhs_grad[row * cols + col] = sum;
}

Status LaunchFillFloat(RuntimeContext &ctx, Tensor *output, float value) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(*output, "FillFloat output"));
  auto blocks = detail::BlocksForElements(output->numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    FillFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(output->data_as<float>(),
                                                                           value, output->numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("FillFloat kernel"));
  }
  return Status::Ok();
}

Result<Tensor> TensorOnesLikeFloat(RuntimeContext &ctx, const Tensor &input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input, "Backward target"));
  auto output = Tensor::AllocateAsync(input.shape(), DType::kFloat32, ctx.stream());
  if (!output.ok()) {
    return output.status();
  }
  Tensor result = output.value();
  DLCUDA_RETURN_IF_ERROR(LaunchFillFloat(ctx, &result, 1.0f));
  return result;
}

Status TensorNegateFloat(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("TensorNegateFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input, "TensorNegateFloat input"));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(output, input.shape(), DType::kFloat32, ctx.stream()));
  auto blocks = detail::BlocksForElements(input.numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    NegateFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        input.data_as<float>(), output->data_as<float>(), input.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("TensorNegateFloat kernel"));
  }
  return Status::Ok();
}

Status MaybeReduceBroadcastGradient(RuntimeContext &ctx, const Tensor &output_grad,
                                    const std::vector<int64_t> &input_shape, Tensor *input_grad) {
  if (input_grad == nullptr) {
    return Status::InvalidArgument("MaybeReduceBroadcastGradient output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output_grad, "Broadcast output gradient"));
  if (output_grad.shape() == input_shape) {
    *input_grad = output_grad;
    return Status::Ok();
  }

  auto descriptor =
      BuildBroadcastGradientDescriptor(input_shape, output_grad.shape(), "BroadcastGradient");
  if (!descriptor.ok()) {
    return descriptor.status();
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(input_grad, input_shape, DType::kFloat32, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(input_grad->FillZero(ctx.stream()));

  auto blocks = detail::BlocksForElements(output_grad.numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    BroadcastReduceGradientFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        output_grad.data_as<float>(), input_grad->data_as<float>(), descriptor.value(),
        output_grad.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("BroadcastReduceGradient kernel"));
  }
  return Status::Ok();
}

Status TensorReduceSumBackwardFloat(RuntimeContext &ctx, const Tensor &output_grad,
                                    const std::vector<int64_t> &input_shape, Tensor *input_grad) {
  if (input_grad == nullptr) {
    return Status::InvalidArgument("TensorReduceSumBackwardFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output_grad, "ReduceSum output gradient"));
  if (output_grad.numel() != 1) {
    return Status::InvalidArgument("ReduceSum output gradient must contain one element");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(input_grad, input_shape, DType::kFloat32, ctx.stream()));
  auto blocks = detail::BlocksForElements(input_grad->numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    ReduceSumBackwardFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        output_grad.data_as<float>(), input_grad->data_as<float>(), input_grad->numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ReduceSum backward kernel"));
  }
  return Status::Ok();
}

Status TensorReluFloat(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("TensorReluFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input, "Relu input"));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(output, input.shape(), DType::kFloat32, ctx.stream()));
  auto blocks = detail::BlocksForElements(input.numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    ReluForwardFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        input.data_as<float>(), output->data_as<float>(), input.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Relu forward kernel"));
  }
  return Status::Ok();
}

Status TensorReluBackwardFloat(RuntimeContext &ctx, const Tensor &output_grad, const Tensor &input,
                               Tensor *input_grad) {
  if (input_grad == nullptr) {
    return Status::InvalidArgument("TensorReluBackwardFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output_grad, "Relu output gradient"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input, "Relu cached input"));
  if (output_grad.shape() != input.shape()) {
    return Status::InvalidArgument("Relu gradient shape mismatch");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(input_grad, input.shape(), DType::kFloat32, ctx.stream()));
  auto blocks = detail::BlocksForElements(input.numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    ReluBackwardFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        output_grad.data_as<float>(), input.data_as<float>(), input_grad->data_as<float>(),
        input.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Relu backward kernel"));
  }
  return Status::Ok();
}

Status TensorSigmoidFloat(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("TensorSigmoidFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input, "Sigmoid input"));
  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(output, input.shape(), DType::kFloat32, ctx.stream()));
  auto blocks = detail::BlocksForElements(input.numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    SigmoidForwardFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        input.data_as<float>(), output->data_as<float>(), input.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Sigmoid forward kernel"));
  }
  return Status::Ok();
}

Status TensorSigmoidBackwardFloat(RuntimeContext &ctx, const Tensor &output_grad,
                                  const Tensor &output, Tensor *input_grad) {
  if (input_grad == nullptr) {
    return Status::InvalidArgument("TensorSigmoidBackwardFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output_grad, "Sigmoid output gradient"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output, "Sigmoid cached output"));
  if (output_grad.shape() != output.shape()) {
    return Status::InvalidArgument("Sigmoid gradient shape mismatch");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(input_grad, output.shape(), DType::kFloat32, ctx.stream()));
  auto blocks = detail::BlocksForElements(output.numel(), kAutogradThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    SigmoidBackwardFloatKernel<<<blocks.value(), kAutogradThreads, 0, ctx.stream()>>>(
        output_grad.data_as<float>(), output.data_as<float>(), input_grad->data_as<float>(),
        output.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Sigmoid backward kernel"));
  }
  return Status::Ok();
}

Status TensorMatMulBackwardLeftFloat(RuntimeContext &ctx, const Tensor &output_grad,
                                     const Tensor &rhs, const std::vector<int64_t> &lhs_shape,
                                     Tensor *lhs_grad) {
  if (lhs_grad == nullptr) {
    return Status::InvalidArgument("TensorMatMulBackwardLeftFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output_grad, "MatMul output gradient"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(rhs, "MatMul rhs"));
  if (lhs_shape.size() != 2 || rhs.rank() != 2 || output_grad.rank() != 2) {
    return Status::InvalidArgument("MatMul backward requires rank-2 tensors");
  }
  int64_t rows = lhs_shape[0];
  int64_t inner = lhs_shape[1];
  int64_t cols = rhs.dim(1);
  if (rhs.dim(0) != inner || output_grad.dim(0) != rows || output_grad.dim(1) != cols) {
    return Status::InvalidArgument("MatMul backward-left shape mismatch");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(lhs_grad, lhs_shape, DType::kFloat32, ctx.stream()));
  if (rows == 0 || inner == 0) {
    return Status::Ok();
  }
  auto rows_int = detail::CheckedInt(rows, "MatMul backward rows");
  if (!rows_int.ok()) {
    return rows_int.status();
  }
  auto inner_int = detail::CheckedInt(inner, "MatMul backward inner");
  if (!inner_int.ok()) {
    return inner_int.status();
  }
  dim3 block(kMatMulTile, kMatMulTile);
  dim3 grid((inner_int.value() + kMatMulTile - 1) / kMatMulTile,
            (rows_int.value() + kMatMulTile - 1) / kMatMulTile);
  MatMulBackwardLeftFloatKernel<<<grid, block, 0, ctx.stream()>>>(
      output_grad.data_as<float>(), rhs.data_as<float>(), lhs_grad->data_as<float>(), rows, inner,
      cols);
  return detail::CheckKernelLaunch("MatMul backward-left kernel");
}

Status TensorMatMulBackwardRightFloat(RuntimeContext &ctx, const Tensor &lhs,
                                      const Tensor &output_grad,
                                      const std::vector<int64_t> &rhs_shape, Tensor *rhs_grad) {
  if (rhs_grad == nullptr) {
    return Status::InvalidArgument("TensorMatMulBackwardRightFloat output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(lhs, "MatMul lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(output_grad, "MatMul output gradient"));
  if (lhs.rank() != 2 || rhs_shape.size() != 2 || output_grad.rank() != 2) {
    return Status::InvalidArgument("MatMul backward requires rank-2 tensors");
  }
  int64_t rows = lhs.dim(0);
  int64_t inner = lhs.dim(1);
  int64_t cols = rhs_shape[1];
  if (rhs_shape[0] != inner || output_grad.dim(0) != rows || output_grad.dim(1) != cols) {
    return Status::InvalidArgument("MatMul backward-right shape mismatch");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(rhs_grad, rhs_shape, DType::kFloat32, ctx.stream()));
  if (inner == 0 || cols == 0) {
    return Status::Ok();
  }
  auto inner_int = detail::CheckedInt(inner, "MatMul backward inner");
  if (!inner_int.ok()) {
    return inner_int.status();
  }
  auto cols_int = detail::CheckedInt(cols, "MatMul backward cols");
  if (!cols_int.ok()) {
    return cols_int.status();
  }
  dim3 block(kMatMulTile, kMatMulTile);
  dim3 grid((cols_int.value() + kMatMulTile - 1) / kMatMulTile,
            (inner_int.value() + kMatMulTile - 1) / kMatMulTile);
  MatMulBackwardRightFloatKernel<<<grid, block, 0, ctx.stream()>>>(
      lhs.data_as<float>(), output_grad.data_as<float>(), rhs_grad->data_as<float>(), rows, inner,
      cols);
  return detail::CheckKernelLaunch("MatMul backward-right kernel");
}

bool AnyRequiresGrad(const std::vector<AutoTensor> &inputs) {
  for (const AutoTensor &input : inputs) {
    if (input.requires_grad()) {
      return true;
    }
  }
  return false;
}

} // namespace

AutoTensor::AutoTensor(GradientTape *tape, int64_t id, Tensor value, bool requires_grad)
    : tape_(tape), id_(id), value_(std::move(value)), requires_grad_(requires_grad) {}

bool AutoTensor::defined() const {
  return tape_ != nullptr && value_.defined();
}

Status AutoTensor::Backward(RuntimeContext &ctx) const {
  if (tape_ == nullptr) {
    return Status::InvalidArgument("AutoTensor is not attached to a GradientTape");
  }
  return tape_->Backward(ctx, *this);
}

Status AutoTensor::Backward(RuntimeContext &ctx, const Tensor &initial_gradient) const {
  if (tape_ == nullptr) {
    return Status::InvalidArgument("AutoTensor is not attached to a GradientTape");
  }
  return tape_->Backward(ctx, *this, initial_gradient);
}

Result<Tensor> AutoTensor::grad() const {
  if (tape_ == nullptr) {
    return Status::InvalidArgument("AutoTensor is not attached to a GradientTape");
  }
  return tape_->Gradient(*this);
}

AutoTensor GradientTape::Variable(const Tensor &value, bool requires_grad) {
  return CreateTensor(value, requires_grad);
}

AutoTensor GradientTape::Constant(const Tensor &value) {
  return CreateTensor(value, false);
}

Result<AutoTensor> GradientTape::Add(RuntimeContext &ctx, const AutoTensor &lhs,
                                     const AutoTensor &rhs) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(rhs, "rhs"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorAdd(ctx, lhs.value_, rhs.value_, &output));
  bool requires_grad = lhs.requires_grad_ || rhs.requires_grad_;
  if (requires_grad) {
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(lhs.value_, "Add lhs"));
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(rhs.value_, "Add rhs"));
  }

  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t lhs_id = lhs.id_;
    int64_t rhs_id = rhs.id_;
    bool lhs_requires_grad = lhs.requires_grad_;
    bool rhs_requires_grad = rhs.requires_grad_;
    std::vector<int64_t> lhs_shape = lhs.value_.shape();
    std::vector<int64_t> rhs_shape = rhs.value_.shape();
    RecordNode(Node{result.id_, "add",
                    [lhs_id, rhs_id, lhs_requires_grad, rhs_requires_grad, lhs_shape, rhs_shape](
                        RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad) {
                      if (lhs_requires_grad) {
                        Tensor lhs_grad;
                        DLCUDA_RETURN_IF_ERROR(
                            MaybeReduceBroadcastGradient(ctx, output_grad, lhs_shape, &lhs_grad));
                        DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, lhs_id, lhs_grad));
                      }
                      if (rhs_requires_grad) {
                        Tensor rhs_grad;
                        DLCUDA_RETURN_IF_ERROR(
                            MaybeReduceBroadcastGradient(ctx, output_grad, rhs_shape, &rhs_grad));
                        DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, rhs_id, rhs_grad));
                      }
                      return Status::Ok();
                    }});
  }
  return result;
}

Result<AutoTensor> GradientTape::Subtract(RuntimeContext &ctx, const AutoTensor &lhs,
                                          const AutoTensor &rhs) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(rhs, "rhs"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorSubtract(ctx, lhs.value_, rhs.value_, &output));
  bool requires_grad = lhs.requires_grad_ || rhs.requires_grad_;
  if (requires_grad) {
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(lhs.value_, "Subtract lhs"));
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(rhs.value_, "Subtract rhs"));
  }

  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t lhs_id = lhs.id_;
    int64_t rhs_id = rhs.id_;
    bool lhs_requires_grad = lhs.requires_grad_;
    bool rhs_requires_grad = rhs.requires_grad_;
    std::vector<int64_t> lhs_shape = lhs.value_.shape();
    std::vector<int64_t> rhs_shape = rhs.value_.shape();
    RecordNode(Node{result.id_, "subtract",
                    [lhs_id, rhs_id, lhs_requires_grad, rhs_requires_grad, lhs_shape, rhs_shape](
                        RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad) {
                      if (lhs_requires_grad) {
                        Tensor lhs_grad;
                        DLCUDA_RETURN_IF_ERROR(
                            MaybeReduceBroadcastGradient(ctx, output_grad, lhs_shape, &lhs_grad));
                        DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, lhs_id, lhs_grad));
                      }
                      if (rhs_requires_grad) {
                        Tensor neg_grad;
                        DLCUDA_RETURN_IF_ERROR(TensorNegateFloat(ctx, output_grad, &neg_grad));
                        Tensor rhs_grad;
                        DLCUDA_RETURN_IF_ERROR(
                            MaybeReduceBroadcastGradient(ctx, neg_grad, rhs_shape, &rhs_grad));
                        DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, rhs_id, rhs_grad));
                      }
                      return Status::Ok();
                    }});
  }
  return result;
}

Result<AutoTensor> GradientTape::Multiply(RuntimeContext &ctx, const AutoTensor &lhs,
                                          const AutoTensor &rhs) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(rhs, "rhs"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorMultiply(ctx, lhs.value_, rhs.value_, &output));
  bool requires_grad = lhs.requires_grad_ || rhs.requires_grad_;
  if (requires_grad) {
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(lhs.value_, "Multiply lhs"));
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(rhs.value_, "Multiply rhs"));
  }

  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t lhs_id = lhs.id_;
    int64_t rhs_id = rhs.id_;
    bool lhs_requires_grad = lhs.requires_grad_;
    bool rhs_requires_grad = rhs.requires_grad_;
    Tensor lhs_value = lhs.value_;
    Tensor rhs_value = rhs.value_;
    std::vector<int64_t> lhs_shape = lhs.value_.shape();
    std::vector<int64_t> rhs_shape = rhs.value_.shape();
    RecordNode(Node{
        result.id_, "multiply",
        [lhs_id, rhs_id, lhs_requires_grad, rhs_requires_grad, lhs_value, rhs_value, lhs_shape,
         rhs_shape](RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad) {
          if (lhs_requires_grad) {
            Tensor lhs_contribution;
            DLCUDA_RETURN_IF_ERROR(TensorMultiply(ctx, output_grad, rhs_value, &lhs_contribution));
            Tensor lhs_grad;
            DLCUDA_RETURN_IF_ERROR(
                MaybeReduceBroadcastGradient(ctx, lhs_contribution, lhs_shape, &lhs_grad));
            DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, lhs_id, lhs_grad));
          }
          if (rhs_requires_grad) {
            Tensor rhs_contribution;
            DLCUDA_RETURN_IF_ERROR(TensorMultiply(ctx, output_grad, lhs_value, &rhs_contribution));
            Tensor rhs_grad;
            DLCUDA_RETURN_IF_ERROR(
                MaybeReduceBroadcastGradient(ctx, rhs_contribution, rhs_shape, &rhs_grad));
            DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, rhs_id, rhs_grad));
          }
          return Status::Ok();
        }});
  }
  return result;
}

Result<AutoTensor> GradientTape::Divide(RuntimeContext &ctx, const AutoTensor &lhs,
                                        const AutoTensor &rhs) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(rhs, "rhs"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorDivide(ctx, lhs.value_, rhs.value_, &output));
  bool requires_grad = lhs.requires_grad_ || rhs.requires_grad_;
  if (requires_grad) {
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(lhs.value_, "Divide lhs"));
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(rhs.value_, "Divide rhs"));
  }

  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t lhs_id = lhs.id_;
    int64_t rhs_id = rhs.id_;
    bool lhs_requires_grad = lhs.requires_grad_;
    bool rhs_requires_grad = rhs.requires_grad_;
    Tensor lhs_value = lhs.value_;
    Tensor rhs_value = rhs.value_;
    std::vector<int64_t> lhs_shape = lhs.value_.shape();
    std::vector<int64_t> rhs_shape = rhs.value_.shape();
    RecordNode(Node{
        result.id_, "divide",
        [lhs_id, rhs_id, lhs_requires_grad, rhs_requires_grad, lhs_value, rhs_value, lhs_shape,
         rhs_shape](RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad) {
          if (lhs_requires_grad) {
            Tensor lhs_contribution;
            DLCUDA_RETURN_IF_ERROR(TensorDivide(ctx, output_grad, rhs_value, &lhs_contribution));
            Tensor lhs_grad;
            DLCUDA_RETURN_IF_ERROR(
                MaybeReduceBroadcastGradient(ctx, lhs_contribution, lhs_shape, &lhs_grad));
            DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, lhs_id, lhs_grad));
          }
          if (rhs_requires_grad) {
            Tensor numerator;
            DLCUDA_RETURN_IF_ERROR(TensorMultiply(ctx, output_grad, lhs_value, &numerator));
            Tensor rhs_squared;
            DLCUDA_RETURN_IF_ERROR(TensorMultiply(ctx, rhs_value, rhs_value, &rhs_squared));
            Tensor quotient;
            DLCUDA_RETURN_IF_ERROR(TensorDivide(ctx, numerator, rhs_squared, &quotient));
            Tensor neg_quotient;
            DLCUDA_RETURN_IF_ERROR(TensorNegateFloat(ctx, quotient, &neg_quotient));
            Tensor rhs_grad;
            DLCUDA_RETURN_IF_ERROR(
                MaybeReduceBroadcastGradient(ctx, neg_quotient, rhs_shape, &rhs_grad));
            DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, rhs_id, rhs_grad));
          }
          return Status::Ok();
        }});
  }
  return result;
}

Result<AutoTensor> GradientTape::MatMul(RuntimeContext &ctx, const AutoTensor &lhs,
                                        const AutoTensor &rhs) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(rhs, "rhs"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorMatMul(ctx, lhs.value_, rhs.value_, &output));
  bool requires_grad = lhs.requires_grad_ || rhs.requires_grad_;
  if (requires_grad) {
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(lhs.value_, "MatMul lhs"));
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(rhs.value_, "MatMul rhs"));
  }

  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t lhs_id = lhs.id_;
    int64_t rhs_id = rhs.id_;
    bool lhs_requires_grad = lhs.requires_grad_;
    bool rhs_requires_grad = rhs.requires_grad_;
    Tensor lhs_value = lhs.value_;
    Tensor rhs_value = rhs.value_;
    std::vector<int64_t> lhs_shape = lhs.value_.shape();
    std::vector<int64_t> rhs_shape = rhs.value_.shape();
    RecordNode(
        Node{result.id_, "matmul",
             [lhs_id, rhs_id, lhs_requires_grad, rhs_requires_grad, lhs_value, rhs_value, lhs_shape,
              rhs_shape](RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad) {
               if (lhs_requires_grad) {
                 Tensor lhs_grad;
                 DLCUDA_RETURN_IF_ERROR(TensorMatMulBackwardLeftFloat(ctx, output_grad, rhs_value,
                                                                      lhs_shape, &lhs_grad));
                 DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, lhs_id, lhs_grad));
               }
               if (rhs_requires_grad) {
                 Tensor rhs_grad;
                 DLCUDA_RETURN_IF_ERROR(TensorMatMulBackwardRightFloat(ctx, lhs_value, output_grad,
                                                                       rhs_shape, &rhs_grad));
                 DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, rhs_id, rhs_grad));
               }
               return Status::Ok();
             }});
  }
  return result;
}

Result<AutoTensor> GradientTape::ReduceSum(RuntimeContext &ctx, const AutoTensor &input) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(input, "input"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorReduceSum(ctx, input.value_, &output));
  bool requires_grad = input.requires_grad_;
  if (requires_grad) {
    DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input.value_, "ReduceSum input"));
  }

  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t input_id = input.id_;
    std::vector<int64_t> input_shape = input.value_.shape();
    RecordNode(Node{result.id_, "reduce_sum",
                    [input_id, input_shape](RuntimeContext &ctx, GradientTape &tape,
                                            const Tensor &output_grad) {
                      Tensor input_grad;
                      DLCUDA_RETURN_IF_ERROR(
                          TensorReduceSumBackwardFloat(ctx, output_grad, input_shape, &input_grad));
                      return tape.AccumulateGradient(ctx, input_id, input_grad);
                    }});
  }
  return result;
}

Result<AutoTensor> GradientTape::Relu(RuntimeContext &ctx, const AutoTensor &input) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(input, "input"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input.value_, "Relu input"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorReluFloat(ctx, input.value_, &output));
  bool requires_grad = input.requires_grad_;
  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t input_id = input.id_;
    Tensor input_value = input.value_;
    RecordNode(Node{result.id_, "relu",
                    [input_id, input_value](RuntimeContext &ctx, GradientTape &tape,
                                            const Tensor &output_grad) {
                      Tensor input_grad;
                      DLCUDA_RETURN_IF_ERROR(
                          TensorReluBackwardFloat(ctx, output_grad, input_value, &input_grad));
                      return tape.AccumulateGradient(ctx, input_id, input_grad);
                    }});
  }
  return result;
}

Result<AutoTensor> GradientTape::Sigmoid(RuntimeContext &ctx, const AutoTensor &input) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(input, "input"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloat32Tensor(input.value_, "Sigmoid input"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorSigmoidFloat(ctx, input.value_, &output));
  bool requires_grad = input.requires_grad_;
  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t input_id = input.id_;
    Tensor output_value = output;
    RecordNode(Node{result.id_, "sigmoid",
                    [input_id, output_value](RuntimeContext &ctx, GradientTape &tape,
                                             const Tensor &output_grad) {
                      Tensor input_grad;
                      DLCUDA_RETURN_IF_ERROR(
                          TensorSigmoidBackwardFloat(ctx, output_grad, output_value, &input_grad));
                      return tape.AccumulateGradient(ctx, input_id, input_grad);
                    }});
  }
  return result;
}

Result<AutoTensor> GradientTape::ApplyModule(RuntimeContext &ctx, Module &module,
                                             const AutoTensor &input) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(input, "input"));

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(module.Forward(ctx, input.value_, &output));

  std::vector<ParameterRef> params;
  module.AppendParameters("", &params);
  bool has_parameters = false;
  for (const ParameterRef &param : params) {
    if (param.value != nullptr && param.grad != nullptr) {
      has_parameters = true;
      break;
    }
  }

  bool requires_grad = input.requires_grad_ || has_parameters;
  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    int64_t input_id = input.id_;
    bool input_requires_grad = input.requires_grad_;
    Module *module_ptr = &module;
    RecordNode(
        Node{result.id_, "module",
             [input_id, input_requires_grad, module_ptr](RuntimeContext &ctx, GradientTape &tape,
                                                         const Tensor &output_grad) {
               Tensor input_grad;
               Tensor *input_grad_ptr = input_requires_grad ? &input_grad : nullptr;
               DLCUDA_RETURN_IF_ERROR(module_ptr->Backward(ctx, output_grad, input_grad_ptr));
               if (input_requires_grad && input_grad.defined()) {
                 DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, input_id, input_grad));
               }
               return Status::Ok();
             }});
  }
  return result;
}

Status GradientTape::RegisterCustomOp(std::string name, CustomAutogradForward forward,
                                      CustomAutogradBackward backward) {
  if (name.empty()) {
    return Status::InvalidArgument("Custom autograd op name must not be empty");
  }
  if (!forward) {
    return Status::InvalidArgument("Custom autograd op forward function is empty");
  }
  if (!backward) {
    return Status::InvalidArgument("Custom autograd op backward function is empty");
  }
  custom_ops_.insert_or_assign(std::move(name), CustomOp{std::move(forward), std::move(backward)});
  return Status::Ok();
}

Result<AutoTensor> GradientTape::ApplyCustomOp(RuntimeContext &ctx, const std::string &name,
                                               const std::vector<AutoTensor> &inputs) {
  auto op_it = custom_ops_.find(name);
  if (op_it == custom_ops_.end()) {
    return Status::NotFound("Custom autograd op not registered: " + name);
  }

  std::vector<Tensor> raw_inputs;
  raw_inputs.reserve(inputs.size());
  std::vector<int64_t> input_ids;
  input_ids.reserve(inputs.size());
  std::vector<bool> input_requires_grad;
  input_requires_grad.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string input_name = "input " + std::to_string(i);
    DLCUDA_RETURN_IF_ERROR(ValidateTensor(inputs[i], input_name.c_str()));
    raw_inputs.push_back(inputs[i].value_);
    input_ids.push_back(inputs[i].id_);
    input_requires_grad.push_back(inputs[i].requires_grad_);
  }

  Tensor output;
  DLCUDA_RETURN_IF_ERROR(op_it->second.forward(ctx, raw_inputs, &output));
  if (!output.defined()) {
    return Status::RuntimeError("Custom autograd op returned an undefined output: " + name);
  }

  bool requires_grad = AnyRequiresGrad(inputs);
  AutoTensor result = CreateTensor(output, requires_grad);
  if (requires_grad) {
    CustomAutogradBackward backward = op_it->second.backward;
    Tensor output_value = output;
    RecordNode(Node{
        result.id_, name,
        [backward, raw_inputs, output_value, input_ids, input_requires_grad,
         name](RuntimeContext &ctx, GradientTape &tape, const Tensor &output_grad) {
          std::vector<Tensor> input_grads;
          DLCUDA_RETURN_IF_ERROR(
              backward(ctx, output_grad, raw_inputs, output_value, &input_grads));
          if (input_grads.size() != input_ids.size()) {
            return Status::RuntimeError(
                "Custom autograd op returned " + std::to_string(input_grads.size()) +
                " input gradients for " + std::to_string(input_ids.size()) + " inputs: " + name);
          }
          for (size_t i = 0; i < input_ids.size(); ++i) {
            if (input_requires_grad[i] && input_grads[i].defined()) {
              DLCUDA_RETURN_IF_ERROR(tape.AccumulateGradient(ctx, input_ids[i], input_grads[i]));
            }
          }
          return Status::Ok();
        }});
  }

  return result;
}

Status GradientTape::Backward(RuntimeContext &ctx, const AutoTensor &target) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(target, "target"));
  if (target.value_.numel() != 1) {
    return Status::InvalidArgument(
        "Backward without an initial gradient requires a one-element target");
  }
  auto initial_gradient = TensorOnesLikeFloat(ctx, target.value_);
  if (!initial_gradient.ok()) {
    return initial_gradient.status();
  }
  return Backward(ctx, target, initial_gradient.value());
}

Status GradientTape::Backward(RuntimeContext &ctx, const AutoTensor &target,
                              const Tensor &initial_gradient) {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(target, "target"));
  if (!target.requires_grad_) {
    return Status::InvalidArgument("Backward target does not require gradients");
  }
  if (!initial_gradient.defined()) {
    return Status::InvalidArgument("Initial gradient is undefined");
  }
  if (initial_gradient.shape() != target.value_.shape()) {
    return Status::InvalidArgument("Initial gradient shape must match target shape");
  }
  if (initial_gradient.dtype() != target.value_.dtype()) {
    return Status::InvalidArgument("Initial gradient dtype must match target dtype");
  }

  gradients_.clear();
  gradients_.insert_or_assign(target.id_, initial_gradient);

  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    auto grad_it = gradients_.find(it->output_id);
    if (grad_it == gradients_.end()) {
      continue;
    }
    Tensor output_grad = grad_it->second;
    if (!it->backward) {
      return Status::RuntimeError("Autograd node has no backward function: " + it->op_name);
    }
    DLCUDA_RETURN_IF_ERROR(it->backward(ctx, *this, output_grad));
  }
  return Status::Ok();
}

Result<Tensor> GradientTape::Gradient(const AutoTensor &tensor) const {
  DLCUDA_RETURN_IF_ERROR(ValidateTensor(tensor, "tensor"));
  auto it = gradients_.find(tensor.id_);
  if (it == gradients_.end()) {
    return Status::NotFound("No gradient is available for tensor id " + std::to_string(tensor.id_));
  }
  return it->second;
}

void GradientTape::ClearGradients() {
  gradients_.clear();
}

void GradientTape::Reset() {
  nodes_.clear();
  gradients_.clear();
}

AutoTensor GradientTape::CreateTensor(Tensor value, bool requires_grad) {
  return AutoTensor(this, next_id_++, std::move(value), requires_grad);
}

Status GradientTape::ValidateTensor(const AutoTensor &tensor, const char *name) const {
  if (tensor.tape_ != this) {
    return Status::InvalidArgument(std::string(name) + " is not attached to this GradientTape");
  }
  if (!tensor.value_.defined()) {
    return Status::InvalidArgument(std::string(name) + " tensor is undefined");
  }
  return Status::Ok();
}

Status GradientTape::AccumulateGradient(RuntimeContext &ctx, int64_t id, const Tensor &grad) {
  if (!grad.defined()) {
    return Status::Ok();
  }
  auto it = gradients_.find(id);
  if (it == gradients_.end()) {
    gradients_.insert_or_assign(id, grad);
    return Status::Ok();
  }
  if (it->second.shape() != grad.shape()) {
    return Status::RuntimeError("Gradient accumulation shape mismatch for tensor id " +
                                std::to_string(id));
  }
  if (it->second.dtype() != grad.dtype()) {
    return Status::RuntimeError("Gradient accumulation dtype mismatch for tensor id " +
                                std::to_string(id));
  }

  Tensor accumulated;
  DLCUDA_RETURN_IF_ERROR(TensorAdd(ctx, it->second, grad, &accumulated));
  it->second = accumulated;
  return Status::Ok();
}

void GradientTape::RecordNode(Node node) {
  nodes_.push_back(std::move(node));
}

} // namespace dlcuda
