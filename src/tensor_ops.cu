#include "dl_cuda/tensor_ops.hpp"

#include "dl_cuda/detail/cuda_dtype.cuh"
#include "dl_cuda/detail/cuda_utils.hpp"
#include "dl_cuda/detail/shape_utils.hpp"
#include "dl_cuda/detail/tensor_validation.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace dlcuda {
namespace {

constexpr int kTensorOpThreads = 256;
constexpr int kTensorReductionMaxBlocks = 4096;
constexpr int kMatMulTile = 16;
constexpr int kMaxBroadcastRank = 8;

using TensorOpBlockReduce = cub::BlockReduce<float, kTensorOpThreads>;
using detail::ValidateDefinedTensor;
using detail::ValidateFloatingTensor;

struct BroadcastDescriptor {
  int rank = 0;
  int64_t lhs_strides[kMaxBroadcastRank] = {};
  int64_t rhs_strides[kMaxBroadcastRank] = {};
  int64_t out_strides[kMaxBroadcastRank] = {};
};

bool IsSupportedElementwiseDType(DType dtype, TensorBinaryOp op) {
  if (IsFloatingPointDType(dtype)) {
    return true;
  }
  if (op == TensorBinaryOp::kDivide) {
    return false;
  }
  return dtype == DType::kInt32;
}

Result<std::vector<int64_t>> BroadcastShape(const Tensor &lhs, const Tensor &rhs,
                                            const char *op_name) {
  size_t out_rank = std::max(lhs.shape().size(), rhs.shape().size());
  if (out_rank > kMaxBroadcastRank) {
    std::ostringstream oss;
    oss << op_name << " supports broadcast rank up to " << kMaxBroadcastRank;
    return Status::InvalidArgument(oss.str());
  }

  std::vector<int64_t> out_shape(out_rank, 1);
  for (size_t out_axis = 0; out_axis < out_rank; ++out_axis) {
    int lhs_axis = static_cast<int>(out_axis) - static_cast<int>(out_rank - lhs.shape().size());
    int rhs_axis = static_cast<int>(out_axis) - static_cast<int>(out_rank - rhs.shape().size());
    int64_t lhs_dim = lhs_axis >= 0 ? lhs.shape()[static_cast<size_t>(lhs_axis)] : 1;
    int64_t rhs_dim = rhs_axis >= 0 ? rhs.shape()[static_cast<size_t>(rhs_axis)] : 1;

    if (lhs_dim == rhs_dim) {
      out_shape[out_axis] = lhs_dim;
    } else if (lhs_dim == 1) {
      out_shape[out_axis] = rhs_dim;
    } else if (rhs_dim == 1) {
      out_shape[out_axis] = lhs_dim;
    } else {
      std::ostringstream oss;
      oss << op_name << " cannot broadcast dimension " << out_axis << ": " << lhs_dim << " vs "
          << rhs_dim;
      return Status::InvalidArgument(oss.str());
    }
  }

  return out_shape;
}

BroadcastDescriptor BuildBroadcastDescriptor(const Tensor &lhs, const Tensor &rhs,
                                             const std::vector<int64_t> &out_shape) {
  BroadcastDescriptor desc;
  desc.rank = static_cast<int>(out_shape.size());

  std::vector<int64_t> lhs_strides = detail::ContiguousStrides(lhs.shape());
  std::vector<int64_t> rhs_strides = detail::ContiguousStrides(rhs.shape());
  std::vector<int64_t> out_strides = detail::ContiguousStrides(out_shape);

  for (size_t out_axis = 0; out_axis < out_shape.size(); ++out_axis) {
    int lhs_axis =
        static_cast<int>(out_axis) - static_cast<int>(out_shape.size() - lhs.shape().size());
    int rhs_axis =
        static_cast<int>(out_axis) - static_cast<int>(out_shape.size() - rhs.shape().size());

    desc.out_strides[out_axis] = out_strides[out_axis];

    if (lhs_axis >= 0) {
      int64_t lhs_dim = lhs.shape()[static_cast<size_t>(lhs_axis)];
      desc.lhs_strides[out_axis] = lhs_dim == 1 ? 0 : lhs_strides[static_cast<size_t>(lhs_axis)];
    } else {
      desc.lhs_strides[out_axis] = 0;
    }

    if (rhs_axis >= 0) {
      int64_t rhs_dim = rhs.shape()[static_cast<size_t>(rhs_axis)];
      desc.rhs_strides[out_axis] = rhs_dim == 1 ? 0 : rhs_strides[static_cast<size_t>(rhs_axis)];
    } else {
      desc.rhs_strides[out_axis] = 0;
    }
  }

  return desc;
}

template <typename T> __device__ T ApplyBinaryOp(T lhs, T rhs, TensorBinaryOp op) {
  switch (op) {
  case TensorBinaryOp::kAdd:
    return lhs + rhs;
  case TensorBinaryOp::kSubtract:
    return lhs - rhs;
  case TensorBinaryOp::kMultiply:
    return lhs * rhs;
  case TensorBinaryOp::kDivide:
    return lhs / rhs;
  }
  return lhs;
}

__device__ float ApplyBinaryOpFloat(float lhs, float rhs, TensorBinaryOp op) {
  switch (op) {
  case TensorBinaryOp::kAdd:
    return lhs + rhs;
  case TensorBinaryOp::kSubtract:
    return lhs - rhs;
  case TensorBinaryOp::kMultiply:
    return lhs * rhs;
  case TensorBinaryOp::kDivide:
    return lhs / rhs;
  }
  return lhs;
}

template <typename T>
__global__ void TensorBinaryKernel(const T *lhs, const T *rhs, T *output, BroadcastDescriptor desc,
                                   int64_t total, TensorBinaryOp op) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t remaining = idx;
  int64_t lhs_offset = 0;
  int64_t rhs_offset = 0;
  for (int axis = 0; axis < desc.rank; ++axis) {
    int64_t coord = desc.out_strides[axis] == 0 ? 0 : remaining / desc.out_strides[axis];
    remaining = desc.out_strides[axis] == 0 ? 0 : remaining % desc.out_strides[axis];
    lhs_offset += coord * desc.lhs_strides[axis];
    rhs_offset += coord * desc.rhs_strides[axis];
  }

  output[idx] = ApplyBinaryOp(lhs[lhs_offset], rhs[rhs_offset], op);
}

template <typename Codec>
__global__ void
TensorBinaryFloatingKernel(const typename Codec::Storage *lhs, const typename Codec::Storage *rhs,
                           typename Codec::Storage *output, BroadcastDescriptor desc, int64_t total,
                           TensorBinaryOp op) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t remaining = idx;
  int64_t lhs_offset = 0;
  int64_t rhs_offset = 0;
  for (int axis = 0; axis < desc.rank; ++axis) {
    int64_t coord = desc.out_strides[axis] == 0 ? 0 : remaining / desc.out_strides[axis];
    remaining = desc.out_strides[axis] == 0 ? 0 : remaining % desc.out_strides[axis];
    lhs_offset += coord * desc.lhs_strides[axis];
    rhs_offset += coord * desc.rhs_strides[axis];
  }

  float lhs_value = Codec::Load(lhs, lhs_offset);
  float rhs_value = Codec::Load(rhs, rhs_offset);
  Codec::Store(output, idx, ApplyBinaryOpFloat(lhs_value, rhs_value, op));
}

template <typename Codec>
__global__ void
TensorMatMulKernel(const typename Codec::Storage *lhs, const typename Codec::Storage *rhs,
                   typename Codec::Storage *output, int64_t rows, int64_t inner, int64_t cols) {
  __shared__ float lhs_tile[kMatMulTile][kMatMulTile];
  __shared__ float rhs_tile[kMatMulTile][kMatMulTile];

  int64_t row = static_cast<int64_t>(blockIdx.y) * kMatMulTile + threadIdx.y;
  int64_t col = static_cast<int64_t>(blockIdx.x) * kMatMulTile + threadIdx.x;

  float sum = 0.0f;
  for (int64_t tile_start = 0; tile_start < inner; tile_start += kMatMulTile) {
    int64_t lhs_col = tile_start + threadIdx.x;
    int64_t rhs_row = tile_start + threadIdx.y;

    lhs_tile[threadIdx.y][threadIdx.x] =
        (row < rows && lhs_col < inner) ? Codec::Load(lhs, row * inner + lhs_col) : 0.0f;
    rhs_tile[threadIdx.y][threadIdx.x] =
        (rhs_row < inner && col < cols) ? Codec::Load(rhs, rhs_row * cols + col) : 0.0f;
    __syncthreads();

    for (int k = 0; k < kMatMulTile; ++k) {
      sum += lhs_tile[threadIdx.y][k] * rhs_tile[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < rows && col < cols) {
    Codec::Store(output, row * cols + col, sum);
  }
}

template <typename Codec>
__global__ void TensorSumKernel(const typename Codec::Storage *input, float *output, int64_t n) {
  __shared__ typename TensorOpBlockReduce::TempStorage reduce_storage;

  int tid = threadIdx.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0f;
  for (int64_t i = idx; i < n; i += stride) {
    local_sum += Codec::Load(input, i);
  }
  float block_sum = TensorOpBlockReduce(reduce_storage).Sum(local_sum);

  if (tid == 0) {
    atomicAdd(output, block_sum);
  }
}

template <typename Codec>
Status LaunchTensorBinaryFloating(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs,
                                  Tensor *output, BroadcastDescriptor desc, int64_t total,
                                  TensorBinaryOp op, int blocks, const char *op_name) {
  TensorBinaryFloatingKernel<Codec><<<blocks, kTensorOpThreads, 0, ctx.stream()>>>(
      lhs.data_as<typename Codec::Storage>(), rhs.data_as<typename Codec::Storage>(),
      output->data_as<typename Codec::Storage>(), desc, total, op);
  return detail::CheckKernelLaunch(std::string(op_name) + " kernel");
}

Status LaunchTensorBinaryFloating(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs,
                                  Tensor *output, BroadcastDescriptor desc, int64_t total,
                                  TensorBinaryOp op, int blocks, const char *op_name) {
  switch (lhs.dtype()) {
  case DType::kFloat32:
    return LaunchTensorBinaryFloating<detail::Float32Codec>(ctx, lhs, rhs, output, desc, total, op,
                                                            blocks, op_name);
  case DType::kFloat16:
    return LaunchTensorBinaryFloating<detail::Float16Codec>(ctx, lhs, rhs, output, desc, total, op,
                                                            blocks, op_name);
  case DType::kBFloat16:
    return LaunchTensorBinaryFloating<detail::BFloat16Codec>(ctx, lhs, rhs, output, desc, total, op,
                                                             blocks, op_name);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument(std::string(op_name) + " does not support dtype " +
                                 DTypeName(lhs.dtype()));
}

template <typename Codec>
Status LaunchTensorMatMul(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output,
                          int rows, int inner, int cols) {
  dim3 block(kMatMulTile, kMatMulTile);
  dim3 grid((cols + kMatMulTile - 1) / kMatMulTile, (rows + kMatMulTile - 1) / kMatMulTile);
  TensorMatMulKernel<Codec><<<grid, block, 0, ctx.stream()>>>(
      lhs.data_as<typename Codec::Storage>(), rhs.data_as<typename Codec::Storage>(),
      output->data_as<typename Codec::Storage>(), rows, inner, cols);
  return detail::CheckKernelLaunch("TensorMatMul kernel");
}

Status LaunchTensorMatMul(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output,
                          int rows, int inner, int cols) {
  switch (lhs.dtype()) {
  case DType::kFloat32:
    return LaunchTensorMatMul<detail::Float32Codec>(ctx, lhs, rhs, output, rows, inner, cols);
  case DType::kFloat16:
    return LaunchTensorMatMul<detail::Float16Codec>(ctx, lhs, rhs, output, rows, inner, cols);
  case DType::kBFloat16:
    return LaunchTensorMatMul<detail::BFloat16Codec>(ctx, lhs, rhs, output, rows, inner, cols);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("TensorMatMul does not support dtype " +
                                 std::string(DTypeName(lhs.dtype())));
}

template <typename Codec>
Status LaunchTensorReduceSum(RuntimeContext &ctx, const Tensor &input, Tensor *output, int blocks,
                             int64_t n) {
  TensorSumKernel<Codec><<<blocks, kTensorOpThreads, 0, ctx.stream()>>>(
      input.data_as<typename Codec::Storage>(), output->data_as<float>(), n);
  return detail::CheckKernelLaunch("TensorReduceSum kernel");
}

Status LaunchTensorReduceSum(RuntimeContext &ctx, const Tensor &input, Tensor *output, int blocks,
                             int64_t n) {
  switch (input.dtype()) {
  case DType::kFloat32:
    return LaunchTensorReduceSum<detail::Float32Codec>(ctx, input, output, blocks, n);
  case DType::kFloat16:
    return LaunchTensorReduceSum<detail::Float16Codec>(ctx, input, output, blocks, n);
  case DType::kBFloat16:
    return LaunchTensorReduceSum<detail::BFloat16Codec>(ctx, input, output, blocks, n);
  case DType::kInt32:
    break;
  }
  return Status::InvalidArgument("TensorReduceSum does not support dtype " +
                                 std::string(DTypeName(input.dtype())));
}

Status RunTensorBinaryOp(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output,
                         TensorBinaryOp op, const char *op_name) {
  if (output == nullptr) {
    return Status::InvalidArgument(std::string(op_name) + " output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateDefinedTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateDefinedTensor(rhs, "rhs"));
  if (lhs.dtype() != rhs.dtype()) {
    return Status::InvalidArgument(std::string(op_name) + " requires matching input dtypes");
  }
  if (!IsSupportedElementwiseDType(lhs.dtype(), op)) {
    return Status::InvalidArgument(std::string(op_name) + " does not support dtype " +
                                   DTypeName(lhs.dtype()));
  }

  auto out_shape_result = BroadcastShape(lhs, rhs, op_name);
  if (!out_shape_result.ok()) {
    return out_shape_result.status();
  }
  std::vector<int64_t> out_shape = out_shape_result.value();
  if ((output == &lhs || output == &rhs) && output->shape() != out_shape) {
    return Status::InvalidArgument(std::string(op_name) +
                                   " does not support in-place output shape changes");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(output, out_shape, lhs.dtype(), ctx.stream()));
  int64_t total = output->numel();
  if (total == 0) {
    return Status::Ok();
  }

  BroadcastDescriptor desc = BuildBroadcastDescriptor(lhs, rhs, out_shape);
  auto blocks = detail::BlocksForElements(total, kTensorOpThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }

  if (IsFloatingPointDType(lhs.dtype())) {
    return LaunchTensorBinaryFloating(ctx, lhs, rhs, output, desc, total, op, blocks.value(),
                                      op_name);
  }

  if (lhs.dtype() == DType::kInt32) {
    TensorBinaryKernel<int32_t><<<blocks.value(), kTensorOpThreads, 0, ctx.stream()>>>(
        lhs.data_as<int32_t>(), rhs.data_as<int32_t>(), output->data_as<int32_t>(), desc, total,
        op);
    return detail::CheckKernelLaunch(std::string(op_name) + " kernel");
  }

  return Status::InvalidArgument(std::string(op_name) + " does not support dtype " +
                                 DTypeName(lhs.dtype()));
}

} // namespace

Status TensorAdd(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output) {
  return RunTensorBinaryOp(ctx, lhs, rhs, output, TensorBinaryOp::kAdd, "TensorAdd");
}

Status TensorSubtract(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output) {
  return RunTensorBinaryOp(ctx, lhs, rhs, output, TensorBinaryOp::kSubtract, "TensorSubtract");
}

Status TensorMultiply(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output) {
  return RunTensorBinaryOp(ctx, lhs, rhs, output, TensorBinaryOp::kMultiply, "TensorMultiply");
}

Status TensorDivide(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output) {
  return RunTensorBinaryOp(ctx, lhs, rhs, output, TensorBinaryOp::kDivide, "TensorDivide");
}

Status TensorMatMul(RuntimeContext &ctx, const Tensor &lhs, const Tensor &rhs, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("TensorMatMul output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(lhs, "lhs"));
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(rhs, "rhs"));
  if (lhs.dtype() != rhs.dtype()) {
    return Status::InvalidArgument("TensorMatMul requires matching input dtypes");
  }
  if (lhs.rank() != 2 || rhs.rank() != 2) {
    return Status::InvalidArgument("TensorMatMul requires rank-2 inputs");
  }
  int64_t rows = lhs.dim(0);
  int64_t inner = lhs.dim(1);
  int64_t rhs_inner = rhs.dim(0);
  int64_t cols = rhs.dim(1);
  if (inner != rhs_inner) {
    return Status::InvalidArgument("TensorMatMul inner dimensions must match");
  }
  if (output == &lhs || output == &rhs) {
    return Status::InvalidArgument("TensorMatMul does not support in-place output");
  }
  if (output->defined() && (output->data() == lhs.data() || output->data() == rhs.data())) {
    return Status::InvalidArgument("TensorMatMul output aliases input storage");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(output, {rows, cols}, lhs.dtype(), ctx.stream()));
  if (rows == 0 || cols == 0) {
    return Status::Ok();
  }

  auto rows_int = detail::CheckedInt(rows, "TensorMatMul rows");
  if (!rows_int.ok()) {
    return rows_int.status();
  }
  auto cols_int = detail::CheckedInt(cols, "TensorMatMul cols");
  if (!cols_int.ok()) {
    return cols_int.status();
  }
  auto inner_int = detail::CheckedInt(inner, "TensorMatMul inner");
  if (!inner_int.ok()) {
    return inner_int.status();
  }

  return LaunchTensorMatMul(ctx, lhs, rhs, output, rows_int.value(), inner_int.value(),
                            cols_int.value());
}

Status TensorReduceSum(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("TensorReduceSum output is null");
  }
  if (output == &input) {
    return Status::InvalidArgument("TensorReduceSum does not support in-place output");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "input"));
  if (output->defined() && output->data() == input.data()) {
    return Status::InvalidArgument("TensorReduceSum output aliases input storage");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(output, {1}, DType::kFloat32, ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(output->FillZero(ctx.stream()));

  int64_t n = input.numel();
  if (n == 0) {
    return Status::Ok();
  }
  auto blocks = detail::CappedBlocksForElements(n, kTensorOpThreads, kTensorReductionMaxBlocks);
  if (!blocks.ok()) {
    return blocks.status();
  }
  return LaunchTensorReduceSum(ctx, input, output, blocks.value(), n);
}

Result<float> TensorSum(RuntimeContext &ctx, const Tensor &input) {
  Tensor output;
  DLCUDA_RETURN_IF_ERROR(TensorReduceSum(ctx, input, &output));

  float host_sum = 0.0f;
  DLCUDA_RETURN_IF_ERROR(output.CopyToHost(&host_sum, sizeof(host_sum), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());
  return host_sum;
}

} // namespace dlcuda
