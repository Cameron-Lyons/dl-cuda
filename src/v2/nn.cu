#include "dl_cuda/nn.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dlcuda {
namespace {

constexpr int kCudaThreads = 256;
constexpr int kLinearTile = 16;

Status ValidateFloatTensor(const Tensor &tensor, const char *name) {
  if (!tensor.defined()) {
    return Status::InvalidArgument(std::string(name) + " is undefined");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Status::InvalidArgument(std::string(name) + " must be float32");
  }
  return Status::Ok();
}

Status ValidateIntTensor(const Tensor &tensor, const char *name) {
  if (!tensor.defined()) {
    return Status::InvalidArgument(std::string(name) + " is undefined");
  }
  if (tensor.dtype() != DType::kInt32) {
    return Status::InvalidArgument(std::string(name) + " must be int32");
  }
  return Status::Ok();
}

Status ValidateRank(const Tensor &tensor, int64_t rank, const char *name) {
  if (tensor.rank() != rank) {
    std::ostringstream oss;
    oss << name << " must have rank " << rank << ", got " << tensor.rank();
    return Status::InvalidArgument(oss.str());
  }
  return Status::Ok();
}

__global__ void AddBiasKernel(float *output, const float *bias, int64_t batch,
                              int64_t out_features) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * out_features;
  if (idx < total) {
    output[idx] += bias[idx % out_features];
  }
}

__global__ void LinearForwardKernel(const float *input, const float *weight, const float *bias,
                                    float *output, int64_t batch, int64_t in_features,
                                    int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch && col < out_features) {
    float sum = 0.0f;
    for (int64_t i = 0; i < in_features; ++i) {
      sum += input[row * in_features + i] * weight[i * out_features + col];
    }
    output[row * out_features + col] = sum + bias[col];
  }
}

__global__ void LinearBackwardInputKernel(const float *grad_output, const float *weight,
                                          float *grad_input, int64_t batch, int64_t in_features,
                                          int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch && col < in_features) {
    float sum = 0.0f;
    for (int64_t j = 0; j < out_features; ++j) {
      sum += grad_output[row * out_features + j] * weight[col * out_features + j];
    }
    grad_input[row * in_features + col] = sum;
  }
}

__global__ void LinearBackwardWeightKernel(const float *input, const float *grad_output,
                                           float *grad_weight, int64_t batch, int64_t in_features,
                                           int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < in_features && col < out_features) {
    float sum = 0.0f;
    for (int64_t n = 0; n < batch; ++n) {
      sum += input[n * in_features + row] * grad_output[n * out_features + col];
    }
    grad_weight[row * out_features + col] = sum;
  }
}

__global__ void LinearBackwardBiasKernel(const float *grad_output, float *grad_bias, int64_t batch,
                                         int64_t out_features) {
  __shared__ float shared[kCudaThreads];

  int64_t col = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (col >= out_features) {
    return;
  }

  float sum = 0.0f;
  for (int64_t n = tid; n < batch; n += blockDim.x) {
    sum += grad_output[n * out_features + col];
  }
  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    grad_bias[col] = shared[0];
  }
}

__global__ void ReLUForwardKernel(const float *input, float *output, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float v = input[idx];
    output[idx] = v > 0.0f ? v : 0.0f;
  }
}

__global__ void ReLUBackwardKernel(const float *grad_output, const float *cached_input,
                                   float *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = cached_input[idx] > 0.0f ? grad_output[idx] : 0.0f;
  }
}

__global__ void SigmoidForwardKernel(const float *input, float *output, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = input[idx];
    output[idx] = 1.0f / (1.0f + expf(-x));
  }
}

__global__ void SigmoidBackwardKernel(const float *grad_output, const float *cached_output,
                                      float *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float s = cached_output[idx];
    grad_input[idx] = grad_output[idx] * s * (1.0f - s);
  }
}

__global__ void SoftmaxForwardKernel(const float *input, float *output, int64_t num_rows,
                                     int64_t row_width) {
  __shared__ float shared[kCudaThreads];

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  const float *in_row = input + row * row_width;
  float *out_row = output + row * row_width;

  float local_max = -FLT_MAX;
  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    local_max = fmaxf(local_max, in_row[c]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  float row_max = shared[0];

  float local_sum = 0.0f;
  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    float e = expf(in_row[c] - row_max);
    out_row[c] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float inv_sum = 1.0f / (shared[0] + 1e-20f);

  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    out_row[c] *= inv_sum;
  }
}

__global__ void SoftmaxBackwardKernel(const float *grad_output, const float *softmax_output,
                                      float *grad_input, int64_t num_rows, int64_t row_width) {
  __shared__ float shared[kCudaThreads];

  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  const float *dy = grad_output + row * row_width;
  const float *s = softmax_output + row * row_width;
  float *dx = grad_input + row * row_width;

  float local_dot = 0.0f;
  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    local_dot += dy[c] * s[c];
  }
  shared[tid] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float dot = shared[0];

  for (int64_t c = tid; c < row_width; c += blockDim.x) {
    dx[c] = s[c] * (dy[c] - dot);
  }
}

__global__ void EmbeddingForwardKernel(const float *table, const int32_t *token_ids, float *output,
                                       int64_t num_tokens, int64_t embedding_dim,
                                       int64_t vocab_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_tokens * embedding_dim;
  if (idx < total) {
    int64_t token = idx / embedding_dim;
    int64_t dim = idx % embedding_dim;
    int32_t token_id = token_ids[token];
    if (token_id >= 0 && static_cast<int64_t>(token_id) < vocab_size) {
      output[idx] = table[static_cast<int64_t>(token_id) * embedding_dim + dim];
    } else {
      output[idx] = 0.0f;
    }
  }
}

__global__ void EmbeddingBackwardKernel(const float *grad_output, const int32_t *token_ids,
                                        float *grad_table, int64_t num_tokens,
                                        int64_t embedding_dim, int64_t vocab_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_tokens * embedding_dim;
  if (idx < total) {
    int64_t token = idx / embedding_dim;
    int64_t dim = idx % embedding_dim;
    int32_t token_id = token_ids[token];
    if (token_id >= 0 && static_cast<int64_t>(token_id) < vocab_size) {
      atomicAdd(&grad_table[static_cast<int64_t>(token_id) * embedding_dim + dim],
                grad_output[idx]);
    }
  }
}

Status EnsureSameShapeAndType(const Tensor &a, const Tensor &b, const char *a_name,
                              const char *b_name) {
  if (a.dtype() != b.dtype()) {
    return Status::InvalidArgument(std::string(a_name) + " and " + b_name + " dtype mismatch");
  }
  if (a.shape() != b.shape()) {
    return Status::InvalidArgument(std::string(a_name) + " and " + b_name + " shape mismatch");
  }
  return Status::Ok();
}

std::string JoinParameterName(const std::string &prefix, const char *name) {
  if (prefix.empty()) {
    return std::string(name);
  }
  return prefix + "." + name;
}

} // namespace

Status Sequential::Add(std::unique_ptr<Module> module) {
  if (!module) {
    return Status::InvalidArgument("Sequential::Add received null module");
  }
  modules_.push_back(std::move(module));
  RebuildParameterCache();
  return Status::Ok();
}

Status Sequential::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Forward output pointer is null");
  }

  Tensor current = input;
  for (size_t i = 0; i < modules_.size(); ++i) {
    Tensor next;
    Status status = modules_[i]->Forward(ctx, current, &next);
    if (!status.ok()) {
      return Status::RuntimeError("Forward failed in module " + std::to_string(i) + " (" +
                                  modules_[i]->Name() + "): " + status.message());
    }
    if (!next.defined()) {
      return Status::RuntimeError("Forward output became undefined in module " + std::to_string(i) +
                                  " (" + modules_[i]->Name() + ")");
    }
    current = next;
  }

  *output = current;
  return Status::Ok();
}

Status Sequential::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }

  Tensor current = grad_output;
  for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
    Tensor next;
    Tensor *next_out = (i == 0 && grad_input == nullptr) ? nullptr : &next;
    Status status = modules_[static_cast<size_t>(i)]->Backward(ctx, current, next_out);
    if (!status.ok()) {
      return Status::RuntimeError("Backward failed in module " + std::to_string(i) + " (" +
                                  modules_[static_cast<size_t>(i)]->Name() +
                                  "): " + status.message());
    }
    if (i > 0 && !next.defined()) {
      return Status::RuntimeError("Backward gradient became undefined before first module");
    }
    current = next;
  }

  if (grad_input != nullptr) {
    *grad_input = current;
  }
  return Status::Ok();
}

void Sequential::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  for (size_t i = 0; i < modules_.size(); ++i) {
    std::string child_name = "layers." + std::to_string(i);
    std::string child_prefix = prefix.empty() ? child_name : prefix + "." + child_name;
    modules_[i]->AppendParameters(child_prefix, out);
  }
}

void Sequential::RebuildParameterCache() {
  parameter_cache_.clear();
  AppendParameters("", &parameter_cache_);
}

Linear::Linear(int64_t in_features, int64_t out_features, RuntimeContext &ctx)
    : in_features_(in_features), out_features_(out_features) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    init_status_ = Status::InvalidArgument("Linear dimensions must be positive");
    return;
  }

  auto weight = Tensor::Allocate({in_features_, out_features_}, DType::kFloat32);
  if (!weight.ok()) {
    init_status_ = weight.status();
    return;
  }
  auto bias = Tensor::Allocate({out_features_}, DType::kFloat32);
  if (!bias.ok()) {
    init_status_ = bias.status();
    return;
  }
  auto grad_weight = Tensor::Allocate({in_features_, out_features_}, DType::kFloat32);
  if (!grad_weight.ok()) {
    init_status_ = grad_weight.status();
    return;
  }
  auto grad_bias = Tensor::Allocate({out_features_}, DType::kFloat32);
  if (!grad_bias.ok()) {
    init_status_ = grad_bias.status();
    return;
  }

  weight_ = weight.value();
  bias_ = bias.value();
  grad_weight_ = grad_weight.value();
  grad_bias_ = grad_bias.value();

  std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features_));

  std::vector<float> host_weight(static_cast<size_t>(in_features_ * out_features_));
  for (float &v : host_weight) {
    v = dist(rng);
  }

  init_status_ =
      weight_.CopyFromHost(host_weight.data(), host_weight.size() * sizeof(float), ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = bias_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_weight_.FillZero(ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_bias_.FillZero(ctx.stream());
}

Status Linear::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Linear::Forward output is null");
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "Linear input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Linear input"));

  int64_t batch = input.dim(0);
  int64_t in_features = input.dim(1);
  if (in_features != in_features_) {
    std::ostringstream oss;
    oss << "Linear input feature mismatch: expected " << in_features_ << " got " << in_features;
    return Status::InvalidArgument(oss.str());
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&forward_output_, {batch, out_features_}, DType::kFloat32));
  cached_input_ = input;
  last_batch_ = batch;
  if (batch == 0) {
    *output = forward_output_;
    return Status::Ok();
  }

  if (ctx.use_cublas()) {
    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();

    auto out_features_int = detail::CheckedInt(out_features_, "out_features");
    if (!out_features_int.ok()) {
      return out_features_int.status();
    }
    auto batch_int = detail::CheckedInt(batch, "batch");
    if (!batch_int.ok()) {
      return batch_int.status();
    }
    auto in_features_int = detail::CheckedInt(in_features_, "in_features");
    if (!in_features_int.ok()) {
      return in_features_int.status();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features_int.value(), batch_int.value(),
                    in_features_int.value(), &alpha, weight_.data_as<float>(),
                    out_features_int.value(), input.data_as<float>(), in_features_int.value(),
                    &beta, forward_output_.data_as<float>(), out_features_int.value()),
        "Linear forward cublasSgemm"));

    int64_t total = batch * out_features_;
    auto blocks = detail::BlocksForElements(total, kCudaThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    AddBiasKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        forward_output_.data_as<float>(), bias_.data_as<float>(), batch, out_features_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear add-bias kernel"));
  } else {
    auto x_blocks = detail::BlocksForElements(out_features_, kLinearTile);
    if (!x_blocks.ok()) {
      return x_blocks.status();
    }
    auto y_blocks = detail::BlocksForElements(batch, kLinearTile);
    if (!y_blocks.ok()) {
      return y_blocks.status();
    }
    dim3 threads(kLinearTile, kLinearTile);
    dim3 blocks(static_cast<unsigned int>(x_blocks.value()),
                static_cast<unsigned int>(y_blocks.value()));
    LinearForwardKernel<<<blocks, threads, 0, ctx.stream()>>>(
        input.data_as<float>(), weight_.data_as<float>(), bias_.data_as<float>(),
        forward_output_.data_as<float>(), batch, in_features_, out_features_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear forward kernel"));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Linear::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  bool need_grad_input = grad_input != nullptr;

  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Linear grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Linear grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != out_features_) {
    return Status::InvalidArgument("Linear grad_output shape mismatch");
  }
  if (!cached_input_.defined()) {
    return Status::RuntimeError("Linear backward called before forward");
  }

  if (need_grad_input) {
    DLCUDA_RETURN_IF_ERROR(
        EnsureTensor(&backward_output_, {last_batch_, in_features_}, DType::kFloat32));
  }
  if (last_batch_ == 0) {
    DLCUDA_RETURN_IF_ERROR(grad_weight_.FillZero(ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(grad_bias_.FillZero(ctx.stream()));
    if (need_grad_input) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }

  if (ctx.use_cublas()) {
    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();

    auto in_features_int = detail::CheckedInt(in_features_, "in_features");
    if (!in_features_int.ok()) {
      return in_features_int.status();
    }
    auto batch_int = detail::CheckedInt(last_batch_, "batch");
    if (!batch_int.ok()) {
      return batch_int.status();
    }
    auto out_features_int = detail::CheckedInt(out_features_, "out_features");
    if (!out_features_int.ok()) {
      return out_features_int.status();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (need_grad_input) {
      DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
          cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features_int.value(), batch_int.value(),
                      out_features_int.value(), &alpha, weight_.data_as<float>(),
                      out_features_int.value(), grad_output.data_as<float>(),
                      out_features_int.value(), &beta, backward_output_.data_as<float>(),
                      in_features_int.value()),
          "Linear backward-input cublasSgemm"));
    }

    DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features_int.value(),
                    in_features_int.value(), batch_int.value(), &alpha,
                    grad_output.data_as<float>(), out_features_int.value(),
                    cached_input_.data_as<float>(), in_features_int.value(), &beta,
                    grad_weight_.data_as<float>(), out_features_int.value()),
        "Linear backward-weight cublasSgemm"));

    auto rows = detail::RowsForGrid(out_features_, "linear bias");
    if (!rows.ok()) {
      return rows.status();
    }
    LinearBackwardBiasKernel<<<rows.value(), kCudaThreads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), grad_bias_.data_as<float>(), last_batch_, out_features_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear backward-bias kernel"));
  } else {
    dim3 threads(kLinearTile, kLinearTile);
    if (need_grad_input) {
      auto input_x_blocks = detail::BlocksForElements(in_features_, kLinearTile);
      if (!input_x_blocks.ok()) {
        return input_x_blocks.status();
      }
      auto input_y_blocks = detail::BlocksForElements(last_batch_, kLinearTile);
      if (!input_y_blocks.ok()) {
        return input_y_blocks.status();
      }
      dim3 blocks_input(static_cast<unsigned int>(input_x_blocks.value()),
                        static_cast<unsigned int>(input_y_blocks.value()));
      LinearBackwardInputKernel<<<blocks_input, threads, 0, ctx.stream()>>>(
          grad_output.data_as<float>(), weight_.data_as<float>(), backward_output_.data_as<float>(),
          last_batch_, in_features_, out_features_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear backward-input kernel"));
    }

    auto weight_x_blocks = detail::BlocksForElements(out_features_, kLinearTile);
    if (!weight_x_blocks.ok()) {
      return weight_x_blocks.status();
    }
    auto weight_y_blocks = detail::BlocksForElements(in_features_, kLinearTile);
    if (!weight_y_blocks.ok()) {
      return weight_y_blocks.status();
    }
    dim3 blocks_weight(static_cast<unsigned int>(weight_x_blocks.value()),
                       static_cast<unsigned int>(weight_y_blocks.value()));
    LinearBackwardWeightKernel<<<blocks_weight, threads, 0, ctx.stream()>>>(
        cached_input_.data_as<float>(), grad_output.data_as<float>(), grad_weight_.data_as<float>(),
        last_batch_, in_features_, out_features_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear backward-weight kernel"));

    auto bias_rows = detail::RowsForGrid(out_features_, "linear bias");
    if (!bias_rows.ok()) {
      return bias_rows.status();
    }
    LinearBackwardBiasKernel<<<bias_rows.value(), kCudaThreads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), grad_bias_.data_as<float>(), last_batch_, out_features_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear backward-bias kernel"));
  }

  if (need_grad_input) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void Linear::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "weight"), &weight_, &grad_weight_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "bias"), &bias_, &grad_bias_});
}

Status ReLU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("ReLU::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "ReLU input"));

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&forward_output_, input.shape(), DType::kFloat32));
  cached_input_ = input;

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    ReLUForwardKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        input.data_as<float>(), forward_output_.data_as<float>(), input.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ReLU forward kernel"));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status ReLU::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "ReLU grad_output"));
  if (!cached_input_.defined()) {
    return Status::RuntimeError("ReLU backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_input_, "grad_output", "cached_input"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&backward_output_, grad_output.shape(), DType::kFloat32));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    ReLUBackwardKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), cached_input_.data_as<float>(),
        backward_output_.data_as<float>(), grad_output.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ReLU backward kernel"));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void ReLU::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Sigmoid::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Sigmoid::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "Sigmoid input"));

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&cached_output_, input.shape(), DType::kFloat32));

  auto blocks = detail::BlocksForElements(input.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    SigmoidForwardKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        input.data_as<float>(), cached_output_.data_as<float>(), input.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Sigmoid forward kernel"));
  }

  *output = cached_output_;
  return Status::Ok();
}

Status Sigmoid::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Sigmoid grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Sigmoid backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_output_, "grad_output", "cached_output"));
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&backward_output_, grad_output.shape(), DType::kFloat32));
  auto blocks = detail::BlocksForElements(grad_output.numel(), kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    SigmoidBackwardKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), cached_output_.data_as<float>(),
        backward_output_.data_as<float>(), grad_output.numel());
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Sigmoid backward kernel"));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Sigmoid::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Softmax::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Softmax::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "Softmax input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Softmax input"));

  num_rows_ = input.dim(0);
  row_width_ = input.dim(1);
  if (num_rows_ > 0 && row_width_ == 0) {
    return Status::InvalidArgument("Softmax row width must be positive");
  }

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&cached_output_, input.shape(), DType::kFloat32));

  auto rows = detail::RowsForGrid(num_rows_, "softmax");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    SoftmaxForwardKernel<<<rows.value(), kCudaThreads, 0, ctx.stream()>>>(
        input.data_as<float>(), cached_output_.data_as<float>(), num_rows_, row_width_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Softmax forward kernel"));
  }

  *output = cached_output_;
  return Status::Ok();
}

Status Softmax::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Softmax grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Softmax backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_output_, "grad_output", "cached_output"));

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&backward_output_, grad_output.shape(), DType::kFloat32));
  if (num_rows_ > 0 && row_width_ == 0) {
    return Status::InvalidArgument("Softmax row width must be positive");
  }
  if (grad_input == nullptr) {
    return Status::Ok();
  }

  auto rows = detail::RowsForGrid(num_rows_, "softmax");
  if (!rows.ok()) {
    return rows.status();
  }
  if (rows.value() > 0) {
    SoftmaxBackwardKernel<<<rows.value(), kCudaThreads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), cached_output_.data_as<float>(),
        backward_output_.data_as<float>(), num_rows_, row_width_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Softmax backward kernel"));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Softmax::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Embedding::Embedding(int64_t vocab_size, int64_t embedding_dim, RuntimeContext &ctx)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
  if (vocab_size_ <= 0 || embedding_dim_ <= 0) {
    init_status_ = Status::InvalidArgument("Embedding dimensions must be positive");
    return;
  }

  auto table = Tensor::Allocate({vocab_size_, embedding_dim_}, DType::kFloat32);
  if (!table.ok()) {
    init_status_ = table.status();
    return;
  }
  auto grad_table = Tensor::Allocate({vocab_size_, embedding_dim_}, DType::kFloat32);
  if (!grad_table.ok()) {
    init_status_ = grad_table.status();
    return;
  }

  table_ = table.value();
  grad_table_ = grad_table.value();

  std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embedding_dim_));
  std::vector<float> host_table(static_cast<size_t>(vocab_size_ * embedding_dim_));
  for (float &v : host_table) {
    v = dist(rng);
  }

  init_status_ =
      table_.CopyFromHost(host_table.data(), host_table.size() * sizeof(float), ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_table_.FillZero(ctx.stream());
}

Status Embedding::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Embedding::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateIntTensor(input, "Embedding input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 1, "Embedding input"));

  last_num_tokens_ = input.dim(0);

  cached_token_ids_ = input;

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&forward_output_, {last_num_tokens_, embedding_dim_}, DType::kFloat32));

  int64_t total = last_num_tokens_ * embedding_dim_;
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    EmbeddingForwardKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        table_.data_as<float>(), cached_token_ids_.data_as<int32_t>(),
        forward_output_.data_as<float>(), last_num_tokens_, embedding_dim_, vocab_size_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Embedding forward kernel"));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Embedding::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Embedding grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Embedding grad_output"));
  if (grad_output.dim(0) != last_num_tokens_ || grad_output.dim(1) != embedding_dim_) {
    return Status::InvalidArgument("Embedding grad_output shape mismatch");
  }
  if (!cached_token_ids_.defined()) {
    return Status::RuntimeError("Embedding backward called before forward");
  }

  DLCUDA_RETURN_IF_ERROR(grad_table_.FillZero(ctx.stream()));

  int64_t total = last_num_tokens_ * embedding_dim_;
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    EmbeddingBackwardKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), cached_token_ids_.data_as<int32_t>(),
        grad_table_.data_as<float>(), last_num_tokens_, embedding_dim_, vocab_size_);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Embedding backward kernel"));
  }

  // Token IDs are non-differentiable; upstream gradient terminates here.
  if (grad_input != nullptr) {
    *grad_input = Tensor();
  }
  return Status::Ok();
}

void Embedding::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "table"), &table_, &grad_table_});
}

} // namespace dlcuda
