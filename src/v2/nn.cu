#include "dl_cuda/nn.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dlcuda {
namespace {

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

Status FromCuda(cudaError_t err, const std::string &context) {
  if (err == cudaSuccess) {
    return Status::Ok();
  }
  return Status::RuntimeError(context + ": " + cudaGetErrorString(err));
}

Status FromCublas(cublasStatus_t status, const std::string &context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  std::ostringstream oss;
  oss << context << " failed with cuBLAS status code " << static_cast<int>(status);
  return Status::RuntimeError(oss.str());
}

__global__ void AddBiasKernel(float *output, const float *bias, int64_t batch,
                              int64_t out_features) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * out_features;
  if (idx < total) {
    output[idx] += bias[idx % out_features];
  }
}

__global__ void LinearForwardKernel(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    int64_t batch, int64_t in_features,
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

__global__ void LinearBackwardInputKernel(const float *grad_output,
                                          const float *weight,
                                          float *grad_input, int64_t batch,
                                          int64_t in_features,
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

__global__ void LinearBackwardWeightKernel(const float *input,
                                           const float *grad_output,
                                           float *grad_weight, int64_t batch,
                                           int64_t in_features,
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

__global__ void LinearBackwardBiasKernel(const float *grad_output,
                                         float *grad_bias, int64_t batch,
                                         int64_t out_features) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < out_features) {
    float sum = 0.0f;
    for (int64_t n = 0; n < batch; ++n) {
      sum += grad_output[n * out_features + col];
    }
    grad_bias[col] = sum;
  }
}

__global__ void ReLUForwardKernel(const float *input, float *output,
                                  int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float v = input[idx];
    output[idx] = v > 0.0f ? v : 0.0f;
  }
}

__global__ void ReLUBackwardKernel(const float *grad_output,
                                   const float *cached_input,
                                   float *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] = cached_input[idx] > 0.0f ? grad_output[idx] : 0.0f;
  }
}

__global__ void SigmoidForwardKernel(const float *input, float *output,
                                     int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = input[idx];
    output[idx] = 1.0f / (1.0f + expf(-x));
  }
}

__global__ void SigmoidBackwardKernel(const float *grad_output,
                                      const float *cached_output,
                                      float *grad_input, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    float s = cached_output[idx];
    grad_input[idx] = grad_output[idx] * s * (1.0f - s);
  }
}

__global__ void SoftmaxForwardKernel(const float *input, float *output,
                                     int64_t num_rows, int64_t row_width) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *in_row = input + row * row_width;
    float *out_row = output + row * row_width;

    float row_max = in_row[0];
    for (int64_t c = 1; c < row_width; ++c) {
      row_max = fmaxf(row_max, in_row[c]);
    }

    float sum = 0.0f;
    for (int64_t c = 0; c < row_width; ++c) {
      float e = expf(in_row[c] - row_max);
      out_row[c] = e;
      sum += e;
    }

    float inv_sum = 1.0f / (sum + 1e-20f);
    for (int64_t c = 0; c < row_width; ++c) {
      out_row[c] *= inv_sum;
    }
  }
}

__global__ void SoftmaxBackwardKernel(const float *grad_output,
                                      const float *softmax_output,
                                      float *grad_input, int64_t num_rows,
                                      int64_t row_width) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *dy = grad_output + row * row_width;
    const float *s = softmax_output + row * row_width;
    float *dx = grad_input + row * row_width;

    float dot = 0.0f;
    for (int64_t c = 0; c < row_width; ++c) {
      dot += dy[c] * s[c];
    }

    for (int64_t c = 0; c < row_width; ++c) {
      dx[c] = s[c] * (dy[c] - dot);
    }
  }
}

__global__ void EmbeddingForwardKernel(const float *table, const int32_t *token_ids,
                                       float *output, int64_t num_tokens,
                                       int64_t embedding_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_tokens * embedding_dim;
  if (idx < total) {
    int64_t token = idx / embedding_dim;
    int64_t dim = idx % embedding_dim;
    int32_t token_id = token_ids[token];
    output[idx] = table[static_cast<int64_t>(token_id) * embedding_dim + dim];
  }
}

__global__ void EmbeddingBackwardKernel(const float *grad_output,
                                        const int32_t *token_ids,
                                        float *grad_table,
                                        int64_t num_tokens,
                                        int64_t embedding_dim,
                                        int64_t vocab_size) {
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

Status EnsureSameShapeAndType(const Tensor &a, const Tensor &b,
                              const char *a_name, const char *b_name) {
  if (a.dtype() != b.dtype()) {
    return Status::InvalidArgument(std::string(a_name) + " and " + b_name +
                                   " dtype mismatch");
  }
  if (a.shape() != b.shape()) {
    return Status::InvalidArgument(std::string(a_name) + " and " + b_name +
                                   " shape mismatch");
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

Status Sequential::Forward(RuntimeContext &ctx, const Tensor &input,
                           Tensor *output) {
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
      return Status::RuntimeError("Forward failed in module " +
                                  std::to_string(i) + " (" +
                                  modules_[i]->Name() + "): " +
                                  status.message());
    }
    current = next;
  }

  *output = current;
  return Status::Ok();
}

Status Sequential::Backward(RuntimeContext &ctx, const Tensor &grad_output,
                            Tensor *grad_input) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }

  Tensor current = grad_output;
  for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
    Tensor next;
    Status status = modules_[static_cast<size_t>(i)]->Backward(ctx, current, &next);
    if (!status.ok()) {
      return Status::RuntimeError("Backward failed in module " +
                                  std::to_string(i) + " (" +
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

void Sequential::AppendParameters(const std::string &prefix,
                                  std::vector<ParameterRef> *out) {
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

  init_status_ = weight_.CopyFromHost(host_weight.data(), host_weight.size() * sizeof(float),
                                      ctx.stream());
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
    oss << "Linear input feature mismatch: expected " << in_features_
        << " got " << in_features;
    return Status::InvalidArgument(oss.str());
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&forward_output_, {batch, out_features_}, DType::kFloat32));
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&cached_input_, input.shape(), DType::kFloat32));
  last_batch_ = batch;

  DLCUDA_RETURN_IF_ERROR(CopyTensor(input, &cached_input_, ctx.stream()));

  if (ctx.use_cublas()) {
    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();
    DLCUDA_RETURN_IF_ERROR(
        FromCublas(cublasSetStream(handle, ctx.stream()), "cublasSetStream"));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    DLCUDA_RETURN_IF_ERROR(FromCublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(out_features_), static_cast<int>(batch),
                    static_cast<int>(in_features_), &alpha,
                    weight_.data_as<float>(), static_cast<int>(out_features_),
                    input.data_as<float>(), static_cast<int>(in_features_), &beta,
                    forward_output_.data_as<float>(),
                    static_cast<int>(out_features_)),
        "Linear forward cublasSgemm"));

    int64_t total = batch * out_features_;
    int blocks = static_cast<int>((total + 255) / 256);
    AddBiasKernel<<<blocks, 256, 0, ctx.stream()>>>(
        forward_output_.data_as<float>(), bias_.data_as<float>(), batch,
        out_features_);
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Linear add-bias kernel"));
  } else {
    dim3 threads(16, 16);
    dim3 blocks(static_cast<unsigned int>((out_features_ + threads.x - 1) / threads.x),
                static_cast<unsigned int>((batch + threads.y - 1) / threads.y));
    LinearForwardKernel<<<blocks, threads, 0, ctx.stream()>>>(
        input.data_as<float>(), weight_.data_as<float>(), bias_.data_as<float>(),
        forward_output_.data_as<float>(), batch, in_features_, out_features_);
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Linear forward kernel"));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Linear::Backward(RuntimeContext &ctx, const Tensor &grad_output,
                        Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (grad_input == nullptr) {
    return Status::InvalidArgument("Linear::Backward grad_input is null");
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Linear grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Linear grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != out_features_) {
    return Status::InvalidArgument("Linear grad_output shape mismatch");
  }
  if (!cached_input_.defined()) {
    return Status::RuntimeError("Linear backward called before forward");
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&backward_output_, {last_batch_, in_features_}, DType::kFloat32));

  if (ctx.use_cublas()) {
    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();
    DLCUDA_RETURN_IF_ERROR(
        FromCublas(cublasSetStream(handle, ctx.stream()), "cublasSetStream"));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    DLCUDA_RETURN_IF_ERROR(FromCublas(
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(in_features_), static_cast<int>(last_batch_),
                    static_cast<int>(out_features_), &alpha,
                    weight_.data_as<float>(), static_cast<int>(out_features_),
                    grad_output.data_as<float>(), static_cast<int>(out_features_),
                    &beta, backward_output_.data_as<float>(),
                    static_cast<int>(in_features_)),
        "Linear backward-input cublasSgemm"));

    DLCUDA_RETURN_IF_ERROR(FromCublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    static_cast<int>(out_features_), static_cast<int>(in_features_),
                    static_cast<int>(last_batch_), &alpha,
                    grad_output.data_as<float>(), static_cast<int>(out_features_),
                    cached_input_.data_as<float>(), static_cast<int>(in_features_),
                    &beta, grad_weight_.data_as<float>(),
                    static_cast<int>(out_features_)),
        "Linear backward-weight cublasSgemm"));

    int blocks = static_cast<int>((out_features_ + 255) / 256);
    LinearBackwardBiasKernel<<<blocks, 256, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), grad_bias_.data_as<float>(), last_batch_,
        out_features_);
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Linear backward-bias kernel"));
  } else {
    dim3 threads(16, 16);
    dim3 blocks_input(
        static_cast<unsigned int>((in_features_ + threads.x - 1) / threads.x),
        static_cast<unsigned int>((last_batch_ + threads.y - 1) / threads.y));
    LinearBackwardInputKernel<<<blocks_input, threads, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), weight_.data_as<float>(),
        backward_output_.data_as<float>(), last_batch_, in_features_,
        out_features_);
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Linear backward-input kernel"));

    dim3 blocks_weight(
        static_cast<unsigned int>((out_features_ + threads.x - 1) / threads.x),
        static_cast<unsigned int>((in_features_ + threads.y - 1) / threads.y));
    LinearBackwardWeightKernel<<<blocks_weight, threads, 0, ctx.stream()>>>(
        cached_input_.data_as<float>(), grad_output.data_as<float>(),
        grad_weight_.data_as<float>(), last_batch_, in_features_, out_features_);
    DLCUDA_RETURN_IF_ERROR(
        FromCuda(cudaGetLastError(), "Linear backward-weight kernel"));

    int blocks_bias = static_cast<int>((out_features_ + 255) / 256);
    LinearBackwardBiasKernel<<<blocks_bias, 256, 0, ctx.stream()>>>(
        grad_output.data_as<float>(), grad_bias_.data_as<float>(), last_batch_,
        out_features_);
    DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Linear backward-bias kernel"));
  }

  *grad_input = backward_output_;
  return Status::Ok();
}

void Linear::AppendParameters(const std::string &prefix,
                              std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "weight"), &weight_,
                              &grad_weight_});
  out->push_back(
      ParameterRef{JoinParameterName(prefix, "bias"), &bias_, &grad_bias_});
}

Status ReLU::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("ReLU::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "ReLU input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&forward_output_, input.shape(), DType::kFloat32));
  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&cached_input_, input.shape(), DType::kFloat32));

  DLCUDA_RETURN_IF_ERROR(CopyTensor(input, &cached_input_, ctx.stream()));

  int blocks = static_cast<int>((input.numel() + 255) / 256);
  ReLUForwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      input.data_as<float>(), forward_output_.data_as<float>(), input.numel());
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "ReLU forward kernel"));

  *output = forward_output_;
  return Status::Ok();
}

Status ReLU::Backward(RuntimeContext &ctx, const Tensor &grad_output,
                      Tensor *grad_input) {
  if (grad_input == nullptr) {
    return Status::InvalidArgument("ReLU::Backward grad_input is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "ReLU grad_output"));
  if (!cached_input_.defined()) {
    return Status::RuntimeError("ReLU backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, cached_input_, "grad_output", "cached_input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&backward_output_, grad_output.shape(), DType::kFloat32));
  int blocks = static_cast<int>((grad_output.numel() + 255) / 256);
  ReLUBackwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      grad_output.data_as<float>(), cached_input_.data_as<float>(),
      backward_output_.data_as<float>(), grad_output.numel());
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "ReLU backward kernel"));

  *grad_input = backward_output_;
  return Status::Ok();
}

void ReLU::AppendParameters(const std::string &prefix,
                            std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Sigmoid::Forward(RuntimeContext &ctx, const Tensor &input,
                        Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Sigmoid::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "Sigmoid input"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&cached_output_, input.shape(), DType::kFloat32));

  int blocks = static_cast<int>((input.numel() + 255) / 256);
  SigmoidForwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      input.data_as<float>(), cached_output_.data_as<float>(), input.numel());
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Sigmoid forward kernel"));

  *output = cached_output_;
  return Status::Ok();
}

Status Sigmoid::Backward(RuntimeContext &ctx, const Tensor &grad_output,
                         Tensor *grad_input) {
  if (grad_input == nullptr) {
    return Status::InvalidArgument("Sigmoid::Backward grad_input is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Sigmoid grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Sigmoid backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(EnsureSameShapeAndType(grad_output, cached_output_,
                                                "grad_output", "cached_output"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&backward_output_, grad_output.shape(), DType::kFloat32));
  int blocks = static_cast<int>((grad_output.numel() + 255) / 256);
  SigmoidBackwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      grad_output.data_as<float>(), cached_output_.data_as<float>(),
      backward_output_.data_as<float>(), grad_output.numel());
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Sigmoid backward kernel"));

  *grad_input = backward_output_;
  return Status::Ok();
}

void Sigmoid::AppendParameters(const std::string &prefix,
                               std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Status Softmax::Forward(RuntimeContext &ctx, const Tensor &input,
                        Tensor *output) {
  if (output == nullptr) {
    return Status::InvalidArgument("Softmax::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(input, "Softmax input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Softmax input"));

  num_rows_ = input.dim(0);
  row_width_ = input.dim(1);

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&cached_output_, input.shape(), DType::kFloat32));

  int blocks = static_cast<int>((num_rows_ + 255) / 256);
  SoftmaxForwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      input.data_as<float>(), cached_output_.data_as<float>(), num_rows_,
      row_width_);
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Softmax forward kernel"));

  *output = cached_output_;
  return Status::Ok();
}

Status Softmax::Backward(RuntimeContext &ctx, const Tensor &grad_output,
                         Tensor *grad_input) {
  if (grad_input == nullptr) {
    return Status::InvalidArgument("Softmax::Backward grad_input is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateFloatTensor(grad_output, "Softmax grad_output"));
  if (!cached_output_.defined()) {
    return Status::RuntimeError("Softmax backward called before forward");
  }
  DLCUDA_RETURN_IF_ERROR(EnsureSameShapeAndType(grad_output, cached_output_,
                                                "grad_output", "cached_output"));

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&backward_output_, grad_output.shape(), DType::kFloat32));

  int blocks = static_cast<int>((num_rows_ + 255) / 256);
  SoftmaxBackwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      grad_output.data_as<float>(), cached_output_.data_as<float>(),
      backward_output_.data_as<float>(), num_rows_, row_width_);
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Softmax backward kernel"));

  *grad_input = backward_output_;
  return Status::Ok();
}

void Softmax::AppendParameters(const std::string &prefix,
                               std::vector<ParameterRef> *out) {
  (void)prefix;
  (void)out;
}

Embedding::Embedding(int64_t vocab_size, int64_t embedding_dim,
                     RuntimeContext &ctx)
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

  init_status_ = table_.CopyFromHost(host_table.data(), host_table.size() * sizeof(float),
                                     ctx.stream());
  if (!init_status_.ok()) {
    return;
  }
  init_status_ = grad_table_.FillZero(ctx.stream());
}

Status Embedding::Forward(RuntimeContext &ctx, const Tensor &input,
                          Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Embedding::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateIntTensor(input, "Embedding input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 1, "Embedding input"));

  last_num_tokens_ = input.dim(0);

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensor(&cached_token_ids_, input.shape(), DType::kInt32));
  DLCUDA_RETURN_IF_ERROR(CopyTensor(input, &cached_token_ids_, ctx.stream()));

  DLCUDA_RETURN_IF_ERROR(EnsureTensor(&forward_output_,
                                      {last_num_tokens_, embedding_dim_},
                                      DType::kFloat32));

  int64_t total = last_num_tokens_ * embedding_dim_;
  int blocks = static_cast<int>((total + 255) / 256);
  EmbeddingForwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      table_.data_as<float>(), cached_token_ids_.data_as<int32_t>(),
      forward_output_.data_as<float>(), last_num_tokens_, embedding_dim_);
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Embedding forward kernel"));

  *output = forward_output_;
  return Status::Ok();
}

Status Embedding::Backward(RuntimeContext &ctx, const Tensor &grad_output,
                           Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (grad_input == nullptr) {
    return Status::InvalidArgument("Embedding::Backward grad_input is null");
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
  int blocks = static_cast<int>((total + 255) / 256);
  EmbeddingBackwardKernel<<<blocks, 256, 0, ctx.stream()>>>(
      grad_output.data_as<float>(), cached_token_ids_.data_as<int32_t>(),
      grad_table_.data_as<float>(), last_num_tokens_, embedding_dim_, vocab_size_);
  DLCUDA_RETURN_IF_ERROR(FromCuda(cudaGetLastError(), "Embedding backward kernel"));

  // Token IDs are non-differentiable; upstream gradient terminates here.
  *grad_input = Tensor();
  return Status::Ok();
}

void Embedding::AppendParameters(const std::string &prefix,
                                 std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(
      ParameterRef{JoinParameterName(prefix, "table"), &table_, &grad_table_});
}

} // namespace dlcuda
