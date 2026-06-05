#include "detail/embedding_kernels.cuh"

namespace dlcuda {
namespace {

constexpr int64_t kEmbeddingWarpAggregationMinTokens = 32;

} // namespace

Embedding::Embedding(int64_t vocab_size, int64_t embedding_dim, RuntimeContext &ctx, DType dtype)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim), dtype_(dtype) {
  if (vocab_size_ <= 0 || embedding_dim_ <= 0) {
    init_status_ = Status::InvalidArgument("Embedding dimensions must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "Embedding");
  if (!init_status_.ok()) {
    return;
  }

  auto table = Tensor::AllocateAsync({vocab_size_, embedding_dim_}, dtype_, ctx.stream());
  if (!table.ok()) {
    init_status_ = table.status();
    return;
  }
  auto grad_table =
      Tensor::AllocateAsync({vocab_size_, embedding_dim_}, DType::kFloat32, ctx.stream());
  if (!grad_table.ok()) {
    init_status_ = grad_table.status();
    return;
  }

  table_ = table.value();
  grad_table_ = grad_table.value();

  init_status_ = FillKaimingNormal(ctx, &table_, static_cast<float>(embedding_dim_));
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

  DLCUDA_RETURN_IF_ERROR(EnsureTensorAsync(&forward_output_, {last_num_tokens_, embedding_dim_},
                                           dtype_, ctx.stream()));

  int64_t total = last_num_tokens_ * embedding_dim_;
  auto blocks = detail::BlocksForElements(total, kCudaThreads);
  if (!blocks.ok()) {
    return blocks.status();
  }
  if (blocks.value() > 0) {
    DLCUDA_RETURN_IF_ERROR(LaunchEmbeddingForwardKernel(
        ctx, dtype_, table_, cached_token_ids_, &forward_output_, blocks.value(), last_num_tokens_,
        embedding_dim_, vocab_size_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Embedding::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Embedding grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "Embedding grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Embedding grad_output"));
  if (grad_output.dim(0) != last_num_tokens_ || grad_output.dim(1) != embedding_dim_) {
    return Status::InvalidArgument("Embedding grad_output shape mismatch");
  }
  if (!cached_token_ids_.defined()) {
    return Status::RuntimeError("Embedding backward called before forward");
  }

  DLCUDA_RETURN_IF_ERROR(grad_table_.FillZero(ctx.stream()));

  int64_t total = last_num_tokens_ * embedding_dim_;
  bool used_warp_aggregation = false;
  if (last_num_tokens_ >= kEmbeddingWarpAggregationMinTokens) {
    Status warp_status = LaunchEmbeddingBackwardWarpAggregatedKernel(
        ctx, dtype_, grad_output, cached_token_ids_, &grad_table_, last_num_tokens_, embedding_dim_,
        vocab_size_);
    if (warp_status.ok()) {
      used_warp_aggregation = true;
    } else if (warp_status.code() != StatusCode::kUnsupported) {
      return warp_status;
    }
  }
  if (!used_warp_aggregation) {
    auto blocks = detail::BlocksForElements(total, kCudaThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      DLCUDA_RETURN_IF_ERROR(LaunchEmbeddingBackwardKernel(
          ctx, dtype_, grad_output, cached_token_ids_, &grad_table_, blocks.value(),
          last_num_tokens_, embedding_dim_, vocab_size_));
    }
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
