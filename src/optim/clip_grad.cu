#include "detail/clip_kernels.cuh"

namespace dlcuda {
namespace {

struct NormReductionLaunch {
  const Tensor *grad = nullptr;
  int blocks = 0;
  int64_t partial_offset = 0;
};

} // namespace

Status ClipGradNorm(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float max_norm,
                    float *total_norm) {
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(max_norm, "max_norm"));

  bool has_grad_elements = false;
  std::vector<NormReductionLaunch> norm_reductions;
  int64_t partial_count = 0;
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, "ClipGradNorm"));
    int64_t n = param.grad->numel();
    if (n > 0) {
      has_grad_elements = true;
      auto blocks = detail::CappedBlocksForElements(n, kOptimizerThreads, kNormReductionMaxBlocks);
      if (!blocks.ok()) {
        return blocks.status();
      }
      if (blocks.value() > 0) {
        norm_reductions.push_back({param.grad, blocks.value(), partial_count});
        partial_count += blocks.value();
      }
    }
  }
  if (!has_grad_elements) {
    if (total_norm != nullptr) {
      *total_norm = 0.0f;
    }
    return Status::Ok();
  }

  auto total_norm_sq_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.total_norm_sq", {1}, DType::kFloat32);
  if (!total_norm_sq_tensor.ok()) {
    return total_norm_sq_tensor.status();
  }
  Tensor total_norm_sq_buffer = total_norm_sq_tensor.value();

  auto clip_scale_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.clip_scale", {1}, DType::kFloat32);
  if (!clip_scale_tensor.ok()) {
    return clip_scale_tensor.status();
  }
  Tensor clip_scale_buffer = clip_scale_tensor.value();

  auto partial_norm_sq_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.partial_norm_sq", {partial_count}, DType::kFloat32);
  if (!partial_norm_sq_tensor.ok()) {
    return partial_norm_sq_tensor.status();
  }
  Tensor partial_norm_sq_buffer = partial_norm_sq_tensor.value();

  for (const NormReductionLaunch &reduction : norm_reductions) {
    DLCUDA_RETURN_IF_ERROR(LaunchNormSqPartials(ctx, *reduction.grad, &partial_norm_sq_buffer,
                                                reduction.partial_offset, reduction.blocks));
  }
  DLCUDA_RETURN_IF_ERROR(
      LaunchFinalizeNormSq(ctx, partial_norm_sq_buffer, &total_norm_sq_buffer, partial_count));

  ComputeClipScaleKernel<<<1, 1, 0, ctx.stream()>>>(total_norm_sq_buffer.data_as<float>(), max_norm,
                                                    clip_scale_buffer.data_as<float>());
  DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ComputeClipScaleKernel"));

  for (const auto &param : params) {
    auto blocks = detail::BlocksForElements(param.grad->numel(), kOptimizerThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() <= 0) {
      continue;
    }
    DLCUDA_RETURN_IF_ERROR(
        LaunchScaleByFactor(ctx, param.grad, &clip_scale_buffer, blocks.value()));
  }

  if (total_norm != nullptr) {
    float total_norm_sq = 0.0f;
    DLCUDA_RETURN_IF_ERROR(
        total_norm_sq_buffer.CopyToHost(&total_norm_sq, sizeof(total_norm_sq), ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());
    *total_norm = std::sqrt(total_norm_sq);
  }

  return Status::Ok();
}

} // namespace dlcuda
