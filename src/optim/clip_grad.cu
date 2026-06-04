#include "detail/clip_kernels.cuh"

namespace dlcuda {

Status ClipGradNorm(RuntimeContext &ctx, const std::vector<ParameterRef> &params, float max_norm,
                    float *total_norm) {
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(max_norm, "max_norm"));

  bool has_grad_elements = false;
  for (const auto &param : params) {
    DLCUDA_RETURN_IF_ERROR(ValidateGradient(param, "ClipGradNorm"));
    if (param.grad->numel() > 0) {
      has_grad_elements = true;
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
  DLCUDA_RETURN_IF_ERROR(total_norm_sq_buffer.FillZero(ctx.stream()));

  auto clip_scale_tensor =
      ctx.ScratchTensor("optim.clip_grad_norm.clip_scale", {1}, DType::kFloat32);
  if (!clip_scale_tensor.ok()) {
    return clip_scale_tensor.status();
  }
  Tensor clip_scale_buffer = clip_scale_tensor.value();

  for (const auto &param : params) {
    int64_t n = param.grad->numel();
    auto blocks = detail::CappedBlocksForElements(n, kOptimizerThreads, kNormReductionMaxBlocks);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() <= 0) {
      continue;
    }
    DLCUDA_RETURN_IF_ERROR(
        LaunchAccumulateNormSq(ctx, *param.grad, &total_norm_sq_buffer, blocks.value()));
  }

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
