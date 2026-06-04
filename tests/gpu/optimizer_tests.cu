#include "common.hpp"

namespace dlcuda::gpu_tests {

bool RunOptimizerCoverageSmoke(dlcuda::RuntimeContext &ctx) {
  auto make_param = [&](const char *label, const std::vector<float> &values,
                        const std::vector<float> &grads, dlcuda::Tensor *param,
                        dlcuda::Tensor *grad) -> bool {
    auto param_result = dlcuda::Tensor::AllocateAsync({static_cast<int64_t>(values.size())},
                                                      dlcuda::DType::kFloat32, ctx.stream());
    auto grad_result = dlcuda::Tensor::AllocateAsync({static_cast<int64_t>(grads.size())},
                                                     dlcuda::DType::kFloat32, ctx.stream());
    if (!param_result.ok() || !grad_result.ok()) {
      std::fprintf(stderr, "%s optimizer tensor allocation failed\n", label);
      return false;
    }
    *param = param_result.value();
    *grad = grad_result.value();
    return CopyFloatsToTensor(ctx, param, values, label) &&
           CopyFloatsToTensor(ctx, grad, grads, label);
  };

  std::vector<float> host_values;

  dlcuda::Tensor sgd_param;
  dlcuda::Tensor sgd_grad;
  if (!make_param("sgd", {1.0f}, {0.5f}, &sgd_param, &sgd_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> sgd_params = {{"sgd.weight", &sgd_param, &sgd_grad}};
  dlcuda::SGDOptimizer sgd(0.1f, 0.9f);
  if (!sgd.Step(ctx, sgd_params).ok() || !sgd.Step(ctx, sgd_params).ok() ||
      !CopyTensorToFloats(ctx, sgd_param, &host_values, "sgd param") ||
      !CheckCloseVector(host_values, {0.855f}, "SGD momentum")) {
    return false;
  }

  dlcuda::Tensor wd_param;
  dlcuda::Tensor wd_grad;
  if (!make_param("sgd weight decay", {1.0f, -2.0f}, {0.5f, -0.25f}, &wd_param, &wd_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> wd_params = {{"wd.weight", &wd_param, &wd_grad}};
  dlcuda::SGDOptimizer sgd_wd(0.1f, 0.0f, 0.1f);
  if (!sgd_wd.Step(ctx, wd_params).ok() ||
      !CopyTensorToFloats(ctx, wd_param, &host_values, "sgd weight decay param") ||
      !CheckCloseVector(host_values, {0.94f, -1.955f}, "SGD weight decay")) {
    return false;
  }

  dlcuda::Tensor adamw_param;
  dlcuda::Tensor adamw_grad;
  if (!make_param("adamw", {1.0f}, {0.5f}, &adamw_param, &adamw_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> adamw_params = {{"adamw.weight", &adamw_param, &adamw_grad}};
  dlcuda::AdamWOptimizer adamw(0.1f, 0.01f);
  if (!adamw.Step(ctx, adamw_params).ok() ||
      !CopyTensorToFloats(ctx, adamw_param, &host_values, "adamw param") ||
      !CheckCloseVector(host_values, {0.899f}, "AdamW decoupled weight decay", 2e-4f)) {
    return false;
  }

  dlcuda::Tensor rms_param;
  dlcuda::Tensor rms_grad;
  if (!make_param("rmsprop", {1.0f}, {0.5f}, &rms_param, &rms_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> rms_params = {{"rms.weight", &rms_param, &rms_grad}};
  dlcuda::RMSPropOptimizer rmsprop(0.1f, 0.0f);
  if (!rmsprop.Step(ctx, rms_params).ok() ||
      !CopyTensorToFloats(ctx, rms_param, &host_values, "rmsprop param") ||
      !CheckCloseVector(host_values, {0.9f}, "RMSProp update", 2e-4f)) {
    return false;
  }

  dlcuda::Tensor group_a;
  dlcuda::Tensor group_a_grad;
  dlcuda::Tensor group_b;
  dlcuda::Tensor group_b_grad;
  if (!make_param("group a", {1.0f}, {1.0f}, &group_a, &group_a_grad) ||
      !make_param("group b", {1.0f}, {1.0f}, &group_b, &group_b_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> group_params = {{"group.a", &group_a, &group_a_grad},
                                                    {"group.b", &group_b, &group_b_grad}};
  dlcuda::SGDOptimizer grouped(std::vector<dlcuda::OptimizerParamGroup>{
      {{"group.a"}, 0.1f, 0.0f}, {{"group.b"}, 0.01f, 0.0f}});
  dlcuda::StepLRScheduler step_lr(1, 0.5f);
  if (!grouped.Step(ctx, group_params, step_lr).ok() ||
      !grouped.Step(ctx, group_params, step_lr).ok() ||
      !CopyTensorToFloats(ctx, group_a, &host_values, "group a param") ||
      !CheckCloseVector(host_values, {0.85f}, "parameter group a scheduler") ||
      !CopyTensorToFloats(ctx, group_b, &host_values, "group b param") ||
      !CheckCloseVector(host_values, {0.985f}, "parameter group b scheduler")) {
    return false;
  }
  dlcuda::ConstantLRScheduler constant_lr;
  dlcuda::ExponentialLRScheduler exponential_lr(0.5f);
  dlcuda::CosineAnnealingLRScheduler cosine_lr(4, 0.1f);
  auto constant_rate = constant_lr.LearningRate(7, 0.25f);
  auto exponential_rate = exponential_lr.LearningRate(3, 0.8f);
  auto cosine_start = cosine_lr.LearningRate(0, 1.0f);
  auto cosine_mid = cosine_lr.LearningRate(2, 1.0f);
  auto cosine_end = cosine_lr.LearningRate(8, 1.0f);
  if (!constant_rate.ok() || !exponential_rate.ok() || !cosine_start.ok() || !cosine_mid.ok() ||
      !cosine_end.ok() || !AlmostEqual(constant_rate.value(), 0.25f) ||
      !AlmostEqual(exponential_rate.value(), 0.1f) || !AlmostEqual(cosine_start.value(), 1.0f) ||
      !AlmostEqual(cosine_mid.value(), 0.55f) || !AlmostEqual(cosine_end.value(), 0.1f)) {
    return false;
  }

  dlcuda::Tensor checkpoint_param;
  dlcuda::Tensor checkpoint_grad;
  dlcuda::Tensor restored_param;
  dlcuda::Tensor restored_grad;
  if (!make_param("checkpoint", {1.0f}, {0.25f}, &checkpoint_param, &checkpoint_grad) ||
      !make_param("checkpoint restore", {1.0f}, {0.25f}, &restored_param, &restored_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> checkpoint_params = {
      {"checkpoint.weight", &checkpoint_param, &checkpoint_grad}};
  std::vector<dlcuda::ParameterRef> restored_params = {
      {"checkpoint.weight", &restored_param, &restored_grad}};
  dlcuda::AdamOptimizer checkpoint_adam;
  const char *checkpoint_path = "/tmp/dlcuda_optimizer_smoke.ckpt";
  if (!checkpoint_adam.Step(ctx, checkpoint_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, checkpoint_param, &host_values, "checkpoint after first step") ||
      !CopyFloatsToTensor(ctx, &restored_param, host_values, "checkpoint restore param") ||
      !checkpoint_adam.SaveCheckpoint(ctx, checkpoint_path, checkpoint_params).ok() ||
      !checkpoint_adam.Step(ctx, checkpoint_params, 0.1f).ok()) {
    std::fprintf(stderr, "Optimizer checkpoint save path failed\n");
    std::remove(checkpoint_path);
    return false;
  }
  dlcuda::AdamOptimizer restored_adam;
  dlcuda::AdamOptimizer mismatched_adam(0.8f, 0.999f, 1e-8f);
  if (mismatched_adam.LoadCheckpoint(ctx, checkpoint_path, restored_params).ok()) {
    std::fprintf(stderr, "Optimizer checkpoint should reject mismatched hyperparameters\n");
    std::remove(checkpoint_path);
    return false;
  }
  if (!restored_adam.LoadCheckpoint(ctx, checkpoint_path, restored_params).ok() ||
      restored_adam.step_count() != 1 || !restored_adam.Step(ctx, restored_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, checkpoint_param, &host_values, "checkpoint continuous param")) {
    std::fprintf(stderr, "Optimizer checkpoint restore path failed\n");
    std::remove(checkpoint_path);
    return false;
  }
  std::vector<float> restored_values;
  if (!CopyTensorToFloats(ctx, restored_param, &restored_values, "checkpoint restored param") ||
      !CheckCloseVector(restored_values, host_values, "Optimizer checkpoint restore", 2e-4f)) {
    std::remove(checkpoint_path);
    return false;
  }
  std::remove(checkpoint_path);

  dlcuda::Tensor full_param;
  dlcuda::Tensor full_grad;
  dlcuda::Tensor full_restored_param;
  dlcuda::Tensor full_restored_grad;
  if (!make_param("full checkpoint", {1.0f}, {0.25f}, &full_param, &full_grad) ||
      !make_param("full checkpoint restore", {1.0f}, {0.25f}, &full_restored_param,
                  &full_restored_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> full_params = {{"full.weight", &full_param, &full_grad}};
  std::vector<dlcuda::ParameterRef> full_restored_params = {
      {"full.weight", &full_restored_param, &full_restored_grad}};
  dlcuda::AdamOptimizer full_adam;
  const char *full_checkpoint_path = "/tmp/dlcuda_full_smoke.ckpt";
  dlcuda::CheckpointMetadata full_metadata;
  full_metadata.model_name = "full-smoke";
  full_metadata.epoch = 7;
  full_metadata.training_config = {{"lr", "0.1"}};
  full_metadata.rng_states = {{"offset_rng", "1234"}};
  if (!full_adam.Step(ctx, full_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, full_param, &host_values, "full checkpoint after first step") ||
      !CopyFloatsToTensor(ctx, &full_restored_param, host_values,
                          "full checkpoint restore param")) {
    std::fprintf(stderr, "Full checkpoint setup failed\n");
    std::remove(full_checkpoint_path);
    return false;
  }
  full_metadata.step = full_adam.step_count();
  if (!dlcuda::SaveCheckpoint(ctx, full_checkpoint_path, full_metadata, full_params, &full_adam)
           .ok() ||
      !full_adam.Step(ctx, full_params, 0.1f).ok()) {
    std::fprintf(stderr, "Full checkpoint save path failed\n");
    std::remove(full_checkpoint_path);
    return false;
  }
  dlcuda::AdamOptimizer full_restored_adam;
  dlcuda::CheckpointMetadata loaded_full_metadata;
  if (!dlcuda::LoadCheckpoint(ctx, full_checkpoint_path, "full-smoke", full_restored_params,
                              &full_restored_adam, &loaded_full_metadata)
           .ok() ||
      loaded_full_metadata.epoch != 7 || loaded_full_metadata.step != 1 ||
      loaded_full_metadata.training_config.size() != 1 ||
      loaded_full_metadata.training_config[0].key != "lr" ||
      loaded_full_metadata.training_config[0].value != "0.1" ||
      loaded_full_metadata.rng_states.size() != 1 || full_restored_adam.step_count() != 1 ||
      !full_restored_adam.Step(ctx, full_restored_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, full_param, &host_values, "full checkpoint continuous param")) {
    std::fprintf(stderr, "Full checkpoint restore path failed\n");
    std::remove(full_checkpoint_path);
    return false;
  }
  if (!CopyTensorToFloats(ctx, full_restored_param, &restored_values,
                          "full checkpoint restored param") ||
      !CheckCloseVector(restored_values, host_values, "Full checkpoint restore", 2e-4f)) {
    std::remove(full_checkpoint_path);
    return false;
  }
  std::remove(full_checkpoint_path);

  return true;
}

} // namespace dlcuda::gpu_tests
