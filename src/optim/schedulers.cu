#include "detail/common.cuh"

namespace dlcuda {

Result<float> ConstantLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  return base_lr;
}

Result<float> StepLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  if (step_size_ <= 0) {
    return Status::InvalidArgument("StepLRScheduler step_size must be > 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(gamma_, "StepLRScheduler gamma"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  int64_t intervals = step_index / step_size_;
  return base_lr * std::pow(gamma_, static_cast<float>(intervals));
}

Result<float> ExponentialLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(gamma_, "ExponentialLRScheduler gamma"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  return base_lr * std::pow(gamma_, static_cast<float>(step_index));
}

Result<float> CosineAnnealingLRScheduler::LearningRate(int64_t step_index, float base_lr) const {
  if (step_index < 0) {
    return Status::InvalidArgument("Scheduler step_index must be non-negative");
  }
  if (max_steps_ <= 0) {
    return Status::InvalidArgument("CosineAnnealingLRScheduler max_steps must be > 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(base_lr, "Scheduler base_lr"));
  DLCUDA_RETURN_IF_ERROR(ValidateNonNegativeFinite(min_lr_, "CosineAnnealingLRScheduler min_lr"));
  int64_t clamped_step = std::min(step_index, max_steps_);
  double cosine =
      std::cos(kPi * static_cast<double>(clamped_step) / static_cast<double>(max_steps_));
  return static_cast<float>(min_lr_ +
                            0.5 * (static_cast<double>(base_lr) - min_lr_) * (1.0 + cosine));
}

} // namespace dlcuda
