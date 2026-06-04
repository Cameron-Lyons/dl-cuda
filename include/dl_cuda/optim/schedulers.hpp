#pragma once

#include "dl_cuda/status.hpp"

#include <cstdint>

namespace dlcuda {

class LearningRateScheduler {
public:
  virtual ~LearningRateScheduler() = default;

  virtual Result<float> LearningRate(int64_t step_index, float base_lr) const = 0;
};

class ConstantLRScheduler : public LearningRateScheduler {
public:
  Result<float> LearningRate(int64_t step_index, float base_lr) const override;
};

class StepLRScheduler : public LearningRateScheduler {
public:
  StepLRScheduler(int64_t step_size, float gamma) : step_size_(step_size), gamma_(gamma) {}

  Result<float> LearningRate(int64_t step_index, float base_lr) const override;

private:
  int64_t step_size_ = 1;
  float gamma_ = 1.0f;
};

class ExponentialLRScheduler : public LearningRateScheduler {
public:
  explicit ExponentialLRScheduler(float gamma) : gamma_(gamma) {}

  Result<float> LearningRate(int64_t step_index, float base_lr) const override;

private:
  float gamma_ = 1.0f;
};

class CosineAnnealingLRScheduler : public LearningRateScheduler {
public:
  CosineAnnealingLRScheduler(int64_t max_steps, float min_lr = 0.0f)
      : max_steps_(max_steps), min_lr_(min_lr) {}

  Result<float> LearningRate(int64_t step_index, float base_lr) const override;

private:
  int64_t max_steps_ = 1;
  float min_lr_ = 0.0f;
};

} // namespace dlcuda
