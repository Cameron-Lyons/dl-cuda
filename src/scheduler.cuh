#pragma once

#include <cmath>

class LRScheduler {
public:
  virtual float get_lr(int step) = 0;
  virtual ~LRScheduler() = default;
};

class CosineAnnealingScheduler : public LRScheduler {
private:
  float lr_max_, lr_min_;
  int t_max_;

public:
  CosineAnnealingScheduler(float lr_max, float lr_min, int t_max)
      : lr_max_(lr_max), lr_min_(lr_min), t_max_(t_max) {}

  float get_lr(int step) override {
    if (step >= t_max_)
      return lr_min_;
    float cosine = cosf(static_cast<float>(M_PI) * step / t_max_);
    return lr_min_ + 0.5f * (lr_max_ - lr_min_) * (1.0f + cosine);
  }
};

class StepDecayScheduler : public LRScheduler {
private:
  float initial_lr_, gamma_;
  int step_size_;

public:
  StepDecayScheduler(float initial_lr, float gamma, int step_size)
      : initial_lr_(initial_lr), gamma_(gamma), step_size_(step_size) {}

  float get_lr(int step) override {
    int num_decays = step / step_size_;
    return initial_lr_ * powf(gamma_, static_cast<float>(num_decays));
  }
};

class WarmupScheduler : public LRScheduler {
private:
  float target_lr_;
  int warmup_steps_;
  LRScheduler *after_warmup_;

public:
  WarmupScheduler(float target_lr, int warmup_steps,
                  LRScheduler *after_warmup = nullptr)
      : target_lr_(target_lr), warmup_steps_(warmup_steps),
        after_warmup_(after_warmup) {}

  float get_lr(int step) override {
    if (step < warmup_steps_) {
      return target_lr_ * (static_cast<float>(step + 1) / warmup_steps_);
    }
    if (after_warmup_) {
      return after_warmup_->get_lr(step - warmup_steps_);
    }
    return target_lr_;
  }
};
