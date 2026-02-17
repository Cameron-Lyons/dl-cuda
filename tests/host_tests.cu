#include "sampling.cuh"
#include "scheduler.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
bool nearly_equal(float a, float b, float eps = 1e-6f) {
  return std::fabs(a - b) <= eps;
}
} // namespace

int main() {
  {
    CosineAnnealingScheduler sched(1.0f, 0.1f, 100);
    if (!nearly_equal(sched.get_lr(0), 1.0f)) {
      std::fprintf(stderr, "Cosine scheduler start LR mismatch\n");
      return 1;
    }
    if (!nearly_equal(sched.get_lr(100), 0.1f)) {
      std::fprintf(stderr, "Cosine scheduler end LR mismatch\n");
      return 1;
    }
  }

  {
    StepDecayScheduler sched(0.1f, 0.5f, 10);
    if (!nearly_equal(sched.get_lr(0), 0.1f) ||
        !nearly_equal(sched.get_lr(10), 0.05f) ||
        !nearly_equal(sched.get_lr(20), 0.025f)) {
      std::fprintf(stderr, "Step scheduler mismatch\n");
      return 1;
    }
  }

  {
    CosineAnnealingScheduler after(1e-3f, 1e-4f, 20);
    WarmupScheduler sched(1e-3f, 5, &after);
    if (!nearly_equal(sched.get_lr(0), 2e-4f)) {
      std::fprintf(stderr, "Warmup scheduler mismatch at step 0\n");
      return 1;
    }
    if (!nearly_equal(sched.get_lr(4), 1e-3f)) {
      std::fprintf(stderr, "Warmup scheduler mismatch at end warmup\n");
      return 1;
    }
  }

  {
    std::vector<float> probs = {0.9f, 0.06f, 0.03f, 0.01f};
    applyTopK(probs, 1);
    if (!(probs[0] > 0.0f && probs[1] == 0.0f && probs[2] == 0.0f &&
          probs[3] == 0.0f)) {
      std::fprintf(stderr, "Top-k filtering mismatch\n");
      return 1;
    }
  }

  {
    std::vector<float> probs = {0.8f, 0.1f, 0.07f, 0.03f};
    applyTopP(probs, 0.85f);
    if (!(probs[0] > 0.0f && probs[1] > 0.0f && probs[2] == 0.0f &&
          probs[3] == 0.0f)) {
      std::fprintf(stderr, "Top-p filtering mismatch\n");
      return 1;
    }
  }

  {
    std::mt19937 rng(123);
    std::vector<float> probs = {1.0f, 0.0f, 0.0f};
    int sampled = sampleWithStrategy(probs, 1.0f, 0, 1.0f, rng);
    if (sampled != 0) {
      std::fprintf(stderr, "Sampling mismatch\n");
      return 1;
    }
  }

  std::printf("host_tests: PASS\n");
  return 0;
}
