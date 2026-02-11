#pragma once

#include "sequential.cuh"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

__global__ void temperatureScaleKernel(const float *probs, float *scaled,
                                       int vocab_size, float temperature) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vocab_size) {
    scaled[idx] = powf(probs[idx], 1.0f / temperature);
  }
}

inline int sampleFromProbs(const std::vector<float> &probs, std::mt19937 &rng) {
  float sum = 0.0f;
  for (float p : probs)
    sum += p;
  std::uniform_real_distribution<float> dist(0.0f, sum);
  float r = dist(rng);
  float cumsum = 0.0f;
  for (int i = 0; i < static_cast<int>(probs.size()); i++) {
    cumsum += probs[i];
    if (r < cumsum)
      return i;
  }
  return static_cast<int>(probs.size()) - 1;
}

inline void applyTopK(std::vector<float> &probs, int k) {
  int n = static_cast<int>(probs.size());
  if (k >= n)
    return;
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](int a, int b) { return probs[a] > probs[b]; });
  std::vector<bool> keep(n, false);
  for (int i = 0; i < k; i++)
    keep[indices[i]] = true;
  for (int i = 0; i < n; i++) {
    if (!keep[i])
      probs[i] = 0.0f;
  }
}

inline void applyTopP(std::vector<float> &probs, float p) {
  int n = static_cast<int>(probs.size());
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });
  float cumsum = 0.0f;
  int cutoff = n;
  for (int i = 0; i < n; i++) {
    cumsum += probs[indices[i]];
    if (cumsum >= p) {
      cutoff = i + 1;
      break;
    }
  }
  for (int i = cutoff; i < n; i++) {
    probs[indices[i]] = 0.0f;
  }
}

inline int sampleWithStrategy(const std::vector<float> &raw_probs,
                              float temperature, int top_k, float top_p,
                              std::mt19937 &rng) {
  std::vector<float> probs = raw_probs;
  if (temperature != 1.0f) {
    float inv_t = 1.0f / temperature;
    for (float &p : probs)
      p = powf(p, inv_t);
  }
  if (top_k > 0)
    applyTopK(probs, top_k);
  if (top_p < 1.0f)
    applyTopP(probs, top_p);
  return sampleFromProbs(probs, rng);
}
