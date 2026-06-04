#pragma once

#include "dl_cuda/status.hpp"

#include <cstdint>
#include <string>

namespace dlcuda {

struct TrainXorConfig {
  int epochs = 3000;
  int print_every = 300;
  int hidden_size = 8;
  float lr = 0.1f;
  float grad_clip = 1.0f;
  bool use_cublas = true;
  bool tf32 = true;
  uint64_t seed = 777ULL;
  std::string checkpoint_path = "xor.ckpt";
  bool resume = false;
  bool save = true;
};

struct TrainCharConfig {
  int seq_len = 64;
  int d_model = 64;
  int epochs = 800;
  int print_every = 50;
  float lr = 3e-3f;
  float grad_clip = 1.0f;
  float temperature = 0.8f;
  float top_p = 0.9f;
  int gen_len = 200;
  bool use_cublas = true;
  bool tf32 = true;
  uint64_t seed = 12345ULL;
  uint64_t sample_seed = 123ULL;
  std::string checkpoint_path = "char.ckpt";
  bool resume = false;
  bool save = true;
};

struct SampleCharConfig {
  int seq_len = 64;
  int d_model = 64;
  int gen_len = 200;
  float temperature = 0.8f;
  float top_p = 0.9f;
  bool use_cublas = true;
  bool tf32 = true;
  uint64_t seed = 12345ULL;
  uint64_t sample_seed = 123ULL;
  std::string checkpoint_path = "char.ckpt";
};

Status TrainXor(const TrainXorConfig &cfg);
Status TrainChar(const TrainCharConfig &cfg);
Result<std::string> SampleChar(const SampleCharConfig &cfg);

} // namespace dlcuda
