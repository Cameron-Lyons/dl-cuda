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
  int val_every = 50;
  int val_windows = 16;
  int test_windows = 32;
  int early_stop_patience = 0;
  float lr = 3e-3f;
  float grad_clip = 1.0f;
  float val_fraction = 0.1f;
  float test_fraction = 0.1f;
  float min_delta = 0.0f;
  float temperature = 0.8f;
  float top_p = 0.9f;
  int gen_len = 200;
  bool use_cublas = true;
  bool tf32 = true;
  uint64_t seed = 12345ULL;
  uint64_t sample_seed = 123ULL;
  std::string checkpoint_path = "char.ckpt";
  std::string best_checkpoint_path;
  std::string data_path;
  std::string prompt;
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
  std::string data_path;
  std::string prompt;
};

Status TrainXor(const TrainXorConfig &cfg);
Status TrainChar(const TrainCharConfig &cfg);
Result<std::string> SampleChar(const SampleCharConfig &cfg);

} // namespace dlcuda
