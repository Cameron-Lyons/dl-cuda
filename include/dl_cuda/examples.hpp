#pragma once

#include <cstdint>
#include <string>

namespace dlcuda {

struct CharLMConfig {
  int seq_len = 64;
  int batch_size = 1;
  int d_model = 64;
  int d_ff = 256;
  int num_heads = 4;
  int num_layers = 3;
  int epochs = 800;
  int print_every = 50;
  int gen_len = 200;
  float grad_clip = 1.0f;
  float temperature = 0.8f;
  float top_p = 0.9f;
  float lr_max = 3e-3f;
  float lr_min = 1e-5f;
  int warmup_steps = 50;
  uint64_t init_seed = 12345ULL;
  uint64_t sample_seed = 123ULL;
  std::string weights_path = "model.bin";
  bool load_weights = false;
  bool save_weights = true;
};

struct XorConfig {
  int epochs = 3000;
  int print_every = 300;
  float lr = 0.1f;
  uint64_t init_seed = 777ULL;
};

int run_char_lm(const CharLMConfig &cfg);
int run_xor(const XorConfig &cfg);

} // namespace dlcuda
