#include "../include/dl_cuda/examples.hpp"
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
  dlcuda::CharLMConfig cfg;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--epochs" && i + 1 < argc)
      cfg.epochs = std::atoi(argv[++i]);
    else if (arg == "--seq-len" && i + 1 < argc)
      cfg.seq_len = std::atoi(argv[++i]);
    else if (arg == "--gen-len" && i + 1 < argc)
      cfg.gen_len = std::atoi(argv[++i]);
    else if (arg == "--print-every" && i + 1 < argc)
      cfg.print_every = std::atoi(argv[++i]);
    else if (arg == "--lr-max" && i + 1 < argc)
      cfg.lr_max = std::atof(argv[++i]);
    else if (arg == "--lr-min" && i + 1 < argc)
      cfg.lr_min = std::atof(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      cfg.init_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (arg == "--sample-seed" && i + 1 < argc)
      cfg.sample_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
  }

  return dlcuda::run_char_lm(cfg);
}
