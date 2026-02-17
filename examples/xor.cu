#include "../include/dl_cuda/examples.hpp"
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
  dlcuda::XorConfig cfg;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--epochs" && i + 1 < argc)
      cfg.epochs = std::atoi(argv[++i]);
    else if (arg == "--print-every" && i + 1 < argc)
      cfg.print_every = std::atoi(argv[++i]);
    else if (arg == "--lr" && i + 1 < argc)
      cfg.lr = std::atof(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      cfg.init_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
  }

  return dlcuda::run_xor(cfg);
}
