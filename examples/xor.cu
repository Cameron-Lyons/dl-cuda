#include "../include/dl_cuda/examples.hpp"
#include "cli_parse_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
  dlcuda::XorConfig cfg;

  auto print_usage = []() {
    std::puts("Usage: dl-cuda-xor [options]");
    std::puts("  --epochs N");
    std::puts("  --print-every N");
    std::puts("  --lr F");
    std::puts("  --seed N");
    std::puts("  --help");
  };

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--epochs") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--epochs", nullptr, print_usage);
      if (!cli_parse::parse_int(argv[++i], &cfg.epochs) || cfg.epochs < 0)
        return cli_parse::invalid_value("--epochs", argv[i], print_usage);
    } else if (arg == "--print-every") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--print-every", nullptr, print_usage);
      if (!cli_parse::parse_int(argv[++i], &cfg.print_every) || cfg.print_every <= 0)
        return cli_parse::invalid_value("--print-every", argv[i], print_usage);
    } else if (arg == "--lr") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--lr", nullptr, print_usage);
      if (!cli_parse::parse_float(argv[++i], &cfg.lr) || cfg.lr <= 0.0f)
        return cli_parse::invalid_value("--lr", argv[i], print_usage);
    } else if (arg == "--seed") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--seed", nullptr, print_usage);
      if (!cli_parse::parse_u64(argv[++i], &cfg.init_seed))
        return cli_parse::invalid_value("--seed", argv[i], print_usage);
    } else if (arg == "--help") {
      print_usage();
      return 0;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      print_usage();
      return 1;
    }
  }

  return dlcuda::run_xor(cfg);
}
