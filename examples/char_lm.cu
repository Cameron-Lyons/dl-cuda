#include "../include/dl_cuda/examples.hpp"
#include "cli_parse_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
  dlcuda::CharLMConfig cfg;

  auto print_usage = []() {
    std::puts("Usage: dl-cuda-char-lm [options]");
    std::puts("  --epochs N");
    std::puts("  --seq-len N");
    std::puts("  --gen-len N");
    std::puts("  --print-every N");
    std::puts("  --lr-max F");
    std::puts("  --lr-min F");
    std::puts("  --seed N");
    std::puts("  --sample-seed N");
    std::puts("  --weights PATH");
    std::puts("  --load-weights");
    std::puts("  --no-save");
    std::puts("  --help");
  };

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--epochs") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--epochs", nullptr, print_usage);
      if (!cli_parse::parse_int(argv[++i], &cfg.epochs) || cfg.epochs < 0)
        return cli_parse::invalid_value("--epochs", argv[i], print_usage);
    } else if (arg == "--seq-len") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--seq-len", nullptr, print_usage);
      if (!cli_parse::parse_int(argv[++i], &cfg.seq_len) || cfg.seq_len <= 0)
        return cli_parse::invalid_value("--seq-len", argv[i], print_usage);
    } else if (arg == "--gen-len") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--gen-len", nullptr, print_usage);
      if (!cli_parse::parse_int(argv[++i], &cfg.gen_len) || cfg.gen_len < 0)
        return cli_parse::invalid_value("--gen-len", argv[i], print_usage);
    } else if (arg == "--print-every") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--print-every", nullptr, print_usage);
      if (!cli_parse::parse_int(argv[++i], &cfg.print_every) || cfg.print_every <= 0)
        return cli_parse::invalid_value("--print-every", argv[i], print_usage);
    } else if (arg == "--lr-max") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--lr-max", nullptr, print_usage);
      if (!cli_parse::parse_float(argv[++i], &cfg.lr_max) || cfg.lr_max <= 0.0f)
        return cli_parse::invalid_value("--lr-max", argv[i], print_usage);
    } else if (arg == "--lr-min") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--lr-min", nullptr, print_usage);
      if (!cli_parse::parse_float(argv[++i], &cfg.lr_min) || cfg.lr_min < 0.0f)
        return cli_parse::invalid_value("--lr-min", argv[i], print_usage);
    } else if (arg == "--seed") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--seed", nullptr, print_usage);
      if (!cli_parse::parse_u64(argv[++i], &cfg.init_seed))
        return cli_parse::invalid_value("--seed", argv[i], print_usage);
    } else if (arg == "--sample-seed") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--sample-seed", nullptr, print_usage);
      if (!cli_parse::parse_u64(argv[++i], &cfg.sample_seed))
        return cli_parse::invalid_value("--sample-seed", argv[i], print_usage);
    } else if (arg == "--weights") {
      if (i + 1 >= argc)
        return cli_parse::invalid_value("--weights", nullptr, print_usage);
      cfg.weights_path = argv[++i];
    } else if (arg == "--load-weights") {
      cfg.load_weights = true;
    } else if (arg == "--no-save") {
      cfg.save_weights = false;
    } else if (arg == "--help") {
      print_usage();
      return 0;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      print_usage();
      return 1;
    }
  }

  if (cfg.lr_min > cfg.lr_max) {
    std::fprintf(stderr, "--lr-min must be <= --lr-max\n");
    return 1;
  }

  return dlcuda::run_char_lm(cfg);
}
