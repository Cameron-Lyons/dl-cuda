#include "../include/dl_cuda/examples.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

int main(int argc, char **argv) {
  dlcuda::CharLMConfig cfg;

  auto parse_int = [](const char *text, int *out) {
    if (!text || !out)
      return false;
    errno = 0;
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
      return false;
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max())
      return false;
    *out = static_cast<int>(value);
    return true;
  };

  auto parse_float = [](const char *text, float *out) {
    if (!text || !out)
      return false;
    errno = 0;
    char *end = nullptr;
    float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0')
      return false;
    if (!std::isfinite(value))
      return false;
    *out = value;
    return true;
  };

  auto parse_u64 = [](const char *text, uint64_t *out) {
    if (!text || !out)
      return false;
    errno = 0;
    char *end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
      return false;
    *out = static_cast<uint64_t>(value);
    return true;
  };

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

  auto invalid_value = [&](const char *option, const char *value) {
    std::fprintf(stderr, "Invalid value for %s: %s\n", option,
                 value ? value : "<missing>");
    print_usage();
    return 1;
  };

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--epochs") {
      if (i + 1 >= argc)
        return invalid_value("--epochs", nullptr);
      if (!parse_int(argv[++i], &cfg.epochs) || cfg.epochs < 0)
        return invalid_value("--epochs", argv[i]);
    } else if (arg == "--seq-len") {
      if (i + 1 >= argc)
        return invalid_value("--seq-len", nullptr);
      if (!parse_int(argv[++i], &cfg.seq_len) || cfg.seq_len <= 0)
        return invalid_value("--seq-len", argv[i]);
    } else if (arg == "--gen-len") {
      if (i + 1 >= argc)
        return invalid_value("--gen-len", nullptr);
      if (!parse_int(argv[++i], &cfg.gen_len) || cfg.gen_len < 0)
        return invalid_value("--gen-len", argv[i]);
    } else if (arg == "--print-every") {
      if (i + 1 >= argc)
        return invalid_value("--print-every", nullptr);
      if (!parse_int(argv[++i], &cfg.print_every) || cfg.print_every <= 0)
        return invalid_value("--print-every", argv[i]);
    } else if (arg == "--lr-max") {
      if (i + 1 >= argc)
        return invalid_value("--lr-max", nullptr);
      if (!parse_float(argv[++i], &cfg.lr_max) || cfg.lr_max <= 0.0f)
        return invalid_value("--lr-max", argv[i]);
    } else if (arg == "--lr-min") {
      if (i + 1 >= argc)
        return invalid_value("--lr-min", nullptr);
      if (!parse_float(argv[++i], &cfg.lr_min) || cfg.lr_min < 0.0f)
        return invalid_value("--lr-min", argv[i]);
    } else if (arg == "--seed") {
      if (i + 1 >= argc)
        return invalid_value("--seed", nullptr);
      if (!parse_u64(argv[++i], &cfg.init_seed))
        return invalid_value("--seed", argv[i]);
    } else if (arg == "--sample-seed") {
      if (i + 1 >= argc)
        return invalid_value("--sample-seed", nullptr);
      if (!parse_u64(argv[++i], &cfg.sample_seed))
        return invalid_value("--sample-seed", argv[i]);
    } else if (arg == "--weights") {
      if (i + 1 >= argc)
        return invalid_value("--weights", nullptr);
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
