#include "../include/dl_cuda/examples.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

int main(int argc, char **argv) {
  dlcuda::XorConfig cfg;

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
    std::puts("Usage: dl-cuda-xor [options]");
    std::puts("  --epochs N");
    std::puts("  --print-every N");
    std::puts("  --lr F");
    std::puts("  --seed N");
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
    } else if (arg == "--print-every") {
      if (i + 1 >= argc)
        return invalid_value("--print-every", nullptr);
      if (!parse_int(argv[++i], &cfg.print_every) || cfg.print_every <= 0)
        return invalid_value("--print-every", argv[i]);
    } else if (arg == "--lr") {
      if (i + 1 >= argc)
        return invalid_value("--lr", nullptr);
      if (!parse_float(argv[++i], &cfg.lr) || cfg.lr <= 0.0f)
        return invalid_value("--lr", argv[i]);
    } else if (arg == "--seed") {
      if (i + 1 >= argc)
        return invalid_value("--seed", nullptr);
      if (!parse_u64(argv[++i], &cfg.init_seed))
        return invalid_value("--seed", argv[i]);
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
