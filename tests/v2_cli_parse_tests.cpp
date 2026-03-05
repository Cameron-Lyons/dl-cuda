#include "cli_parse_utils.hpp"

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Config {
  int epochs = 10;
  float lr = 0.1f;
  uint64_t seed = 1;
  bool save = true;
  std::string checkpoint = "default.ckpt";
};

const std::vector<cli_parse::OptionSpec<Config>> &Options() {
  static const std::vector<cli_parse::OptionSpec<Config>> options = {
      cli_parse::int_option("--epochs", &Config::epochs),
      cli_parse::float_option("--lr", &Config::lr),
      cli_parse::u64_option("--seed", &Config::seed),
      cli_parse::string_option("--checkpoint", &Config::checkpoint),
      cli_parse::bool_flag_option("--no-save", &Config::save, false, "save"),
  };
  return options;
}

} // namespace

int main() {
  Config cfg;
  std::unordered_map<std::string, std::string> cfg_map = {
      {"epochs", "42"},
      {"lr", "0.25"},
      {"seed", "99"},
      {"checkpoint", "from-config.ckpt"},
      {"save", "false"},
  };
  std::string error;
  if (!cli_parse::apply_config_map(cfg_map, Options(), &cfg, &error)) {
    std::fprintf(stderr, "Config parsing failed: %s\n", error.c_str());
    return 1;
  }
  if (cfg.epochs != 42 || cfg.lr != 0.25f || cfg.seed != 99 ||
      cfg.checkpoint != "from-config.ckpt" || cfg.save) {
    std::fprintf(stderr, "Config map values were not applied correctly\n");
    return 1;
  }

  char arg0[] = "dl-cuda";
  char arg1[] = "train";
  char arg2[] = "--epochs";
  char arg3[] = "100";
  char arg4[] = "--checkpoint";
  char arg5[] = "from-cli.ckpt";
  char arg6[] = "--no-save";
  char *argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};

  cli_parse::ParseResult parse_result =
      cli_parse::apply_command_line(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                    argv, 2, Options(), &cfg, &error);
  if (parse_result != cli_parse::ParseResult::kOk) {
    std::fprintf(stderr, "CLI parsing failed: %s\n", error.c_str());
    return 1;
  }
  if (cfg.epochs != 100 || cfg.checkpoint != "from-cli.ckpt" || cfg.save) {
    std::fprintf(stderr, "CLI values were not applied correctly\n");
    return 1;
  }

  std::unordered_map<std::string, std::string> bad_cfg = {{"unknown", "1"}};
  if (cli_parse::apply_config_map(bad_cfg, Options(), &cfg, &error)) {
    std::fprintf(stderr, "Unknown config key should have failed\n");
    return 1;
  }

  std::printf("v2_cli_tests: PASS\n");
  return 0;
}
