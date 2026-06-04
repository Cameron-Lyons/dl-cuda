#include "cli_parse_utils.hpp"
#include "example_cli_options.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
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

const std::vector<cli_parse::OptionSpec<Config>> &MalformedOptions() {
  static const std::vector<cli_parse::OptionSpec<Config>> options = {
      cli_parse::int_option("--epochs", &Config::epochs),
      cli_parse::OptionSpec<Config>{"--broken-value", nullptr, true, {}, {}},
      cli_parse::OptionSpec<Config>{"--broken-flag", nullptr, false, {}, {}},
  };
  return options;
}

} // namespace

int main() {
  Config cfg;
  std::unordered_map<std::string, std::string> cfg_map = {
      {"epochs", "42"},  {"lr", "0.25"}, {"seed", "99"}, {"checkpoint", "from-config.ckpt"},
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

  cli_parse::ParseResult parse_result = cli_parse::apply_command_line(
      static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, 2, Options(), &cfg, &error);
  if (parse_result != cli_parse::ParseResult::kOk) {
    std::fprintf(stderr, "CLI parsing failed: %s\n", error.c_str());
    return 1;
  }
  if (cfg.epochs != 100 || cfg.checkpoint != "from-cli.ckpt" || cfg.save) {
    std::fprintf(stderr, "CLI values were not applied correctly\n");
    return 1;
  }
  Config configured_cfg;
  int configure_status =
      cli_parse::configure_command(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, 2,
                                   Options(), &configured_cfg, []() {});
  if (configure_status != -1 || configured_cfg.epochs != 100 ||
      configured_cfg.checkpoint != "from-cli.ckpt" || configured_cfg.save) {
    std::fprintf(stderr, "configure_command did not apply CLI values correctly\n");
    return 1;
  }

  char help_arg[] = "--help";
  char *help_argv[] = {arg0, arg1, arg2, help_arg};
  bool printed_usage = false;
  configure_status = cli_parse::configure_command(
      static_cast<int>(sizeof(help_argv) / sizeof(help_argv[0])), help_argv, 2, Options(),
      &configured_cfg, [&printed_usage]() { printed_usage = true; });
  if (configure_status != 0 || !printed_usage) {
    std::fprintf(stderr, "configure_command should prioritize help\n");
    return 1;
  }

  std::unordered_map<std::string, std::string> bad_cfg = {{"unknown", "1"}};
  if (cli_parse::apply_config_map(bad_cfg, Options(), &cfg, &error)) {
    std::fprintf(stderr, "Unknown config key should have failed\n");
    return 1;
  }

  uint64_t parsed_seed = 0;
  if (cli_parse::parse_u64("-1", &parsed_seed)) {
    std::fprintf(stderr, "Negative u64 parse should have failed\n");
    return 1;
  }
  if (!cli_parse::parse_u64("18446744073709551615", &parsed_seed) ||
      parsed_seed != std::numeric_limits<uint64_t>::max()) {
    std::fprintf(stderr, "Max u64 parse failed\n");
    return 1;
  }
  float parsed_float = 0.0f;
  if (cli_parse::parse_float("inf", &parsed_float) ||
      cli_parse::parse_float("nan", &parsed_float)) {
    std::fprintf(stderr, "Non-finite float parse should have failed\n");
    return 1;
  }
  if (cli_parse::parse_bool("true", nullptr)) {
    std::fprintf(stderr, "Null bool parse output should have failed\n");
    return 1;
  }
  bool parsed_bool = true;
  if (!cli_parse::parse_bool("FALSE", &parsed_bool) || parsed_bool) {
    std::fprintf(stderr, "Case-insensitive bool parse failed\n");
    return 1;
  }

  Config inline_cfg;
  char inline_arg2[] = "--epochs=125";
  char inline_arg3[] = "--lr=0.75";
  char inline_arg4[] = "--seed=1234";
  char inline_arg5[] = "--checkpoint=inline.ckpt";
  char *inline_argv[] = {arg0, arg1, inline_arg2, inline_arg3, inline_arg4, inline_arg5};
  parse_result =
      cli_parse::apply_command_line(static_cast<int>(sizeof(inline_argv) / sizeof(inline_argv[0])),
                                    inline_argv, 2, Options(), &inline_cfg, &error);
  if (parse_result != cli_parse::ParseResult::kOk || inline_cfg.epochs != 125 ||
      inline_cfg.lr != 0.75f || inline_cfg.seed != 1234 || inline_cfg.checkpoint != "inline.ckpt") {
    std::fprintf(stderr, "Inline CLI value parsing failed: %s\n", error.c_str());
    return 1;
  }

  const char config_path[] = "cli_parse_test.cfg";
  {
    std::ofstream config_file(config_path);
    config_file << "lr=0.5\n";
  }
  std::unordered_map<std::string, std::string> loaded_cfg = {{"epochs", "stale"}};
  if (!cli_parse::load_config_file(config_path, &loaded_cfg, &error)) {
    std::fprintf(stderr, "Config file load failed: %s\n", error.c_str());
    std::remove(config_path);
    return 1;
  }
  std::remove(config_path);
  if (loaded_cfg.count("epochs") != 0 || loaded_cfg["lr"] != "0.5") {
    std::fprintf(stderr, "Config file load should replace existing map entries\n");
    return 1;
  }

  char inline_config_arg[] = "--config=cli_parse_test.cfg";
  char *inline_config_argv[] = {arg0, arg1, inline_config_arg};
  std::string inline_config_path;
  {
    std::ofstream config_file(config_path);
    config_file << "save=FALSE\n";
  }
  if (!cli_parse::get_config_path(
          static_cast<int>(sizeof(inline_config_argv) / sizeof(inline_config_argv[0])),
          inline_config_argv, 2, &inline_config_path, &error) ||
      inline_config_path != config_path) {
    std::fprintf(stderr, "Inline --config path parsing failed: %s\n", error.c_str());
    std::remove(config_path);
    return 1;
  }
  Config inline_config_cfg;
  if (cli_parse::configure_command(
          static_cast<int>(sizeof(inline_config_argv) / sizeof(inline_config_argv[0])),
          inline_config_argv, 2, Options(), &inline_config_cfg, []() {}) != -1 ||
      inline_config_cfg.save) {
    std::fprintf(stderr, "Inline --config command parsing failed\n");
    std::remove(config_path);
    return 1;
  }
  std::remove(config_path);

  if (cli_parse::find_option_by_config_key(example_cli::TrainXorOptions(), "use-cublas") ==
          nullptr ||
      cli_parse::find_option_by_config_key(example_cli::TrainCharOptions(), "tf32") == nullptr ||
      cli_parse::find_option_by_flag(example_cli::TrainCharOptions(), "--top-p") == nullptr ||
      cli_parse::find_option_by_flag(example_cli::TrainCharOptions(), "--data") == nullptr ||
      cli_parse::find_option_by_flag(example_cli::TrainCharOptions(), "--val-fraction") ==
          nullptr ||
      cli_parse::find_option_by_flag(example_cli::TrainCharOptions(), "--best-checkpoint") ==
          nullptr ||
      cli_parse::find_option_by_flag(example_cli::TrainCharOptions(), "--prompt") == nullptr ||
      cli_parse::find_option_by_flag(example_cli::SampleCharOptions(), "--checkpoint") == nullptr ||
      cli_parse::find_option_by_config_key(example_cli::SampleCharOptions(), "data") == nullptr ||
      cli_parse::find_option_by_flag(example_cli::SampleCharOptions(), "--prompt") == nullptr) {
    std::fprintf(stderr, "Shared example option tables are missing expected options\n");
    return 1;
  }

  char train_char_arg0[] = "dl-cuda";
  char train_char_arg1[] = "train-char";
  char train_char_arg2[] = "--data";
  char train_char_arg3[] = "corpus.txt";
  char train_char_arg4[] = "--val-fraction";
  char train_char_arg5[] = "0.2";
  char train_char_arg6[] = "--early-stop-patience";
  char train_char_arg7[] = "3";
  char train_char_arg8[] = "--best-checkpoint";
  char train_char_arg9[] = "best.ckpt";
  char train_char_arg10[] = "--prompt";
  char train_char_arg11[] = "To be";
  char *train_char_argv[] = {train_char_arg0, train_char_arg1, train_char_arg2,  train_char_arg3,
                             train_char_arg4, train_char_arg5, train_char_arg6,  train_char_arg7,
                             train_char_arg8, train_char_arg9, train_char_arg10, train_char_arg11};
  dlcuda::TrainCharConfig train_char_cfg;
  parse_result = cli_parse::apply_command_line(
      static_cast<int>(sizeof(train_char_argv) / sizeof(train_char_argv[0])), train_char_argv, 2,
      example_cli::TrainCharOptions(), &train_char_cfg, &error);
  if (parse_result != cli_parse::ParseResult::kOk || train_char_cfg.data_path != "corpus.txt" ||
      train_char_cfg.val_fraction != 0.2f || train_char_cfg.early_stop_patience != 3 ||
      train_char_cfg.best_checkpoint_path != "best.ckpt" || train_char_cfg.prompt != "To be") {
    std::fprintf(stderr, "Train char --data parsing failed: %s\n", error.c_str());
    return 1;
  }

  dlcuda::SampleCharConfig sample_char_cfg;
  std::unordered_map<std::string, std::string> sample_char_config = {{"data", "sample.txt"},
                                                                     {"prompt", "hello"}};
  if (!cli_parse::apply_config_map(sample_char_config, example_cli::SampleCharOptions(),
                                   &sample_char_cfg, &error) ||
      sample_char_cfg.data_path != "sample.txt" || sample_char_cfg.prompt != "hello") {
    std::fprintf(stderr, "Sample char data config parsing failed: %s\n", error.c_str());
    return 1;
  }

  char cfg_arg0[] = "dl-cuda";
  char cfg_arg1[] = "train";
  char cfg_arg2[] = "--config";
  char *dangling_config_argv[] = {cfg_arg0, cfg_arg1, cfg_arg2};
  parse_result = cli_parse::apply_command_line(
      static_cast<int>(sizeof(dangling_config_argv) / sizeof(dangling_config_argv[0])),
      dangling_config_argv, 2, Options(), &cfg, &error);
  if (parse_result != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Dangling --config should have failed\n");
    return 1;
  }
  if (cli_parse::apply_config_map(cfg_map, Options(), static_cast<Config *>(nullptr), &error)) {
    std::fprintf(stderr, "Null config target should have failed\n");
    return 1;
  }
  if (cli_parse::apply_command_line(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, 2,
                                    Options(), static_cast<Config *>(nullptr),
                                    &error) != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Null CLI target should have failed\n");
    return 1;
  }

  std::string config_path_out = "stale.cfg";
  char *no_config_argv[] = {arg0, arg1};
  if (!cli_parse::get_config_path(
          static_cast<int>(sizeof(no_config_argv) / sizeof(no_config_argv[0])), no_config_argv, 2,
          &config_path_out, &error) ||
      !config_path_out.empty()) {
    std::fprintf(stderr, "Missing --config should clear the output path\n");
    return 1;
  }
  if (cli_parse::get_config_path(3, nullptr, 2, &config_path_out, &error)) {
    std::fprintf(stderr, "Null argv config lookup should have failed\n");
    return 1;
  }
  if (cli_parse::get_config_path(2, no_config_argv, 3, &config_path_out, &error)) {
    std::fprintf(stderr, "Invalid config lookup range should have failed\n");
    return 1;
  }
  if (cli_parse::apply_command_line(3, nullptr, 2, Options(), &cfg, &error) !=
      cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Null argv CLI parse should have failed\n");
    return 1;
  }
  if (cli_parse::apply_command_line(2, no_config_argv, 3, Options(), &cfg, &error) !=
      cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Invalid CLI argument range should have failed\n");
    return 1;
  }
  bool has_help = true;
  if (!cli_parse::has_help_arg(static_cast<int>(sizeof(no_config_argv) / sizeof(no_config_argv[0])),
                               no_config_argv, 2, &has_help, &error) ||
      has_help) {
    std::fprintf(stderr, "Missing help should clear the help output flag\n");
    return 1;
  }
  if (cli_parse::has_help_arg(3, nullptr, 2, &has_help, &error)) {
    std::fprintf(stderr, "Null argv help scan should have failed\n");
    return 1;
  }

  char *null_value_argv[] = {arg0, arg1, arg2, nullptr};
  if (cli_parse::apply_command_line(
          static_cast<int>(sizeof(null_value_argv) / sizeof(null_value_argv[0])), null_value_argv,
          2, Options(), &cfg, &error) != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Null option value should have failed\n");
    return 1;
  }

  char *null_config_value_argv[] = {cfg_arg0, cfg_arg1, cfg_arg2, nullptr};
  if (cli_parse::get_config_path(
          static_cast<int>(sizeof(null_config_value_argv) / sizeof(null_config_value_argv[0])),
          null_config_value_argv, 2, &config_path_out, &error)) {
    std::fprintf(stderr, "Null --config path should have failed\n");
    return 1;
  }

  char empty_inline_config_arg[] = "--config=";
  char *empty_inline_config_argv[] = {cfg_arg0, cfg_arg1, empty_inline_config_arg};
  if (cli_parse::get_config_path(
          static_cast<int>(sizeof(empty_inline_config_argv) / sizeof(empty_inline_config_argv[0])),
          empty_inline_config_argv, 2, &config_path_out, &error)) {
    std::fprintf(stderr, "Empty inline --config path should have failed\n");
    return 1;
  }
  if (cli_parse::apply_command_line(
          static_cast<int>(sizeof(empty_inline_config_argv) / sizeof(empty_inline_config_argv[0])),
          empty_inline_config_argv, 2, Options(), &cfg, &error) != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Empty inline --config CLI parse should have failed\n");
    return 1;
  }

  char no_save_inline_arg[] = "--no-save=false";
  char *no_save_inline_argv[] = {arg0, arg1, no_save_inline_arg};
  if (cli_parse::apply_command_line(
          static_cast<int>(sizeof(no_save_inline_argv) / sizeof(no_save_inline_argv[0])),
          no_save_inline_argv, 2, Options(), &cfg, &error) != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Inline value on flag-only option should have failed\n");
    return 1;
  }

  char broken_value_arg[] = "--broken-value";
  char broken_value[] = "5";
  char *broken_value_argv[] = {arg0, arg1, broken_value_arg, broken_value};
  if (cli_parse::apply_command_line(
          static_cast<int>(sizeof(broken_value_argv) / sizeof(broken_value_argv[0])),
          broken_value_argv, 2, MalformedOptions(), &cfg,
          &error) != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Malformed value option should have failed\n");
    return 1;
  }

  char broken_flag_arg[] = "--broken-flag";
  char *broken_flag_argv[] = {arg0, arg1, broken_flag_arg};
  if (cli_parse::apply_command_line(
          static_cast<int>(sizeof(broken_flag_argv) / sizeof(broken_flag_argv[0])),
          broken_flag_argv, 2, MalformedOptions(), &cfg,
          &error) != cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "Malformed flag option should have failed\n");
    return 1;
  }

  std::printf("cli_tests: PASS\n");
  return 0;
}
