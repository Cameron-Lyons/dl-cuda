#include "cli_parse_utils.hpp"
#include "dl_cuda_examples.hpp"

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

template <typename Config>
void PrintCommandUsage(const char *command,
                       const std::vector<cli_parse::OptionSpec<Config>> &options) {
  std::printf("Usage: dl-cuda %s [options]\n", command);
  std::puts("  --config PATH");
  for (const auto &option : options) {
    if (option.cli_takes_value) {
      std::printf("  %s VALUE\n", option.flag);
    } else {
      std::printf("  %s\n", option.flag);
    }
  }
}

void PrintUsage() {
  std::puts("Usage: dl-cuda <subcommand> [options]");
  std::puts("Subcommands:");
  std::puts("  train-xor   Train XOR MLP model");
  std::puts("  train-char  Train char-level model");
  std::puts("  sample-char Generate text from char-level checkpoint");
  std::puts("\nCommon options:");
  std::puts("  --config PATH     Load key=value config file");
  std::puts("\nConfig file keys now match CLI option names without the leading '--'.");
  std::puts("Example: print-every=50, use-cublas=false");
}

const std::vector<cli_parse::OptionSpec<dlcuda::TrainXorConfig>> &
TrainXorOptions() {
  static const std::vector<cli_parse::OptionSpec<dlcuda::TrainXorConfig>> options = {
      cli_parse::int_option("--epochs", &dlcuda::TrainXorConfig::epochs),
      cli_parse::int_option("--print-every",
                            &dlcuda::TrainXorConfig::print_every),
      cli_parse::int_option("--hidden-size",
                            &dlcuda::TrainXorConfig::hidden_size),
      cli_parse::float_option("--lr", &dlcuda::TrainXorConfig::lr),
      cli_parse::float_option("--grad-clip",
                              &dlcuda::TrainXorConfig::grad_clip),
      cli_parse::u64_option("--seed", &dlcuda::TrainXorConfig::seed),
      cli_parse::string_option("--checkpoint",
                               &dlcuda::TrainXorConfig::checkpoint_path),
      cli_parse::bool_flag_option("--resume", &dlcuda::TrainXorConfig::resume,
                                  true),
      cli_parse::bool_flag_option("--no-save", &dlcuda::TrainXorConfig::save,
                                  false, "save"),
      cli_parse::bool_flag_option("--no-cublas",
                                  &dlcuda::TrainXorConfig::use_cublas, false,
                                  "use-cublas"),
      cli_parse::bool_flag_option("--no-tf32", &dlcuda::TrainXorConfig::tf32,
                                  false),
  };
  return options;
}

const std::vector<cli_parse::OptionSpec<dlcuda::TrainCharConfig>> &
TrainCharOptions() {
  static const std::vector<cli_parse::OptionSpec<dlcuda::TrainCharConfig>> options = {
      cli_parse::int_option("--seq-len", &dlcuda::TrainCharConfig::seq_len),
      cli_parse::int_option("--d-model", &dlcuda::TrainCharConfig::d_model),
      cli_parse::int_option("--epochs", &dlcuda::TrainCharConfig::epochs),
      cli_parse::int_option("--print-every",
                            &dlcuda::TrainCharConfig::print_every),
      cli_parse::float_option("--lr", &dlcuda::TrainCharConfig::lr),
      cli_parse::float_option("--grad-clip",
                              &dlcuda::TrainCharConfig::grad_clip),
      cli_parse::float_option("--temperature",
                              &dlcuda::TrainCharConfig::temperature),
      cli_parse::float_option("--top-p", &dlcuda::TrainCharConfig::top_p),
      cli_parse::int_option("--gen-len", &dlcuda::TrainCharConfig::gen_len),
      cli_parse::u64_option("--seed", &dlcuda::TrainCharConfig::seed),
      cli_parse::u64_option("--sample-seed",
                            &dlcuda::TrainCharConfig::sample_seed),
      cli_parse::string_option("--checkpoint",
                               &dlcuda::TrainCharConfig::checkpoint_path),
      cli_parse::bool_flag_option("--resume", &dlcuda::TrainCharConfig::resume,
                                  true),
      cli_parse::bool_flag_option("--no-save", &dlcuda::TrainCharConfig::save,
                                  false, "save"),
      cli_parse::bool_flag_option("--no-cublas",
                                  &dlcuda::TrainCharConfig::use_cublas, false,
                                  "use-cublas"),
      cli_parse::bool_flag_option("--no-tf32", &dlcuda::TrainCharConfig::tf32,
                                  false),
  };
  return options;
}

const std::vector<cli_parse::OptionSpec<dlcuda::SampleCharConfig>> &
SampleCharOptions() {
  static const std::vector<cli_parse::OptionSpec<dlcuda::SampleCharConfig>> options = {
      cli_parse::int_option("--seq-len", &dlcuda::SampleCharConfig::seq_len),
      cli_parse::int_option("--d-model", &dlcuda::SampleCharConfig::d_model),
      cli_parse::int_option("--gen-len", &dlcuda::SampleCharConfig::gen_len),
      cli_parse::float_option("--temperature",
                              &dlcuda::SampleCharConfig::temperature),
      cli_parse::float_option("--top-p", &dlcuda::SampleCharConfig::top_p),
      cli_parse::u64_option("--seed", &dlcuda::SampleCharConfig::seed),
      cli_parse::u64_option("--sample-seed",
                            &dlcuda::SampleCharConfig::sample_seed),
      cli_parse::string_option("--checkpoint",
                               &dlcuda::SampleCharConfig::checkpoint_path),
      cli_parse::bool_flag_option("--no-cublas",
                                  &dlcuda::SampleCharConfig::use_cublas, false,
                                  "use-cublas"),
      cli_parse::bool_flag_option("--no-tf32", &dlcuda::SampleCharConfig::tf32,
                                  false),
  };
  return options;
}

template <typename Config, typename PrintUsageFn>
int ConfigureCommand(int argc, char **argv,
                     const std::vector<cli_parse::OptionSpec<Config>> &options,
                     Config *cfg, const PrintUsageFn &print_usage) {
  std::string config_path;
  std::string error;
  if (!cli_parse::get_config_path(argc, argv, 2, &config_path, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  std::unordered_map<std::string, std::string> cfg_map;
  if (!config_path.empty()) {
    if (!cli_parse::load_config_file(config_path, &cfg_map, &error)) {
      std::fprintf(stderr, "Config error: %s\n", error.c_str());
      return 1;
    }
  }

  if (!cli_parse::apply_config_map(cfg_map, options, cfg, &error)) {
    std::fprintf(stderr, "Config error: %s\n", error.c_str());
    return 1;
  }

  cli_parse::ParseResult parse_result =
      cli_parse::apply_command_line(argc, argv, 2, options, cfg, &error);
  if (parse_result == cli_parse::ParseResult::kHelp) {
    print_usage();
    return 0;
  }
  if (parse_result == cli_parse::ParseResult::kError) {
    std::fprintf(stderr, "%s\n", error.c_str());
    print_usage();
    return 1;
  }

  return -1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string command = argv[1];
  if (command == "--help" || command == "help") {
    PrintUsage();
    return 0;
  }

  if (command == "train-xor") {
    dlcuda::TrainXorConfig cfg;
    auto print_usage = []() { PrintCommandUsage("train-xor", TrainXorOptions()); };
    int setup_status = ConfigureCommand(argc, argv, TrainXorOptions(), &cfg, print_usage);
    if (setup_status >= 0) {
      return setup_status;
    }

    dlcuda::Status status = dlcuda::TrainXor(cfg);
    if (!status.ok()) {
      std::fprintf(stderr, "train-xor failed: %s\n", status.message().c_str());
      return 1;
    }
    return 0;
  }

  if (command == "train-char") {
    dlcuda::TrainCharConfig cfg;
    auto print_usage = []() { PrintCommandUsage("train-char", TrainCharOptions()); };
    int setup_status =
        ConfigureCommand(argc, argv, TrainCharOptions(), &cfg, print_usage);
    if (setup_status >= 0) {
      return setup_status;
    }

    dlcuda::Status status = dlcuda::TrainChar(cfg);
    if (!status.ok()) {
      std::fprintf(stderr, "train-char failed: %s\n", status.message().c_str());
      return 1;
    }
    return 0;
  }

  if (command == "sample-char") {
    dlcuda::SampleCharConfig cfg;
    auto print_usage = []() { PrintCommandUsage("sample-char", SampleCharOptions()); };
    int setup_status =
        ConfigureCommand(argc, argv, SampleCharOptions(), &cfg, print_usage);
    if (setup_status >= 0) {
      return setup_status;
    }

    auto result = dlcuda::SampleChar(cfg);
    if (!result.ok()) {
      std::fprintf(stderr, "sample-char failed: %s\n",
                   result.status().message().c_str());
      return 1;
    }
    std::printf("%s\n", result.value().c_str());
    return 0;
  }

  std::fprintf(stderr, "Unknown subcommand: %s\n", command.c_str());
  PrintUsage();
  return 1;
}
