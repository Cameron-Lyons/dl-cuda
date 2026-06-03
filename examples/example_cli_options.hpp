#pragma once

#include "cli_parse_utils.hpp"
#include "dl_cuda/examples.hpp"

#include <cstdio>
#include <vector>

namespace example_cli {

inline const std::vector<cli_parse::OptionSpec<dlcuda::TrainXorConfig>> &TrainXorOptions() {
  static const std::vector<cli_parse::OptionSpec<dlcuda::TrainXorConfig>> options = {
      cli_parse::int_option("--epochs", &dlcuda::TrainXorConfig::epochs),
      cli_parse::int_option("--print-every", &dlcuda::TrainXorConfig::print_every),
      cli_parse::int_option("--hidden-size", &dlcuda::TrainXorConfig::hidden_size),
      cli_parse::float_option("--lr", &dlcuda::TrainXorConfig::lr),
      cli_parse::float_option("--grad-clip", &dlcuda::TrainXorConfig::grad_clip),
      cli_parse::u64_option("--seed", &dlcuda::TrainXorConfig::seed),
      cli_parse::string_option("--checkpoint", &dlcuda::TrainXorConfig::checkpoint_path),
      cli_parse::bool_flag_option("--resume", &dlcuda::TrainXorConfig::resume, true),
      cli_parse::bool_flag_option("--no-save", &dlcuda::TrainXorConfig::save, false, "save"),
      cli_parse::bool_flag_option("--no-cublas", &dlcuda::TrainXorConfig::use_cublas, false,
                                  "use-cublas"),
      cli_parse::bool_flag_option("--no-tf32", &dlcuda::TrainXorConfig::tf32, false, "tf32"),
  };
  return options;
}

inline const std::vector<cli_parse::OptionSpec<dlcuda::TrainCharConfig>> &TrainCharOptions() {
  static const std::vector<cli_parse::OptionSpec<dlcuda::TrainCharConfig>> options = {
      cli_parse::int_option("--seq-len", &dlcuda::TrainCharConfig::seq_len),
      cli_parse::int_option("--d-model", &dlcuda::TrainCharConfig::d_model),
      cli_parse::int_option("--epochs", &dlcuda::TrainCharConfig::epochs),
      cli_parse::int_option("--print-every", &dlcuda::TrainCharConfig::print_every),
      cli_parse::float_option("--lr", &dlcuda::TrainCharConfig::lr),
      cli_parse::float_option("--grad-clip", &dlcuda::TrainCharConfig::grad_clip),
      cli_parse::float_option("--temperature", &dlcuda::TrainCharConfig::temperature),
      cli_parse::float_option("--top-p", &dlcuda::TrainCharConfig::top_p),
      cli_parse::int_option("--gen-len", &dlcuda::TrainCharConfig::gen_len),
      cli_parse::u64_option("--seed", &dlcuda::TrainCharConfig::seed),
      cli_parse::u64_option("--sample-seed", &dlcuda::TrainCharConfig::sample_seed),
      cli_parse::string_option("--checkpoint", &dlcuda::TrainCharConfig::checkpoint_path),
      cli_parse::bool_flag_option("--resume", &dlcuda::TrainCharConfig::resume, true),
      cli_parse::bool_flag_option("--no-save", &dlcuda::TrainCharConfig::save, false, "save"),
      cli_parse::bool_flag_option("--no-cublas", &dlcuda::TrainCharConfig::use_cublas, false,
                                  "use-cublas"),
      cli_parse::bool_flag_option("--no-tf32", &dlcuda::TrainCharConfig::tf32, false, "tf32"),
  };
  return options;
}

inline const std::vector<cli_parse::OptionSpec<dlcuda::SampleCharConfig>> &SampleCharOptions() {
  static const std::vector<cli_parse::OptionSpec<dlcuda::SampleCharConfig>> options = {
      cli_parse::int_option("--seq-len", &dlcuda::SampleCharConfig::seq_len),
      cli_parse::int_option("--d-model", &dlcuda::SampleCharConfig::d_model),
      cli_parse::int_option("--gen-len", &dlcuda::SampleCharConfig::gen_len),
      cli_parse::float_option("--temperature", &dlcuda::SampleCharConfig::temperature),
      cli_parse::float_option("--top-p", &dlcuda::SampleCharConfig::top_p),
      cli_parse::u64_option("--seed", &dlcuda::SampleCharConfig::seed),
      cli_parse::u64_option("--sample-seed", &dlcuda::SampleCharConfig::sample_seed),
      cli_parse::string_option("--checkpoint", &dlcuda::SampleCharConfig::checkpoint_path),
      cli_parse::bool_flag_option("--no-cublas", &dlcuda::SampleCharConfig::use_cublas, false,
                                  "use-cublas"),
      cli_parse::bool_flag_option("--no-tf32", &dlcuda::SampleCharConfig::tf32, false, "tf32"),
  };
  return options;
}

template <typename Config>
inline void PrintOptions(const std::vector<cli_parse::OptionSpec<Config>> &options) {
  std::puts("  --config PATH");
  std::puts("  --help");
  for (const auto &option : options) {
    if (option.cli_takes_value) {
      std::printf("  %s VALUE\n", option.flag);
    } else {
      std::printf("  %s\n", option.flag);
    }
  }
}

template <typename Config>
inline void PrintCommandUsage(const char *invocation,
                              const std::vector<cli_parse::OptionSpec<Config>> &options) {
  std::printf("Usage: %s [options]\n", invocation);
  PrintOptions(options);
}

} // namespace example_cli
