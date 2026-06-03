#include "example_cli_options.hpp"
#include "dl_cuda_examples.hpp"

#include <cstdio>
#include <string>

namespace {

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
    auto print_usage = []() {
      example_cli::PrintCommandUsage("dl-cuda train-xor", example_cli::TrainXorOptions());
    };
    int setup_status = cli_parse::configure_command(argc, argv, 2, example_cli::TrainXorOptions(),
                                                    &cfg, print_usage);
    if (setup_status >= 0) {
      return setup_status;
    }

    dlcuda::Status status = dlcuda::TrainXor(cfg);
    if (!status.ok()) {
      std::fprintf(stderr, "train-xor failed: %s\n", status.ToString().c_str());
      return 1;
    }
    return 0;
  }

  if (command == "train-char") {
    dlcuda::TrainCharConfig cfg;
    auto print_usage = []() {
      example_cli::PrintCommandUsage("dl-cuda train-char", example_cli::TrainCharOptions());
    };
    int setup_status = cli_parse::configure_command(argc, argv, 2, example_cli::TrainCharOptions(),
                                                    &cfg, print_usage);
    if (setup_status >= 0) {
      return setup_status;
    }

    dlcuda::Status status = dlcuda::TrainChar(cfg);
    if (!status.ok()) {
      std::fprintf(stderr, "train-char failed: %s\n", status.ToString().c_str());
      return 1;
    }
    return 0;
  }

  if (command == "sample-char") {
    dlcuda::SampleCharConfig cfg;
    auto print_usage = []() {
      example_cli::PrintCommandUsage("dl-cuda sample-char", example_cli::SampleCharOptions());
    };
    int setup_status = cli_parse::configure_command(argc, argv, 2, example_cli::SampleCharOptions(),
                                                    &cfg, print_usage);
    if (setup_status >= 0) {
      return setup_status;
    }

    auto result = dlcuda::SampleChar(cfg);
    if (!result.ok()) {
      std::fprintf(stderr, "sample-char failed: %s\n", result.status().ToString().c_str());
      return 1;
    }
    std::printf("%s\n", result.value().c_str());
    return 0;
  }

  std::fprintf(stderr, "Unknown subcommand: %s\n", command.c_str());
  PrintUsage();
  return 1;
}
