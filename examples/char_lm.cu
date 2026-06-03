#include "example_cli_options.hpp"

#include <cstdio>

namespace {

void PrintUsage() {
  example_cli::PrintCommandUsage("dl-cuda-char-lm", example_cli::TrainCharOptions());
}

} // namespace

int main(int argc, char **argv) {
  dlcuda::TrainCharConfig cfg;
  int setup_status = cli_parse::configure_command(argc, argv, 1, example_cli::TrainCharOptions(),
                                                  &cfg, PrintUsage);
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
