#include "example_cli_options.hpp"

#include <cstdio>

namespace {

void PrintUsage() {
  example_cli::PrintCommandUsage("dl-cuda-xor", example_cli::TrainXorOptions());
}

} // namespace

int main(int argc, char **argv) {
  dlcuda::TrainXorConfig cfg;
  int setup_status =
      cli_parse::configure_command(argc, argv, 1, example_cli::TrainXorOptions(), &cfg, PrintUsage);
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
