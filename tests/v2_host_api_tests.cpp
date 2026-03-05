#include "dl_cuda/data.hpp"
#include "dl_cuda/status.hpp"

#include <cstdio>
#include <string>

int main() {
  dlcuda::Status bad = dlcuda::Status::InvalidArgument("bad input");
  if (bad.ok() || bad.code() != dlcuda::StatusCode::kInvalidArgument ||
      bad.message() != "bad input") {
    std::fprintf(stderr, "Status round-trip mismatch\n");
    return 1;
  }

  dlcuda::Result<int> ok_value(7);
  if (!ok_value.ok() || ok_value.value() != 7) {
    std::fprintf(stderr, "Result success mismatch\n");
    return 1;
  }

  dlcuda::Result<int> bad_value(dlcuda::Status::NotFound("missing"));
  if (bad_value.ok() ||
      bad_value.status().code() != dlcuda::StatusCode::kNotFound) {
    std::fprintf(stderr, "Result error mismatch\n");
    return 1;
  }

  auto vocab_result = dlcuda::CharVocab::Build("cab cab");
  if (!vocab_result.ok()) {
    std::fprintf(stderr, "CharVocab build failed: %s\n",
                 vocab_result.status().message().c_str());
    return 1;
  }

  dlcuda::CharVocab vocab = vocab_result.value();
  if (vocab.size() != 4) {
    std::fprintf(stderr, "Unexpected vocabulary size: %d\n", vocab.size());
    return 1;
  }
  if (vocab.Encode(' ') != 0 || vocab.Encode('a') != 1 || vocab.Encode('b') != 2 ||
      vocab.Encode('c') != 3) {
    std::fprintf(stderr, "Unexpected CharVocab ordering\n");
    return 1;
  }
  if (vocab.Decode(3) != 'c' || vocab.Encode('z') != -1) {
    std::fprintf(stderr, "Unexpected CharVocab encode/decode behavior\n");
    return 1;
  }

  std::printf("v2_host_tests: PASS\n");
  return 0;
}
