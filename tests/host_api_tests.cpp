#include "dl_cuda.hpp"
#include "dl_cuda/data.hpp"
#include "dl_cuda/dtype.hpp"
#include "dl_cuda/status.hpp"

#include <cstdio>
#include <string>

namespace {

struct NoDefault {
  explicit NoDefault(int value_in) : value(value_in) {}

  int value;
};

} // namespace

int main() {
  dlcuda::Status bad = dlcuda::Status::InvalidArgument("bad input");
  if (bad.ok() || bad.code() != dlcuda::StatusCode::kInvalidArgument ||
      bad.message() != "bad input") {
    std::fprintf(stderr, "Status round-trip mismatch\n");
    return 1;
  }
  if (std::string(bad.code_name()) != "InvalidArgument" ||
      bad.ToString() != "InvalidArgument: bad input" ||
      std::string(dlcuda::StatusCodeName(dlcuda::StatusCode::kOk)) != "Ok") {
    std::fprintf(stderr, "Status code names mismatch\n");
    return 1;
  }

  if (dlcuda::DTypeSize(dlcuda::DType::kFloat32) != 4 ||
      dlcuda::DTypeSize(dlcuda::DType::kInt32) != 4 ||
      dlcuda::DTypeSize(dlcuda::DType::kFloat16) != 2 ||
      dlcuda::DTypeSize(dlcuda::DType::kBFloat16) != 2) {
    std::fprintf(stderr, "DType sizes mismatch\n");
    return 1;
  }
  if (std::string(dlcuda::DTypeName(dlcuda::DType::kFloat16)) != "float16" ||
      std::string(dlcuda::DTypeName(dlcuda::DType::kBFloat16)) != "bfloat16" ||
      !dlcuda::IsFloatingPointDType(dlcuda::DType::kFloat16) ||
      !dlcuda::IsFloatingPointDType(dlcuda::DType::kBFloat16) ||
      dlcuda::IsFloatingPointDType(dlcuda::DType::kInt32)) {
    std::fprintf(stderr, "DType metadata mismatch\n");
    return 1;
  }

  dlcuda::Result<int> ok_value(7);
  if (!ok_value.ok() || ok_value.value() != 7) {
    std::fprintf(stderr, "Result success mismatch\n");
    return 1;
  }

  dlcuda::Result<int> bad_value(dlcuda::Status::NotFound("missing"));
  if (bad_value.ok() || bad_value.status().code() != dlcuda::StatusCode::kNotFound) {
    std::fprintf(stderr, "Result error mismatch\n");
    return 1;
  }
  dlcuda::Result<NoDefault> non_default(NoDefault(11));
  if (!non_default.ok() || non_default.value().value != 11) {
    std::fprintf(stderr, "Result non-default value mismatch\n");
    return 1;
  }
  dlcuda::Result<NoDefault> non_default_error(dlcuda::Status::Unsupported("unsupported"));
  if (non_default_error.ok() ||
      non_default_error.status().code() != dlcuda::StatusCode::kUnsupported) {
    std::fprintf(stderr, "Result non-default error mismatch\n");
    return 1;
  }

  dlcuda::CharVocab empty_vocab;
  if (empty_vocab.size() != 0 || empty_vocab.Encode('x') != -1) {
    std::fprintf(stderr, "Default CharVocab should reject all encodes\n");
    return 1;
  }

  auto vocab_result = dlcuda::CharVocab::Build("cab cab");
  if (!vocab_result.ok()) {
    std::fprintf(stderr, "CharVocab build failed: %s\n", vocab_result.status().message().c_str());
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

  std::printf("host_tests: PASS\n");
  return 0;
}
