#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class Embedding : public Module {
public:
  Embedding(int64_t vocab_size, int64_t embedding_dim, RuntimeContext &ctx,
            DType dtype = DType::kFloat32);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t vocab_size_ = 0;
  int64_t embedding_dim_ = 0;
  int64_t last_num_tokens_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor table_;
  Tensor grad_table_;
  Tensor cached_token_ids_;
  Tensor forward_output_;
};

} // namespace dlcuda
