#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class Conv2d : public Module {
public:
  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, RuntimeContext &ctx,
         DType dtype = DType::kFloat32);
  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
         RuntimeContext &ctx, int64_t stride_h = 1, int64_t stride_w = 1, int64_t padding_h = 0,
         int64_t padding_w = 0, DType dtype = DType::kFloat32);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t in_channels_ = 0;
  int64_t out_channels_ = 0;
  int64_t kernel_h_ = 0;
  int64_t kernel_w_ = 0;
  int64_t stride_h_ = 1;
  int64_t stride_w_ = 1;
  int64_t padding_h_ = 0;
  int64_t padding_w_ = 0;
  int64_t last_batch_ = 0;
  int64_t last_input_h_ = 0;
  int64_t last_input_w_ = 0;
  int64_t last_output_h_ = 0;
  int64_t last_output_w_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor weight_;
  Tensor bias_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
  Tensor column_buffer_;
  Tensor grad_column_buffer_;
  Tensor grad_output_matrix_;
};

} // namespace dlcuda
