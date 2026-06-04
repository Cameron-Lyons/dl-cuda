#pragma once

#include "dl_cuda/nn/module.hpp"

namespace dlcuda {

class MaxPool2d : public Module {
public:
  explicit MaxPool2d(int64_t kernel_size, int64_t stride = 0);
  MaxPool2d(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
            int64_t padding_h = 0, int64_t padding_w = 0);

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override;
  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override;

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override;

private:
  Status init_status_;
  int64_t kernel_h_ = 0;
  int64_t kernel_w_ = 0;
  int64_t stride_h_ = 0;
  int64_t stride_w_ = 0;
  int64_t padding_h_ = 0;
  int64_t padding_w_ = 0;
  int64_t last_batch_ = 0;
  int64_t last_channels_ = 0;
  int64_t last_input_h_ = 0;
  int64_t last_input_w_ = 0;
  int64_t last_output_h_ = 0;
  int64_t last_output_w_ = 0;
  DType dtype_ = DType::kFloat32;
  Tensor argmax_indices_;
  Tensor forward_output_;
  Tensor backward_output_;
};

} // namespace dlcuda
