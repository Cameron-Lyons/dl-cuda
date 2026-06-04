#include "common.hpp"

namespace dlcuda::gpu_tests {

bool RunLayerCoverageSmoke(dlcuda::RuntimeContext &ctx) {
  auto residual_input_result =
      dlcuda::Tensor::AllocateAsync({2}, dlcuda::DType::kFloat32, ctx.stream());
  auto residual_grad_result =
      dlcuda::Tensor::AllocateAsync({2}, dlcuda::DType::kFloat32, ctx.stream());
  if (!residual_input_result.ok() || !residual_grad_result.ok()) {
    std::fprintf(stderr, "Residual tensor allocation failed\n");
    return false;
  }
  dlcuda::Tensor residual_input = residual_input_result.value();
  dlcuda::Tensor residual_grad = residual_grad_result.value();
  if (!CopyFloatsToTensor(ctx, &residual_input, {-1.0f, 2.0f}, "residual input") ||
      !CopyFloatsToTensor(ctx, &residual_grad, {1.0f, 1.0f}, "residual grad")) {
    return false;
  }
  dlcuda::Residual residual(std::make_unique<dlcuda::ReLU>());
  dlcuda::Tensor residual_output;
  dlcuda::Tensor residual_input_grad;
  std::vector<float> host_values;
  if (!residual.Forward(ctx, residual_input, &residual_output).ok() ||
      !CopyTensorToFloats(ctx, residual_output, &host_values, "residual output") ||
      !CheckCloseVector(host_values, {-1.0f, 4.0f}, "Residual output") ||
      !residual.Backward(ctx, residual_grad, &residual_input_grad).ok() ||
      !CopyTensorToFloats(ctx, residual_input_grad, &host_values, "residual input grad") ||
      !CheckCloseVector(host_values, {1.0f, 2.0f}, "Residual grad")) {
    return false;
  }

  auto conv_input_result =
      dlcuda::Tensor::AllocateAsync({1, 1, 3, 3}, dlcuda::DType::kFloat32, ctx.stream());
  auto conv_grad_result =
      dlcuda::Tensor::AllocateAsync({1, 1, 2, 2}, dlcuda::DType::kFloat32, ctx.stream());
  if (!conv_input_result.ok() || !conv_grad_result.ok()) {
    std::fprintf(stderr, "Conv2d tensor allocation failed\n");
    return false;
  }
  dlcuda::Tensor conv_input = conv_input_result.value();
  dlcuda::Tensor conv_grad = conv_grad_result.value();
  if (!CopyFloatsToTensor(ctx, &conv_input, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                          "conv input") ||
      !CopyFloatsToTensor(ctx, &conv_grad, {1.0f, 1.0f, 1.0f, 1.0f}, "conv grad")) {
    return false;
  }
  dlcuda::Conv2d conv(1, 1, 2, 2, ctx);
  std::vector<dlcuda::ParameterRef> conv_params;
  conv.AppendParameters("", &conv_params);
  if (conv_params.size() != 2 ||
      !CopyFloatsToTensor(ctx, conv_params[0].value, {1.0f, 0.0f, 0.0f, -1.0f}, "conv weight") ||
      !CopyFloatsToTensor(ctx, conv_params[1].value, {0.5f}, "conv bias")) {
    return false;
  }
  dlcuda::Tensor conv_output;
  dlcuda::Tensor conv_input_grad;
  if (!conv.Forward(ctx, conv_input, &conv_output).ok() ||
      !CopyTensorToFloats(ctx, conv_output, &host_values, "conv output") ||
      !CheckCloseVector(host_values, {-3.5f, -3.5f, -3.5f, -3.5f}, "Conv2d output") ||
      !conv.Backward(ctx, conv_grad, &conv_input_grad).ok() ||
      !CopyTensorToFloats(ctx, conv_input_grad, &host_values, "conv input grad") ||
      !CheckCloseVector(host_values, {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, -1.0f, -1.0f},
                        "Conv2d input grad") ||
      !CopyTensorToFloats(ctx, *conv_params[0].grad, &host_values, "conv weight grad") ||
      !CheckCloseVector(host_values, {12.0f, 16.0f, 24.0f, 28.0f}, "Conv2d weight grad") ||
      !CopyTensorToFloats(ctx, *conv_params[1].grad, &host_values, "conv bias grad") ||
      !CheckCloseVector(host_values, {4.0f}, "Conv2d bias grad")) {
    return false;
  }

  auto pool_input_result =
      dlcuda::Tensor::AllocateAsync({1, 1, 2, 3}, dlcuda::DType::kFloat32, ctx.stream());
  auto pool_grad_result =
      dlcuda::Tensor::AllocateAsync({1, 1, 1, 2}, dlcuda::DType::kFloat32, ctx.stream());
  if (!pool_input_result.ok() || !pool_grad_result.ok()) {
    std::fprintf(stderr, "MaxPool2d tensor allocation failed\n");
    return false;
  }
  dlcuda::Tensor pool_input = pool_input_result.value();
  dlcuda::Tensor pool_grad = pool_grad_result.value();
  if (!CopyFloatsToTensor(ctx, &pool_input, {1.0f, 3.0f, 2.0f, 4.0f, 0.0f, 5.0f}, "pool input") ||
      !CopyFloatsToTensor(ctx, &pool_grad, {1.0f, 1.0f}, "pool grad")) {
    return false;
  }
  dlcuda::MaxPool2d pool(2, 2, 1, 1);
  dlcuda::Tensor pool_output;
  dlcuda::Tensor pool_input_grad;
  if (!pool.Forward(ctx, pool_input, &pool_output).ok() ||
      !CopyTensorToFloats(ctx, pool_output, &host_values, "pool output") ||
      !CheckCloseVector(host_values, {4.0f, 5.0f}, "MaxPool2d output") ||
      !pool.Backward(ctx, pool_grad, &pool_input_grad).ok() ||
      !CopyTensorToFloats(ctx, pool_input_grad, &host_values, "pool input grad") ||
      !CheckCloseVector(host_values, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f}, "MaxPool2d grad")) {
    return false;
  }

  auto unary_input_result =
      dlcuda::Tensor::AllocateAsync({3}, dlcuda::DType::kFloat32, ctx.stream());
  auto unary_grad_result =
      dlcuda::Tensor::AllocateAsync({3}, dlcuda::DType::kFloat32, ctx.stream());
  if (!unary_input_result.ok() || !unary_grad_result.ok()) {
    std::fprintf(stderr, "Unary layer allocation failed\n");
    return false;
  }
  dlcuda::Tensor unary_input = unary_input_result.value();
  dlcuda::Tensor unary_grad = unary_grad_result.value();
  if (!CopyFloatsToTensor(ctx, &unary_input, {-1.0f, 0.0f, 2.0f}, "unary input") ||
      !CopyFloatsToTensor(ctx, &unary_grad, {1.0f, 1.0f, 1.0f}, "unary grad")) {
    return false;
  }
  dlcuda::GELU gelu;
  dlcuda::Tensor gelu_output;
  dlcuda::Tensor gelu_input_grad;
  if (!gelu.Forward(ctx, unary_input, &gelu_output).ok() ||
      !CopyTensorToFloats(ctx, gelu_output, &host_values, "gelu output") ||
      !CheckCloseVector(host_values, {GELUValue(-1.0f), GELUValue(0.0f), GELUValue(2.0f)},
                        "GELU output") ||
      !gelu.Backward(ctx, unary_grad, &gelu_input_grad).ok() ||
      !CopyTensorToFloats(ctx, gelu_input_grad, &host_values, "gelu input grad") ||
      !CheckCloseVector(host_values, {GELUGrad(-1.0f), GELUGrad(0.0f), GELUGrad(2.0f)},
                        "GELU grad")) {
    return false;
  }

  dlcuda::Dropout dropout(0.0f, 1234ULL);
  dlcuda::Tensor dropout_output;
  dlcuda::Tensor dropout_input_grad;
  if (!dropout.Forward(ctx, unary_input, &dropout_output).ok() ||
      !CopyTensorToFloats(ctx, dropout_output, &host_values, "dropout output") ||
      !CheckCloseVector(host_values, {-1.0f, 0.0f, 2.0f}, "Dropout output") ||
      !dropout.Backward(ctx, unary_grad, &dropout_input_grad).ok() ||
      !CopyTensorToFloats(ctx, dropout_input_grad, &host_values, "dropout grad") ||
      !CheckCloseVector(host_values, {1.0f, 1.0f, 1.0f}, "Dropout grad")) {
    return false;
  }

  auto norm_input_result =
      dlcuda::Tensor::AllocateAsync({2, 3}, dlcuda::DType::kFloat32, ctx.stream());
  auto norm_grad_result =
      dlcuda::Tensor::AllocateAsync({2, 3}, dlcuda::DType::kFloat32, ctx.stream());
  if (!norm_input_result.ok() || !norm_grad_result.ok()) {
    std::fprintf(stderr, "LayerNorm tensor allocation failed\n");
    return false;
  }
  dlcuda::Tensor norm_input = norm_input_result.value();
  dlcuda::Tensor norm_grad = norm_grad_result.value();
  std::vector<float> norm_values = {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 4.0f};
  if (!CopyFloatsToTensor(ctx, &norm_input, norm_values, "layernorm input") ||
      !CopyFloatsToTensor(ctx, &norm_grad, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                          "layernorm grad")) {
    return false;
  }
  std::vector<float> expected_layernorm(6);
  for (int row = 0; row < 2; ++row) {
    float mean =
        (norm_values[row * 3] + norm_values[row * 3 + 1] + norm_values[row * 3 + 2]) / 3.0f;
    float var = 0.0f;
    for (int col = 0; col < 3; ++col) {
      float centered = norm_values[row * 3 + col] - mean;
      var += centered * centered;
    }
    float inv_std = 1.0f / std::sqrt(var / 3.0f + 1e-5f);
    for (int col = 0; col < 3; ++col) {
      expected_layernorm[row * 3 + col] = (norm_values[row * 3 + col] - mean) * inv_std;
    }
  }
  dlcuda::LayerNorm layer_norm(3, ctx);
  std::vector<dlcuda::ParameterRef> layer_norm_params;
  layer_norm.AppendParameters("", &layer_norm_params);
  dlcuda::Tensor norm_output;
  dlcuda::Tensor norm_input_grad;
  if (!layer_norm.Forward(ctx, norm_input, &norm_output).ok() ||
      !CopyTensorToFloats(ctx, norm_output, &host_values, "layernorm output") ||
      !CheckCloseVector(host_values, expected_layernorm, "LayerNorm output", 2e-4f) ||
      !layer_norm.Backward(ctx, norm_grad, &norm_input_grad).ok() ||
      !CopyTensorToFloats(ctx, norm_input_grad, &host_values, "layernorm input grad") ||
      !CheckCloseVector(host_values, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "LayerNorm grad",
                        2e-4f) ||
      !CopyTensorToFloats(ctx, *layer_norm_params[1].grad, &host_values, "layernorm beta grad") ||
      !CheckCloseVector(host_values, {2.0f, 2.0f, 2.0f}, "LayerNorm beta grad")) {
    return false;
  }

  auto batch_input_result =
      dlcuda::Tensor::AllocateAsync({3, 2}, dlcuda::DType::kFloat32, ctx.stream());
  auto batch_grad_result =
      dlcuda::Tensor::AllocateAsync({3, 2}, dlcuda::DType::kFloat32, ctx.stream());
  if (!batch_input_result.ok() || !batch_grad_result.ok()) {
    std::fprintf(stderr, "BatchNorm1d tensor allocation failed\n");
    return false;
  }
  dlcuda::Tensor batch_input = batch_input_result.value();
  dlcuda::Tensor batch_grad = batch_grad_result.value();
  std::vector<float> batch_values = {1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 4.0f};
  if (!CopyFloatsToTensor(ctx, &batch_input, batch_values, "batchnorm input") ||
      !CopyFloatsToTensor(ctx, &batch_grad, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                          "batchnorm grad")) {
    return false;
  }
  std::vector<float> expected_batchnorm(6);
  for (int feature = 0; feature < 2; ++feature) {
    float mean =
        (batch_values[feature] + batch_values[2 + feature] + batch_values[4 + feature]) / 3.0f;
    float var = 0.0f;
    for (int row = 0; row < 3; ++row) {
      float centered = batch_values[row * 2 + feature] - mean;
      var += centered * centered;
    }
    float inv_std = 1.0f / std::sqrt(var / 3.0f + 1e-5f);
    for (int row = 0; row < 3; ++row) {
      expected_batchnorm[row * 2 + feature] = (batch_values[row * 2 + feature] - mean) * inv_std;
    }
  }
  dlcuda::BatchNorm1d batch_norm(2, ctx);
  std::vector<dlcuda::ParameterRef> batch_norm_params;
  batch_norm.AppendParameters("", &batch_norm_params);
  dlcuda::Tensor batch_output;
  dlcuda::Tensor batch_input_grad;
  if (!batch_norm.Forward(ctx, batch_input, &batch_output).ok() ||
      !CopyTensorToFloats(ctx, batch_output, &host_values, "batchnorm output") ||
      !CheckCloseVector(host_values, expected_batchnorm, "BatchNorm1d output", 2e-4f) ||
      !batch_norm.Backward(ctx, batch_grad, &batch_input_grad).ok() ||
      !CopyTensorToFloats(ctx, batch_input_grad, &host_values, "batchnorm input grad") ||
      !CheckCloseVector(host_values, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "BatchNorm1d grad",
                        2e-4f) ||
      !CopyTensorToFloats(ctx, *batch_norm_params[1].grad, &host_values, "batchnorm beta grad") ||
      !CheckCloseVector(host_values, {3.0f, 3.0f}, "BatchNorm1d beta grad")) {
    return false;
  }

  return true;
}

} // namespace dlcuda::gpu_tests
