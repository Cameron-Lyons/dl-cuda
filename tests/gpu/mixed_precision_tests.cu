#include "common.hpp"

namespace dlcuda::gpu_tests {

bool RunMixedPrecisionSmoke(dlcuda::RuntimeContext &ctx, dlcuda::DType dtype) {
  const char *dtype_name = dlcuda::DTypeName(dtype);
  constexpr float kTolerance = 3e-2f;

  auto matrix_result = dlcuda::Tensor::AllocateAsync({2, 2}, dtype, ctx.stream());
  auto row_result = dlcuda::Tensor::AllocateAsync({2}, dtype, ctx.stream());
  auto rhs_result = dlcuda::Tensor::AllocateAsync({2, 2}, dtype, ctx.stream());
  if (!matrix_result.ok() || !row_result.ok() || !rhs_result.ok()) {
    std::fprintf(stderr, "%s mixed tensor allocation failed\n", dtype_name);
    return false;
  }
  dlcuda::Tensor matrix = matrix_result.value();
  dlcuda::Tensor row = row_result.value();
  dlcuda::Tensor rhs = rhs_result.value();
  if (!CopyFloatsToTensor(ctx, &matrix, {1.0f, 2.0f, 3.0f, 4.0f}, "mixed matrix") ||
      !CopyFloatsToTensor(ctx, &row, {10.0f, 20.0f}, "mixed row") ||
      !CopyFloatsToTensor(ctx, &rhs, {5.0f, 6.0f, 7.0f, 8.0f}, "mixed rhs")) {
    return false;
  }

  dlcuda::Tensor added;
  if (!dlcuda::TensorAdd(ctx, matrix, row, &added).ok() || added.dtype() != dtype) {
    std::fprintf(stderr, "%s TensorAdd failed\n", dtype_name);
    return false;
  }
  std::vector<float> host_added;
  if (!CopyTensorToFloats(ctx, added, &host_added, "mixed add")) {
    return false;
  }
  std::vector<float> expected_added = {11.0f, 22.0f, 13.0f, 24.0f};
  for (size_t i = 0; i < expected_added.size(); ++i) {
    if (!AlmostEqual(host_added[i], expected_added[i], kTolerance)) {
      std::fprintf(stderr, "%s TensorAdd value mismatch at %zu\n", dtype_name, i);
      return false;
    }
  }

  dlcuda::Tensor matmul;
  if (!dlcuda::TensorMatMul(ctx, matrix, rhs, &matmul).ok() || matmul.dtype() != dtype) {
    std::fprintf(stderr, "%s TensorMatMul failed\n", dtype_name);
    return false;
  }
  std::vector<float> host_matmul;
  if (!CopyTensorToFloats(ctx, matmul, &host_matmul, "mixed matmul")) {
    return false;
  }
  std::vector<float> expected_matmul = {19.0f, 22.0f, 43.0f, 50.0f};
  for (size_t i = 0; i < expected_matmul.size(); ++i) {
    if (!AlmostEqual(host_matmul[i], expected_matmul[i], kTolerance)) {
      std::fprintf(stderr, "%s TensorMatMul value mismatch at %zu\n", dtype_name, i);
      return false;
    }
  }

  auto sum_result = dlcuda::TensorSum(ctx, added);
  if (!sum_result.ok() || !AlmostEqual(sum_result.value(), 70.0f, kTolerance)) {
    std::fprintf(stderr, "%s TensorSum failed\n", dtype_name);
    return false;
  }

  dlcuda::Sequential xor_model;
  if (!xor_model.Add(std::make_unique<dlcuda::Linear>(2, 4, ctx, dtype)).ok() ||
      !xor_model.Add(std::make_unique<dlcuda::ReLU>()).ok() ||
      !xor_model.Add(std::make_unique<dlcuda::Linear>(4, 1, ctx, dtype)).ok() ||
      !xor_model.Add(std::make_unique<dlcuda::Sigmoid>()).ok()) {
    std::fprintf(stderr, "%s mixed model construction failed\n", dtype_name);
    return false;
  }
  auto x_result = dlcuda::Tensor::AllocateAsync({4, 2}, dtype, ctx.stream());
  auto y_result = dlcuda::Tensor::AllocateAsync({4, 1}, dlcuda::DType::kFloat32, ctx.stream());
  if (!x_result.ok() || !y_result.ok()) {
    std::fprintf(stderr, "%s mixed training tensor allocation failed\n", dtype_name);
    return false;
  }
  dlcuda::Tensor x = x_result.value();
  dlcuda::Tensor y = y_result.value();
  if (!CopyFloatsToTensor(ctx, &x, {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, "mixed x") ||
      !CopyFloatsToTensor(ctx, &y, {0.0f, 1.0f, 1.0f, 0.0f}, "mixed y")) {
    return false;
  }

  dlcuda::Tensor predictions;
  if (!xor_model.Forward(ctx, x, &predictions).ok() || predictions.dtype() != dtype) {
    std::fprintf(stderr, "%s mixed forward failed\n", dtype_name);
    return false;
  }
  auto bce_loss = dlcuda::BinaryCrossEntropyLoss(ctx, y, predictions);
  if (!bce_loss.ok() || !std::isfinite(bce_loss.value())) {
    std::fprintf(stderr, "%s mixed BCE loss failed\n", dtype_name);
    return false;
  }
  dlcuda::Tensor loss_grad;
  if (!dlcuda::BinaryCrossEntropyBackward(ctx, y, predictions, &loss_grad).ok() ||
      loss_grad.dtype() != dtype) {
    std::fprintf(stderr, "%s mixed BCE backward failed\n", dtype_name);
    return false;
  }
  if (!xor_model.Backward(ctx, loss_grad, nullptr).ok()) {
    std::fprintf(stderr, "%s mixed model backward failed\n", dtype_name);
    return false;
  }

  float grad_norm = 0.0f;
  dlcuda::AdamOptimizer optimizer;
  const auto &xor_params = xor_model.parameters();
  if (!dlcuda::ClipGradNorm(ctx, xor_params, 1.0f, &grad_norm).ok() || !std::isfinite(grad_norm) ||
      !optimizer.Step(ctx, xor_params, 0.001f).ok()) {
    std::fprintf(stderr, "%s mixed optimizer path failed\n", dtype_name);
    return false;
  }

  dlcuda::AdamOptimizer trainer_optimizer;
  dlcuda::SupervisedTrainer trainer(ctx, xor_model, trainer_optimizer, xor_params);
  dlcuda::TrainStepOptions train_options;
  train_options.learning_rate = 0.001f;
  train_options.max_grad_norm = 1.0f;
  dlcuda::TrainStepResult train_result;
  if (!trainer.TrainBinaryClassificationStep(x, y, train_options, &train_result).ok() ||
      !std::isfinite(train_result.loss) || !std::isfinite(train_result.grad_norm)) {
    std::fprintf(stderr, "%s supervised trainer step failed\n", dtype_name);
    return false;
  }

  dlcuda::Sequential categorical_model;
  if (!categorical_model.Add(std::make_unique<dlcuda::Embedding>(3, 2, ctx, dtype)).ok() ||
      !categorical_model.Add(std::make_unique<dlcuda::Linear>(2, 3, ctx, dtype)).ok()) {
    std::fprintf(stderr, "%s mixed categorical model construction failed\n", dtype_name);
    return false;
  }
  auto token_ids_result = dlcuda::Tensor::AllocateAsync({2}, dlcuda::DType::kInt32, ctx.stream());
  auto target_ids_result = dlcuda::Tensor::AllocateAsync({2}, dlcuda::DType::kInt32, ctx.stream());
  if (!token_ids_result.ok() || !target_ids_result.ok()) {
    std::fprintf(stderr, "%s mixed token allocation failed\n", dtype_name);
    return false;
  }
  dlcuda::Tensor token_ids = token_ids_result.value();
  dlcuda::Tensor target_ids = target_ids_result.value();
  std::vector<int32_t> host_token_ids = {1, 1};
  std::vector<int32_t> host_target_ids = {0, 2};
  if (!token_ids
           .CopyFromHost(host_token_ids.data(), host_token_ids.size() * sizeof(int32_t),
                         ctx.stream())
           .ok() ||
      !target_ids
           .CopyFromHost(host_target_ids.data(), host_target_ids.size() * sizeof(int32_t),
                         ctx.stream())
           .ok()) {
    std::fprintf(stderr, "%s mixed token copy failed\n", dtype_name);
    return false;
  }

  dlcuda::Tensor logits;
  if (!categorical_model.Forward(ctx, token_ids, &logits).ok() || logits.dtype() != dtype) {
    std::fprintf(stderr, "%s mixed categorical forward failed\n", dtype_name);
    return false;
  }
  auto metrics = dlcuda::CategoricalCrossEntropyMetricsFromLogits(ctx, target_ids, logits);
  if (!metrics.ok() || !std::isfinite(metrics.value().loss)) {
    std::fprintf(stderr, "%s mixed categorical metrics failed\n", dtype_name);
    return false;
  }
  dlcuda::Softmax softmax;
  dlcuda::Tensor probabilities;
  dlcuda::Tensor softmax_grad;
  if (!softmax.Forward(ctx, logits, &probabilities).ok() || probabilities.dtype() != dtype ||
      !dlcuda::CategoricalCrossEntropyBackwardFromLogits(ctx, target_ids, logits, &softmax_grad)
           .ok() ||
      softmax_grad.dtype() != dtype || !softmax.Backward(ctx, softmax_grad, nullptr).ok() ||
      !categorical_model.Backward(ctx, softmax_grad, nullptr).ok()) {
    std::fprintf(stderr, "%s mixed categorical backward failed\n", dtype_name);
    return false;
  }

  return true;
}

} // namespace dlcuda::gpu_tests
