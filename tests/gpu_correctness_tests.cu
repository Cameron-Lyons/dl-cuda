#include "gpu/common.hpp"

using namespace dlcuda::gpu_tests;

int main() {
  if (!HasCudaDevice()) {
    std::printf("gpu_correctness_tests: SKIP (no CUDA device)\n");
    return 0;
  }

  dlcuda::RuntimeContext ctx;
  dlcuda::Status init = ctx.Initialize();
  if (!init.ok()) {
    std::fprintf(stderr, "Runtime initialization failed: %s\n", init.ToString().c_str());
    return 1;
  }

  dlcuda::Sequential model;
  if (!model.Add(std::make_unique<dlcuda::Linear>(2, 4, ctx)).ok() ||
      !model.Add(std::make_unique<dlcuda::ReLU>()).ok() ||
      !model.Add(std::make_unique<dlcuda::Linear>(4, 1, ctx)).ok() ||
      !model.Add(std::make_unique<dlcuda::Sigmoid>()).ok()) {
    std::fprintf(stderr, "Failed to build smoke-test model\n");
    return 1;
  }

  const auto &params = model.parameters();
  if (params.size() != 4 || params[0].name != "layers.0.weight" ||
      params[3].name != "layers.2.bias") {
    std::fprintf(stderr, "Unexpected parameter cache contents\n");
    return 1;
  }

  auto x_result = dlcuda::Tensor::AllocateAsync({4, 2}, dlcuda::DType::kFloat32, ctx.stream());
  auto y_result = dlcuda::Tensor::AllocateAsync({4, 1}, dlcuda::DType::kFloat32, ctx.stream());
  if (!x_result.ok() || !y_result.ok()) {
    std::fprintf(stderr, "Tensor allocation failed\n");
    return 1;
  }

  dlcuda::Tensor x = x_result.value();
  dlcuda::Tensor y = y_result.value();
  std::vector<float> host_x = {
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
  };
  std::vector<float> host_y = {0.0f, 1.0f, 1.0f, 0.0f};
  if (!x.CopyFromHost(host_x.data(), host_x.size() * sizeof(float), ctx.stream()).ok() ||
      !y.CopyFromHost(host_y.data(), host_y.size() * sizeof(float), ctx.stream()).ok()) {
    std::fprintf(stderr, "Host-to-device copy failed\n");
    return 1;
  }

  dlcuda::Tensor predictions;
  dlcuda::Status forward = model.Forward(ctx, x, &predictions);
  if (!forward.ok()) {
    std::fprintf(stderr, "Forward failed: %s\n", forward.ToString().c_str());
    return 1;
  }

  auto loss = dlcuda::BinaryCrossEntropyLoss(ctx, y, predictions);
  if (!loss.ok()) {
    std::fprintf(stderr, "Loss failed: %s\n", loss.status().ToString().c_str());
    return 1;
  }

  dlcuda::Tensor loss_grad;
  dlcuda::Status loss_backward =
      dlcuda::BinaryCrossEntropyBackward(ctx, y, predictions, &loss_grad);
  if (!loss_backward.ok()) {
    std::fprintf(stderr, "Loss backward failed: %s\n", loss_backward.ToString().c_str());
    return 1;
  }

  dlcuda::Tensor input_grad;
  dlcuda::Status backward = model.Backward(ctx, loss_grad, &input_grad);
  if (!backward.ok()) {
    std::fprintf(stderr, "Backward failed: %s\n", backward.ToString().c_str());
    return 1;
  }
  dlcuda::Status backward_without_input_grad = model.Backward(ctx, loss_grad, nullptr);
  if (!backward_without_input_grad.ok()) {
    std::fprintf(stderr, "Backward without input grad failed: %s\n",
                 backward_without_input_grad.ToString().c_str());
    return 1;
  }

  float grad_norm = 0.0f;
  dlcuda::Status clip_status = dlcuda::ClipGradNorm(ctx, params, 1.0f, &grad_norm);
  if (!clip_status.ok() || !(grad_norm >= 0.0f)) {
    std::fprintf(stderr, "ClipGradNorm failed\n");
    return 1;
  }

  if (predictions.rank() != 2 || predictions.dim(0) != 4 || predictions.dim(1) != 1 ||
      input_grad.rank() != 2 || input_grad.dim(0) != 4 || input_grad.dim(1) != 2) {
    std::fprintf(stderr, "Unexpected tensor shapes from forward/backward\n");
    return 1;
  }

  auto matrix_result = dlcuda::Tensor::AllocateAsync({2, 3}, dlcuda::DType::kFloat32, ctx.stream());
  auto row_result = dlcuda::Tensor::AllocateAsync({3}, dlcuda::DType::kFloat32, ctx.stream());
  auto scalar_result = dlcuda::Tensor::AllocateAsync({}, dlcuda::DType::kFloat32, ctx.stream());
  if (!matrix_result.ok() || !row_result.ok() || !scalar_result.ok()) {
    std::fprintf(stderr, "Tensor op allocation failed\n");
    return 1;
  }
  dlcuda::Tensor matrix = matrix_result.value();
  dlcuda::Tensor row = row_result.value();
  dlcuda::Tensor scalar = scalar_result.value();
  std::vector<float> host_matrix = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> host_row = {10.0f, 20.0f, 30.0f};
  float host_scalar = 2.0f;
  if (!matrix.CopyFromHost(host_matrix.data(), host_matrix.size() * sizeof(float), ctx.stream())
           .ok() ||
      !row.CopyFromHost(host_row.data(), host_row.size() * sizeof(float), ctx.stream()).ok() ||
      !scalar.CopyFromHost(&host_scalar, sizeof(host_scalar), ctx.stream()).ok()) {
    std::fprintf(stderr, "Tensor op host-to-device copy failed\n");
    return 1;
  }

  dlcuda::Tensor added;
  if (!dlcuda::TensorAdd(ctx, matrix, row, &added).ok()) {
    std::fprintf(stderr, "TensorAdd failed\n");
    return 1;
  }
  if (added.rank() != 2 || added.dim(0) != 2 || added.dim(1) != 3) {
    std::fprintf(stderr, "TensorAdd produced unexpected shape\n");
    return 1;
  }
  std::vector<float> host_added(6);
  if (!added.CopyToHost(host_added.data(), host_added.size() * sizeof(float), ctx.stream()).ok()) {
    std::fprintf(stderr, "TensorAdd copy failed\n");
    return 1;
  }
  if (!ctx.Synchronize().ok()) {
    std::fprintf(stderr, "TensorAdd sync failed\n");
    return 1;
  }
  std::vector<float> expected_added = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
  for (size_t i = 0; i < expected_added.size(); ++i) {
    if (!AlmostEqual(host_added[i], expected_added[i])) {
      std::fprintf(stderr, "TensorAdd value mismatch at %zu\n", i);
      return 1;
    }
  }

  dlcuda::Tensor multiplied;
  if (!dlcuda::TensorMultiply(ctx, matrix, scalar, &multiplied).ok()) {
    std::fprintf(stderr, "TensorMultiply failed\n");
    return 1;
  }
  std::vector<float> host_multiplied(6);
  if (!multiplied
           .CopyToHost(host_multiplied.data(), host_multiplied.size() * sizeof(float), ctx.stream())
           .ok()) {
    std::fprintf(stderr, "TensorMultiply copy failed\n");
    return 1;
  }
  if (!ctx.Synchronize().ok()) {
    std::fprintf(stderr, "TensorMultiply sync failed\n");
    return 1;
  }
  for (size_t i = 0; i < host_matrix.size(); ++i) {
    if (!AlmostEqual(host_multiplied[i], host_matrix[i] * 2.0f)) {
      std::fprintf(stderr, "TensorMultiply value mismatch at %zu\n", i);
      return 1;
    }
  }

  auto rhs_result = dlcuda::Tensor::AllocateAsync({3, 2}, dlcuda::DType::kFloat32, ctx.stream());
  if (!rhs_result.ok()) {
    std::fprintf(stderr, "MatMul RHS allocation failed\n");
    return 1;
  }
  dlcuda::Tensor rhs = rhs_result.value();
  std::vector<float> host_rhs = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  if (!rhs.CopyFromHost(host_rhs.data(), host_rhs.size() * sizeof(float), ctx.stream()).ok()) {
    std::fprintf(stderr, "MatMul RHS copy failed\n");
    return 1;
  }
  dlcuda::Tensor matmul;
  if (!dlcuda::TensorMatMul(ctx, matrix, rhs, &matmul).ok()) {
    std::fprintf(stderr, "TensorMatMul failed\n");
    return 1;
  }
  std::vector<float> host_matmul(4);
  if (!matmul.CopyToHost(host_matmul.data(), host_matmul.size() * sizeof(float), ctx.stream())
           .ok()) {
    std::fprintf(stderr, "TensorMatMul copy failed\n");
    return 1;
  }
  if (!ctx.Synchronize().ok()) {
    std::fprintf(stderr, "TensorMatMul sync failed\n");
    return 1;
  }
  std::vector<float> expected_matmul = {58.0f, 64.0f, 139.0f, 154.0f};
  for (size_t i = 0; i < expected_matmul.size(); ++i) {
    if (!AlmostEqual(host_matmul[i], expected_matmul[i])) {
      std::fprintf(stderr, "TensorMatMul value mismatch at %zu\n", i);
      return 1;
    }
  }

  auto sum_result = dlcuda::TensorSum(ctx, added);
  if (!sum_result.ok() || !AlmostEqual(sum_result.value(), 141.0f)) {
    std::fprintf(stderr, "TensorSum failed or returned unexpected value\n");
    return 1;
  }

  auto reshaped_result = matrix.Reshape({3, 2});
  if (!reshaped_result.ok()) {
    std::fprintf(stderr, "Tensor reshape failed\n");
    return 1;
  }
  dlcuda::Tensor reshaped = reshaped_result.value();
  if (reshaped.rank() != 2 || reshaped.dim(0) != 3 || reshaped.dim(1) != 2 ||
      reshaped.data() != matrix.data()) {
    std::fprintf(stderr, "Tensor reshape produced unexpected view\n");
    return 1;
  }

  if (!RunMixedPrecisionSmoke(ctx, dlcuda::DType::kFloat16) ||
      !RunMixedPrecisionSmoke(ctx, dlcuda::DType::kBFloat16)) {
    return 1;
  }

  if (!RunLayerCoverageSmoke(ctx)) {
    return 1;
  }

  if (!RunOptimizerCoverageSmoke(ctx)) {
    return 1;
  }

  if (!RunAutogradSmoke(ctx)) {
    return 1;
  }

  std::printf("gpu_correctness_tests: PASS\n");
  return 0;
}
