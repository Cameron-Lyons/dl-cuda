#include "dl_cuda.hpp"
#include "dl_cuda/detail/cuda_dtype.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

bool HasCudaDevice() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  return status == cudaSuccess && count > 0;
}

bool AlmostEqual(float actual, float expected, float tolerance = 1e-4f) {
  return std::fabs(actual - expected) <= tolerance;
}

float GELUValue(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

float GELUGrad(float x) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
  return 0.5f * (1.0f + std::erf(x * kInvSqrt2)) + x * std::exp(-0.5f * x * x) * kInvSqrt2Pi;
}

bool CheckCloseVector(const std::vector<float> &actual, const std::vector<float> &expected,
                      const char *label, float tolerance = 1e-4f) {
  if (actual.size() != expected.size()) {
    std::fprintf(stderr, "%s size mismatch\n", label);
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!AlmostEqual(actual[i], expected[i], tolerance)) {
      std::fprintf(stderr, "%s value mismatch at %zu: got %.6f expected %.6f\n", label, i,
                   actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

bool CopyFloatsToTensor(dlcuda::RuntimeContext &ctx, dlcuda::Tensor *tensor,
                        const std::vector<float> &values, const char *label) {
  if (tensor == nullptr || !tensor->defined() ||
      tensor->numel() != static_cast<int64_t>(values.size())) {
    std::fprintf(stderr, "%s copy received invalid tensor\n", label);
    return false;
  }
  dlcuda::Status status;
  switch (tensor->dtype()) {
  case dlcuda::DType::kFloat32:
    status = tensor->CopyFromHost(values.data(), values.size() * sizeof(float), ctx.stream());
    break;
  case dlcuda::DType::kFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = dlcuda::detail::FloatToFloat16Bits(values[i]);
    }
    status =
        tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), ctx.stream());
    break;
  }
  case dlcuda::DType::kBFloat16: {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = dlcuda::detail::FloatToBFloat16Bits(values[i]);
    }
    status =
        tensor->CopyFromHost(converted.data(), converted.size() * sizeof(uint16_t), ctx.stream());
    break;
  }
  case dlcuda::DType::kInt32:
    std::fprintf(stderr, "%s copy does not support int32\n", label);
    return false;
  }
  if (!status.ok()) {
    std::fprintf(stderr, "%s copy failed: %s\n", label, status.ToString().c_str());
    return false;
  }
  return true;
}

bool CopyTensorToFloats(dlcuda::RuntimeContext &ctx, const dlcuda::Tensor &tensor,
                        std::vector<float> *values, const char *label) {
  if (values == nullptr || !tensor.defined()) {
    std::fprintf(stderr, "%s read received invalid tensor\n", label);
    return false;
  }
  values->resize(static_cast<size_t>(tensor.numel()));
  dlcuda::Status status;
  switch (tensor.dtype()) {
  case dlcuda::DType::kFloat32:
    status = tensor.CopyToHost(values->data(), values->size() * sizeof(float), ctx.stream());
    break;
  case dlcuda::DType::kFloat16: {
    std::vector<uint16_t> raw(values->size());
    status = tensor.CopyToHost(raw.data(), raw.size() * sizeof(uint16_t), ctx.stream());
    if (status.ok()) {
      for (size_t i = 0; i < raw.size(); ++i) {
        (*values)[i] = dlcuda::detail::Float16BitsToFloat(raw[i]);
      }
    }
    break;
  }
  case dlcuda::DType::kBFloat16: {
    std::vector<uint16_t> raw(values->size());
    status = tensor.CopyToHost(raw.data(), raw.size() * sizeof(uint16_t), ctx.stream());
    if (status.ok()) {
      for (size_t i = 0; i < raw.size(); ++i) {
        (*values)[i] = dlcuda::detail::BFloat16BitsToFloat(raw[i]);
      }
    }
    break;
  }
  case dlcuda::DType::kInt32:
    std::fprintf(stderr, "%s read does not support int32\n", label);
    return false;
  }
  if (!status.ok() || !ctx.Synchronize().ok()) {
    std::fprintf(stderr, "%s read failed\n", label);
    return false;
  }
  return true;
}

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

bool RunAutogradSmoke(dlcuda::RuntimeContext &ctx) {
  auto x_result = dlcuda::Tensor::AllocateAsync({3}, dlcuda::DType::kFloat32, ctx.stream());
  if (!x_result.ok()) {
    std::fprintf(stderr, "Autograd tensor allocation failed\n");
    return false;
  }
  dlcuda::Tensor x = x_result.value();
  if (!CopyFloatsToTensor(ctx, &x, {1.0f, 2.0f, 3.0f}, "autograd x")) {
    return false;
  }

  dlcuda::GradientTape tape;
  dlcuda::AutoTensor ax = tape.Variable(x);
  auto squared = tape.Multiply(ctx, ax, ax);
  if (!squared.ok()) {
    std::fprintf(stderr, "Autograd multiply failed: %s\n", squared.status().ToString().c_str());
    return false;
  }
  auto shifted = tape.Add(ctx, squared.value(), ax);
  if (!shifted.ok()) {
    std::fprintf(stderr, "Autograd add failed: %s\n", shifted.status().ToString().c_str());
    return false;
  }
  auto loss = tape.ReduceSum(ctx, shifted.value());
  if (!loss.ok()) {
    std::fprintf(stderr, "Autograd reduce-sum failed: %s\n", loss.status().ToString().c_str());
    return false;
  }
  dlcuda::Status backward = loss.value().Backward(ctx);
  if (!backward.ok()) {
    std::fprintf(stderr, "Autograd backward failed: %s\n", backward.ToString().c_str());
    return false;
  }
  if (tape.node_count() != 3) {
    std::fprintf(stderr, "Autograd graph recorded unexpected node count\n");
    return false;
  }

  auto x_grad = ax.grad();
  std::vector<float> host_x_grad;
  if (!x_grad.ok() || !CopyTensorToFloats(ctx, x_grad.value(), &host_x_grad, "autograd x grad")) {
    std::fprintf(stderr, "Autograd x grad read failed\n");
    return false;
  }
  std::vector<float> expected_x_grad = {3.0f, 5.0f, 7.0f};
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    if (!AlmostEqual(host_x_grad[i], expected_x_grad[i])) {
      std::fprintf(stderr, "Autograd x grad mismatch at %zu\n", i);
      return false;
    }
  }

  auto matrix_result = dlcuda::Tensor::AllocateAsync({2, 3}, dlcuda::DType::kFloat32, ctx.stream());
  auto row_result = dlcuda::Tensor::AllocateAsync({3}, dlcuda::DType::kFloat32, ctx.stream());
  if (!matrix_result.ok() || !row_result.ok()) {
    std::fprintf(stderr, "Autograd broadcast allocation failed\n");
    return false;
  }
  dlcuda::Tensor matrix = matrix_result.value();
  dlcuda::Tensor row = row_result.value();
  if (!CopyFloatsToTensor(ctx, &matrix, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, "autograd matrix") ||
      !CopyFloatsToTensor(ctx, &row, {10.0f, 20.0f, 30.0f}, "autograd row")) {
    return false;
  }

  dlcuda::GradientTape broadcast_tape;
  dlcuda::AutoTensor amatrix = broadcast_tape.Variable(matrix);
  dlcuda::AutoTensor arow = broadcast_tape.Variable(row);
  auto added = broadcast_tape.Add(ctx, amatrix, arow);
  if (!added.ok()) {
    std::fprintf(stderr, "Autograd broadcast add failed: %s\n", added.status().ToString().c_str());
    return false;
  }
  auto broadcast_loss = broadcast_tape.ReduceSum(ctx, added.value());
  if (!broadcast_loss.ok() || !broadcast_loss.value().Backward(ctx).ok()) {
    std::fprintf(stderr, "Autograd broadcast backward failed\n");
    return false;
  }
  auto row_grad = arow.grad();
  std::vector<float> host_row_grad;
  if (!row_grad.ok() ||
      !CopyTensorToFloats(ctx, row_grad.value(), &host_row_grad, "autograd row grad")) {
    std::fprintf(stderr, "Autograd row grad read failed\n");
    return false;
  }
  std::vector<float> expected_row_grad = {2.0f, 2.0f, 2.0f};
  for (size_t i = 0; i < expected_row_grad.size(); ++i) {
    if (!AlmostEqual(host_row_grad[i], expected_row_grad[i])) {
      std::fprintf(stderr, "Autograd row grad mismatch at %zu\n", i);
      return false;
    }
  }

  auto lhs_result = dlcuda::Tensor::AllocateAsync({2, 2}, dlcuda::DType::kFloat32, ctx.stream());
  auto rhs_result = dlcuda::Tensor::AllocateAsync({2, 1}, dlcuda::DType::kFloat32, ctx.stream());
  if (!lhs_result.ok() || !rhs_result.ok()) {
    std::fprintf(stderr, "Autograd matmul allocation failed\n");
    return false;
  }
  dlcuda::Tensor lhs = lhs_result.value();
  dlcuda::Tensor rhs = rhs_result.value();
  if (!CopyFloatsToTensor(ctx, &lhs, {1.0f, 2.0f, 3.0f, 4.0f}, "autograd lhs") ||
      !CopyFloatsToTensor(ctx, &rhs, {5.0f, 6.0f}, "autograd rhs")) {
    return false;
  }

  dlcuda::GradientTape matmul_tape;
  dlcuda::AutoTensor alhs = matmul_tape.Variable(lhs);
  dlcuda::AutoTensor arhs = matmul_tape.Variable(rhs);
  auto matmul = matmul_tape.MatMul(ctx, alhs, arhs);
  if (!matmul.ok()) {
    std::fprintf(stderr, "Autograd matmul failed: %s\n", matmul.status().ToString().c_str());
    return false;
  }
  auto matmul_loss = matmul_tape.ReduceSum(ctx, matmul.value());
  if (!matmul_loss.ok() || !matmul_loss.value().Backward(ctx).ok()) {
    std::fprintf(stderr, "Autograd matmul backward failed\n");
    return false;
  }
  auto lhs_grad = alhs.grad();
  auto rhs_grad = arhs.grad();
  std::vector<float> host_lhs_grad;
  std::vector<float> host_rhs_grad;
  if (!lhs_grad.ok() ||
      !CopyTensorToFloats(ctx, lhs_grad.value(), &host_lhs_grad, "autograd lhs grad") ||
      !rhs_grad.ok() ||
      !CopyTensorToFloats(ctx, rhs_grad.value(), &host_rhs_grad, "autograd rhs grad")) {
    std::fprintf(stderr, "Autograd matmul grad read failed\n");
    return false;
  }
  std::vector<float> expected_lhs_grad = {5.0f, 6.0f, 5.0f, 6.0f};
  std::vector<float> expected_rhs_grad = {4.0f, 6.0f};
  for (size_t i = 0; i < expected_lhs_grad.size(); ++i) {
    if (!AlmostEqual(host_lhs_grad[i], expected_lhs_grad[i])) {
      std::fprintf(stderr, "Autograd lhs grad mismatch at %zu\n", i);
      return false;
    }
  }
  for (size_t i = 0; i < expected_rhs_grad.size(); ++i) {
    if (!AlmostEqual(host_rhs_grad[i], expected_rhs_grad[i])) {
      std::fprintf(stderr, "Autograd rhs grad mismatch at %zu\n", i);
      return false;
    }
  }

  dlcuda::GradientTape custom_tape;
  dlcuda::Status register_status = custom_tape.RegisterCustomOp(
      "square",
      [](dlcuda::RuntimeContext &ctx, const std::vector<dlcuda::Tensor> &inputs,
         dlcuda::Tensor *output) {
        if (inputs.size() != 1) {
          return dlcuda::Status::InvalidArgument("square expects one input");
        }
        return dlcuda::TensorMultiply(ctx, inputs[0], inputs[0], output);
      },
      [](dlcuda::RuntimeContext &ctx, const dlcuda::Tensor &output_grad,
         const std::vector<dlcuda::Tensor> &inputs, const dlcuda::Tensor &output,
         std::vector<dlcuda::Tensor> *input_grads) {
        (void)output;
        if (input_grads == nullptr || inputs.size() != 1) {
          return dlcuda::Status::InvalidArgument("square backward expects one input");
        }
        dlcuda::Tensor temp;
        DLCUDA_RETURN_IF_ERROR(dlcuda::TensorMultiply(ctx, output_grad, inputs[0], &temp));
        dlcuda::Tensor doubled;
        DLCUDA_RETURN_IF_ERROR(dlcuda::TensorAdd(ctx, temp, temp, &doubled));
        input_grads->assign(1, doubled);
        return dlcuda::Status::Ok();
      });
  if (!register_status.ok()) {
    std::fprintf(stderr, "Autograd custom op registration failed: %s\n",
                 register_status.ToString().c_str());
    return false;
  }

  dlcuda::AutoTensor custom_x = custom_tape.Variable(x);
  auto custom_square = custom_tape.ApplyCustomOp(ctx, "square", {custom_x});
  if (!custom_square.ok()) {
    std::fprintf(stderr, "Autograd custom op failed: %s\n",
                 custom_square.status().ToString().c_str());
    return false;
  }
  auto custom_loss = custom_tape.ReduceSum(ctx, custom_square.value());
  if (!custom_loss.ok() || !custom_loss.value().Backward(ctx).ok()) {
    std::fprintf(stderr, "Autograd custom backward failed\n");
    return false;
  }
  auto custom_grad = custom_x.grad();
  std::vector<float> host_custom_grad;
  if (!custom_grad.ok() ||
      !CopyTensorToFloats(ctx, custom_grad.value(), &host_custom_grad, "autograd custom grad")) {
    std::fprintf(stderr, "Autograd custom grad read failed\n");
    return false;
  }
  std::vector<float> expected_custom_grad = {2.0f, 4.0f, 6.0f};
  for (size_t i = 0; i < expected_custom_grad.size(); ++i) {
    if (!AlmostEqual(host_custom_grad[i], expected_custom_grad[i])) {
      std::fprintf(stderr, "Autograd custom grad mismatch at %zu\n", i);
      return false;
    }
  }

  auto relu_input_result =
      dlcuda::Tensor::AllocateAsync({3}, dlcuda::DType::kFloat32, ctx.stream());
  if (!relu_input_result.ok()) {
    std::fprintf(stderr, "Autograd module input allocation failed\n");
    return false;
  }
  dlcuda::Tensor relu_input = relu_input_result.value();
  if (!CopyFloatsToTensor(ctx, &relu_input, {-1.0f, 2.0f, 3.0f}, "autograd module input")) {
    return false;
  }

  dlcuda::GradientTape module_tape;
  dlcuda::AutoTensor module_input = module_tape.Variable(relu_input);
  dlcuda::ReLU relu_module;
  auto module_output = module_tape.ApplyModule(ctx, relu_module, module_input);
  if (!module_output.ok()) {
    std::fprintf(stderr, "Autograd module forward failed: %s\n",
                 module_output.status().ToString().c_str());
    return false;
  }
  auto module_loss = module_tape.ReduceSum(ctx, module_output.value());
  if (!module_loss.ok() || !module_loss.value().Backward(ctx).ok()) {
    std::fprintf(stderr, "Autograd module backward failed\n");
    return false;
  }
  auto module_grad = module_input.grad();
  std::vector<float> host_module_grad;
  if (!module_grad.ok() ||
      !CopyTensorToFloats(ctx, module_grad.value(), &host_module_grad, "autograd module grad")) {
    std::fprintf(stderr, "Autograd module grad read failed\n");
    return false;
  }
  std::vector<float> expected_module_grad = {0.0f, 1.0f, 1.0f};
  for (size_t i = 0; i < expected_module_grad.size(); ++i) {
    if (!AlmostEqual(host_module_grad[i], expected_module_grad[i])) {
      std::fprintf(stderr, "Autograd module grad mismatch at %zu\n", i);
      return false;
    }
  }

  return true;
}

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

bool RunOptimizerCoverageSmoke(dlcuda::RuntimeContext &ctx) {
  auto make_param = [&](const char *label, const std::vector<float> &values,
                        const std::vector<float> &grads, dlcuda::Tensor *param,
                        dlcuda::Tensor *grad) -> bool {
    auto param_result = dlcuda::Tensor::AllocateAsync({static_cast<int64_t>(values.size())},
                                                      dlcuda::DType::kFloat32, ctx.stream());
    auto grad_result = dlcuda::Tensor::AllocateAsync({static_cast<int64_t>(grads.size())},
                                                     dlcuda::DType::kFloat32, ctx.stream());
    if (!param_result.ok() || !grad_result.ok()) {
      std::fprintf(stderr, "%s optimizer tensor allocation failed\n", label);
      return false;
    }
    *param = param_result.value();
    *grad = grad_result.value();
    return CopyFloatsToTensor(ctx, param, values, label) &&
           CopyFloatsToTensor(ctx, grad, grads, label);
  };

  std::vector<float> host_values;

  dlcuda::Tensor sgd_param;
  dlcuda::Tensor sgd_grad;
  if (!make_param("sgd", {1.0f}, {0.5f}, &sgd_param, &sgd_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> sgd_params = {{"sgd.weight", &sgd_param, &sgd_grad}};
  dlcuda::SGDOptimizer sgd(0.1f, 0.9f);
  if (!sgd.Step(ctx, sgd_params).ok() || !sgd.Step(ctx, sgd_params).ok() ||
      !CopyTensorToFloats(ctx, sgd_param, &host_values, "sgd param") ||
      !CheckCloseVector(host_values, {0.855f}, "SGD momentum")) {
    return false;
  }

  dlcuda::Tensor wd_param;
  dlcuda::Tensor wd_grad;
  if (!make_param("sgd weight decay", {1.0f, -2.0f}, {0.5f, -0.25f}, &wd_param, &wd_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> wd_params = {{"wd.weight", &wd_param, &wd_grad}};
  dlcuda::SGDOptimizer sgd_wd(0.1f, 0.0f, 0.1f);
  if (!sgd_wd.Step(ctx, wd_params).ok() ||
      !CopyTensorToFloats(ctx, wd_param, &host_values, "sgd weight decay param") ||
      !CheckCloseVector(host_values, {0.94f, -1.955f}, "SGD weight decay")) {
    return false;
  }

  dlcuda::Tensor adamw_param;
  dlcuda::Tensor adamw_grad;
  if (!make_param("adamw", {1.0f}, {0.5f}, &adamw_param, &adamw_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> adamw_params = {{"adamw.weight", &adamw_param, &adamw_grad}};
  dlcuda::AdamWOptimizer adamw(0.1f, 0.01f);
  if (!adamw.Step(ctx, adamw_params).ok() ||
      !CopyTensorToFloats(ctx, adamw_param, &host_values, "adamw param") ||
      !CheckCloseVector(host_values, {0.899f}, "AdamW decoupled weight decay", 2e-4f)) {
    return false;
  }

  dlcuda::Tensor rms_param;
  dlcuda::Tensor rms_grad;
  if (!make_param("rmsprop", {1.0f}, {0.5f}, &rms_param, &rms_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> rms_params = {{"rms.weight", &rms_param, &rms_grad}};
  dlcuda::RMSPropOptimizer rmsprop(0.1f, 0.0f);
  if (!rmsprop.Step(ctx, rms_params).ok() ||
      !CopyTensorToFloats(ctx, rms_param, &host_values, "rmsprop param") ||
      !CheckCloseVector(host_values, {0.9f}, "RMSProp update", 2e-4f)) {
    return false;
  }

  dlcuda::Tensor group_a;
  dlcuda::Tensor group_a_grad;
  dlcuda::Tensor group_b;
  dlcuda::Tensor group_b_grad;
  if (!make_param("group a", {1.0f}, {1.0f}, &group_a, &group_a_grad) ||
      !make_param("group b", {1.0f}, {1.0f}, &group_b, &group_b_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> group_params = {{"group.a", &group_a, &group_a_grad},
                                                    {"group.b", &group_b, &group_b_grad}};
  dlcuda::SGDOptimizer grouped(std::vector<dlcuda::OptimizerParamGroup>{
      {{"group.a"}, 0.1f, 0.0f}, {{"group.b"}, 0.01f, 0.0f}});
  dlcuda::StepLRScheduler step_lr(1, 0.5f);
  if (!grouped.Step(ctx, group_params, step_lr).ok() ||
      !grouped.Step(ctx, group_params, step_lr).ok() ||
      !CopyTensorToFloats(ctx, group_a, &host_values, "group a param") ||
      !CheckCloseVector(host_values, {0.85f}, "parameter group a scheduler") ||
      !CopyTensorToFloats(ctx, group_b, &host_values, "group b param") ||
      !CheckCloseVector(host_values, {0.985f}, "parameter group b scheduler")) {
    return false;
  }
  dlcuda::ConstantLRScheduler constant_lr;
  dlcuda::ExponentialLRScheduler exponential_lr(0.5f);
  dlcuda::CosineAnnealingLRScheduler cosine_lr(4, 0.1f);
  auto constant_rate = constant_lr.LearningRate(7, 0.25f);
  auto exponential_rate = exponential_lr.LearningRate(3, 0.8f);
  auto cosine_start = cosine_lr.LearningRate(0, 1.0f);
  auto cosine_mid = cosine_lr.LearningRate(2, 1.0f);
  auto cosine_end = cosine_lr.LearningRate(8, 1.0f);
  if (!constant_rate.ok() || !exponential_rate.ok() || !cosine_start.ok() || !cosine_mid.ok() ||
      !cosine_end.ok() || !AlmostEqual(constant_rate.value(), 0.25f) ||
      !AlmostEqual(exponential_rate.value(), 0.1f) || !AlmostEqual(cosine_start.value(), 1.0f) ||
      !AlmostEqual(cosine_mid.value(), 0.55f) || !AlmostEqual(cosine_end.value(), 0.1f)) {
    return false;
  }

  dlcuda::Tensor checkpoint_param;
  dlcuda::Tensor checkpoint_grad;
  dlcuda::Tensor restored_param;
  dlcuda::Tensor restored_grad;
  if (!make_param("checkpoint", {1.0f}, {0.25f}, &checkpoint_param, &checkpoint_grad) ||
      !make_param("checkpoint restore", {1.0f}, {0.25f}, &restored_param, &restored_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> checkpoint_params = {
      {"checkpoint.weight", &checkpoint_param, &checkpoint_grad}};
  std::vector<dlcuda::ParameterRef> restored_params = {
      {"checkpoint.weight", &restored_param, &restored_grad}};
  dlcuda::AdamOptimizer checkpoint_adam;
  const char *checkpoint_path = "/tmp/dlcuda_optimizer_smoke.ckpt";
  if (!checkpoint_adam.Step(ctx, checkpoint_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, checkpoint_param, &host_values, "checkpoint after first step") ||
      !CopyFloatsToTensor(ctx, &restored_param, host_values, "checkpoint restore param") ||
      !checkpoint_adam.SaveCheckpoint(ctx, checkpoint_path, checkpoint_params).ok() ||
      !checkpoint_adam.Step(ctx, checkpoint_params, 0.1f).ok()) {
    std::fprintf(stderr, "Optimizer checkpoint save path failed\n");
    std::remove(checkpoint_path);
    return false;
  }
  dlcuda::AdamOptimizer restored_adam;
  dlcuda::AdamOptimizer mismatched_adam(0.8f, 0.999f, 1e-8f);
  if (mismatched_adam.LoadCheckpoint(ctx, checkpoint_path, restored_params).ok()) {
    std::fprintf(stderr, "Optimizer checkpoint should reject mismatched hyperparameters\n");
    std::remove(checkpoint_path);
    return false;
  }
  if (!restored_adam.LoadCheckpoint(ctx, checkpoint_path, restored_params).ok() ||
      restored_adam.step_count() != 1 || !restored_adam.Step(ctx, restored_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, checkpoint_param, &host_values, "checkpoint continuous param")) {
    std::fprintf(stderr, "Optimizer checkpoint restore path failed\n");
    std::remove(checkpoint_path);
    return false;
  }
  std::vector<float> restored_values;
  if (!CopyTensorToFloats(ctx, restored_param, &restored_values, "checkpoint restored param") ||
      !CheckCloseVector(restored_values, host_values, "Optimizer checkpoint restore", 2e-4f)) {
    std::remove(checkpoint_path);
    return false;
  }
  std::remove(checkpoint_path);

  dlcuda::Tensor full_param;
  dlcuda::Tensor full_grad;
  dlcuda::Tensor full_restored_param;
  dlcuda::Tensor full_restored_grad;
  if (!make_param("full checkpoint", {1.0f}, {0.25f}, &full_param, &full_grad) ||
      !make_param("full checkpoint restore", {1.0f}, {0.25f}, &full_restored_param,
                  &full_restored_grad)) {
    return false;
  }
  std::vector<dlcuda::ParameterRef> full_params = {{"full.weight", &full_param, &full_grad}};
  std::vector<dlcuda::ParameterRef> full_restored_params = {
      {"full.weight", &full_restored_param, &full_restored_grad}};
  dlcuda::AdamOptimizer full_adam;
  const char *full_checkpoint_path = "/tmp/dlcuda_full_smoke.ckpt";
  dlcuda::CheckpointMetadata full_metadata;
  full_metadata.model_name = "full-smoke";
  full_metadata.epoch = 7;
  full_metadata.training_config = {{"lr", "0.1"}};
  full_metadata.rng_states = {{"offset_rng", "1234"}};
  if (!full_adam.Step(ctx, full_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, full_param, &host_values, "full checkpoint after first step") ||
      !CopyFloatsToTensor(ctx, &full_restored_param, host_values,
                          "full checkpoint restore param")) {
    std::fprintf(stderr, "Full checkpoint setup failed\n");
    std::remove(full_checkpoint_path);
    return false;
  }
  full_metadata.step = full_adam.step_count();
  if (!dlcuda::SaveCheckpoint(ctx, full_checkpoint_path, full_metadata, full_params, &full_adam)
           .ok() ||
      !full_adam.Step(ctx, full_params, 0.1f).ok()) {
    std::fprintf(stderr, "Full checkpoint save path failed\n");
    std::remove(full_checkpoint_path);
    return false;
  }
  dlcuda::AdamOptimizer full_restored_adam;
  dlcuda::CheckpointMetadata loaded_full_metadata;
  if (!dlcuda::LoadCheckpoint(ctx, full_checkpoint_path, "full-smoke", full_restored_params,
                              &full_restored_adam, &loaded_full_metadata)
           .ok() ||
      loaded_full_metadata.epoch != 7 || loaded_full_metadata.step != 1 ||
      loaded_full_metadata.training_config.size() != 1 ||
      loaded_full_metadata.training_config[0].key != "lr" ||
      loaded_full_metadata.training_config[0].value != "0.1" ||
      loaded_full_metadata.rng_states.size() != 1 || full_restored_adam.step_count() != 1 ||
      !full_restored_adam.Step(ctx, full_restored_params, 0.1f).ok() ||
      !CopyTensorToFloats(ctx, full_param, &host_values, "full checkpoint continuous param")) {
    std::fprintf(stderr, "Full checkpoint restore path failed\n");
    std::remove(full_checkpoint_path);
    return false;
  }
  if (!CopyTensorToFloats(ctx, full_restored_param, &restored_values,
                          "full checkpoint restored param") ||
      !CheckCloseVector(restored_values, host_values, "Full checkpoint restore", 2e-4f)) {
    std::remove(full_checkpoint_path);
    return false;
  }
  std::remove(full_checkpoint_path);

  return true;
}

} // namespace

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
