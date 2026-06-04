#include "common.hpp"

namespace dlcuda::gpu_tests {

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

} // namespace dlcuda::gpu_tests
