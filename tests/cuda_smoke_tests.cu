#include "dl_cuda.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

bool HasCudaDevice() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  return status == cudaSuccess && count > 0;
}

} // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("cuda_smoke_tests: SKIP (no CUDA device)\n");
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

  std::printf("cuda_smoke_tests: PASS\n");
  return 0;
}
