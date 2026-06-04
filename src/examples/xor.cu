#include "detail/common.cuh"

#include "dl_cuda/trainer.hpp"

namespace dlcuda {

Status TrainXor(const TrainXorConfig &cfg) {
  DLCUDA_RETURN_IF_ERROR(ValidateXorConfig(cfg));

  RuntimeContext ctx(OptionsFromXorConfig(cfg));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<Linear>(2, cfg.hidden_size, ctx)));
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<ReLU>()));
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<Linear>(cfg.hidden_size, 1, ctx)));
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<Sigmoid>()));
  const auto &params = model.parameters();

  AdamOptimizer optimizer;

  std::vector<float> host_x = {
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
  };
  std::vector<float> host_y = {0.0f, 1.0f, 1.0f, 0.0f};

  auto x_tensor = Tensor::AllocateAsync({4, 2}, DType::kFloat32, ctx.stream());
  if (!x_tensor.ok()) {
    return x_tensor.status();
  }
  auto y_tensor = Tensor::AllocateAsync({4, 1}, DType::kFloat32, ctx.stream());
  if (!y_tensor.ok()) {
    return y_tensor.status();
  }

  Tensor x = x_tensor.value();
  Tensor y = y_tensor.value();
  DLCUDA_RETURN_IF_ERROR(
      x.CopyFromHost(host_x.data(), host_x.size() * sizeof(float), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(
      y.CopyFromHost(host_y.data(), host_y.size() * sizeof(float), ctx.stream()));

  int64_t start_epoch = 0;
  if (cfg.resume) {
    CheckpointMetadata checkpoint_metadata;
    Status load_status = LoadCheckpoint(ctx, cfg.checkpoint_path, "xor-mlp", params, &optimizer,
                                        &checkpoint_metadata);
    if (!load_status.ok()) {
      return Status::RuntimeError("Failed to resume XOR checkpoint: " + load_status.message());
    }
    DLCUDA_RETURN_IF_ERROR(ValidateXorCheckpointMetadata(checkpoint_metadata, cfg));
    DLCUDA_RETURN_IF_ERROR(ValidateOptimizerStepMetadata(checkpoint_metadata, optimizer));
    start_epoch = checkpoint_metadata.epoch;
    std::printf("Loaded checkpoint: %s (epoch=%lld step=%lld)\n", cfg.checkpoint_path.c_str(),
                static_cast<long long>(checkpoint_metadata.epoch),
                static_cast<long long>(checkpoint_metadata.step));
  }

  auto end_epoch_result = ComputeEndEpoch(start_epoch, cfg.epochs);
  if (!end_epoch_result.ok()) {
    return end_epoch_result.status();
  }
  int64_t end_epoch = end_epoch_result.value();

  std::printf("XOR | epochs=%d lr=%.4f hidden=%d | backend=%s | TF32=%s\n", cfg.epochs, cfg.lr,
              cfg.hidden_size, cfg.use_cublas ? "cuBLAS" : "kernels", cfg.tf32 ? "on" : "off");

  SupervisedTrainer trainer(ctx, model, optimizer, params);
  TrainStepOptions train_options;
  train_options.learning_rate = cfg.lr;
  train_options.max_grad_norm = cfg.grad_clip;
  for (int64_t epoch = start_epoch; epoch < end_epoch; ++epoch) {
    bool should_log = ((epoch - start_epoch) % cfg.print_every) == 0;
    train_options.compute_metrics = should_log;

    TrainStepResult step_result;
    DLCUDA_RETURN_IF_ERROR(
        trainer.TrainBinaryClassificationStep(x, y, train_options, &step_result));

    if (should_log) {
      std::printf("Epoch %4lld | BCE: %.6f | GradNorm: %.4f\n", static_cast<long long>(epoch),
                  step_result.loss, step_result.grad_norm);
    }
  }

  Tensor final_predictions;
  DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, x, &final_predictions));

  std::vector<float> host_pred(4);
  DLCUDA_RETURN_IF_ERROR(final_predictions.CopyToHost(
      host_pred.data(), host_pred.size() * sizeof(float), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

  std::printf("Final predictions:\n");
  std::printf("  [0, 0] -> %.4f (expected 0)\n", host_pred[0]);
  std::printf("  [0, 1] -> %.4f (expected 1)\n", host_pred[1]);
  std::printf("  [1, 0] -> %.4f (expected 1)\n", host_pred[2]);
  std::printf("  [1, 1] -> %.4f (expected 0)\n", host_pred[3]);

  if (cfg.save) {
    CheckpointMetadata metadata =
        BuildXorCheckpointMetadata(cfg, end_epoch, optimizer.step_count());
    DLCUDA_RETURN_IF_ERROR(SaveCheckpoint(ctx, cfg.checkpoint_path, metadata, params, &optimizer));
    std::printf("Saved checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  return Status::Ok();
}

} // namespace dlcuda
