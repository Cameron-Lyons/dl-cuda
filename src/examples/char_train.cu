#include "detail/cuda_graph.cuh"
#include "detail/char_training.cuh"
#include "detail/char_sampling.cuh"

namespace dlcuda {

Status TrainChar(const TrainCharConfig &cfg) {
  DLCUDA_RETURN_IF_ERROR(ValidateCharConfig(cfg));

  auto corpus_result = LoadCharCorpus(cfg.data_path);
  if (!corpus_result.ok()) {
    return corpus_result.status();
  }
  std::string corpus = corpus_result.value();
  auto vocab_result = CharVocab::Build(corpus);
  if (!vocab_result.ok()) {
    return vocab_result.status();
  }
  CharVocab vocab = vocab_result.value();

  DLCUDA_RETURN_IF_ERROR(ValidateCorpusWindow(corpus.size(), cfg.seq_len, "Training"));
  auto splits_result = BuildCharDatasetSplits(corpus.size(), cfg);
  if (!splits_result.ok()) {
    return splits_result.status();
  }
  CharDatasetSplits splits = splits_result.value();

  RuntimeContext ctx(OptionsFromCharConfig(cfg.use_cublas, cfg.tf32, cfg.seed));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(BuildCharModel(&model, ctx, vocab.size(), cfg.d_model));
  const auto &params = model.parameters();

  AdamOptimizer optimizer;

  auto input_ids_result = Tensor::AllocateAsync({cfg.seq_len}, DType::kInt32, ctx.stream());
  if (!input_ids_result.ok()) {
    return input_ids_result.status();
  }
  auto target_ids_result = Tensor::AllocateAsync({cfg.seq_len}, DType::kInt32, ctx.stream());
  if (!target_ids_result.ok()) {
    return target_ids_result.status();
  }

  Tensor input_ids = input_ids_result.value();
  Tensor target_ids = target_ids_result.value();

  std::mt19937 offset_rng(static_cast<uint32_t>(cfg.seed));
  int64_t start_epoch = 0;
  if (cfg.resume) {
    CheckpointMetadata checkpoint_metadata;
    Status load_status = LoadCheckpoint(ctx, cfg.checkpoint_path, kCharModelName, params,
                                        &optimizer, &checkpoint_metadata);
    if (!load_status.ok()) {
      return Status::RuntimeError("Failed to resume char checkpoint: " + load_status.message());
    }
    DLCUDA_RETURN_IF_ERROR(ValidateCharCheckpointMetadata(checkpoint_metadata, cfg.seq_len,
                                                          cfg.d_model, corpus, vocab));
    DLCUDA_RETURN_IF_ERROR(ValidateOptimizerStepMetadata(checkpoint_metadata, optimizer));
    DLCUDA_RETURN_IF_ERROR(
        RestoreMt19937State(checkpoint_metadata, "char.offset_rng", &offset_rng));
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

  std::printf("Char | model=%s context=%d vocab=%d seq_len=%d d_model=%d epochs=%d\n",
              kCharModelName, kCharContextWidth, vocab.size(), cfg.seq_len, cfg.d_model,
              cfg.epochs);
  if (!cfg.data_path.empty()) {
    std::printf("Data: %s (%zu bytes)\n", cfg.data_path.c_str(), corpus.size());
  }
  std::printf("Splits: train=%lld val=%lld test=%lld windows | val_every=%d\n",
              static_cast<long long>(splits.train.count),
              static_cast<long long>(splits.validation.count),
              static_cast<long long>(splits.test.count), cfg.val_every);
  std::printf("Optimizer: Adam | Grad clip: %.2f | temp=%.2f top_p=%.2f\n", cfg.grad_clip,
              cfg.temperature, cfg.top_p);

  auto train_start = std::chrono::steady_clock::now();

  std::vector<int32_t> encoded_corpus(corpus.size());
  for (size_t i = 0; i < corpus.size(); ++i) {
    encoded_corpus[i] = vocab.Encode(corpus[i]);
  }
  auto encoded_corpus_tensor = Tensor::AllocateAsync({static_cast<int64_t>(encoded_corpus.size())},
                                                     DType::kInt32, ctx.stream());
  if (!encoded_corpus_tensor.ok()) {
    return encoded_corpus_tensor.status();
  }
  Tensor encoded_corpus_device = encoded_corpus_tensor.value();
  DLCUDA_RETURN_IF_ERROR(encoded_corpus_device.CopyFromHost(
      encoded_corpus.data(), encoded_corpus.size() * sizeof(int32_t), ctx.stream()));

  std::uniform_int_distribution<int64_t> train_offset_dist(
      splits.train.begin, splits.train.begin + splits.train.count - 1);
  std::vector<MetricSnapshot> metric_history;
  CharRunSummary run_summary;
  std::string best_checkpoint_path = BestCheckpointPath(cfg);
  if (cfg.save && !best_checkpoint_path.empty() && best_checkpoint_path == cfg.checkpoint_path) {
    return Status::InvalidArgument("best_checkpoint_path must differ from checkpoint_path");
  }
  int no_improvement_evals = 0;
  int64_t completed_epoch = start_epoch;
  Tensor logits;
  Tensor loss_grad;
#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
  CudaGraphExec train_graph;
  bool graph_capture_disabled = false;
#endif

  for (int64_t epoch = start_epoch; epoch < end_epoch; ++epoch) {
    int64_t offset = train_offset_dist(offset_rng);
    DLCUDA_RETURN_IF_ERROR(FillTrainingWindow(ctx, encoded_corpus_device, &input_ids, &target_ids,
                                              cfg.seq_len, offset));

    bool should_log = ((epoch - start_epoch) % cfg.print_every) == 0;
    bool should_validate = splits.validation.count > 0 &&
                           (((epoch - start_epoch) % cfg.val_every) == 0 || epoch + 1 == end_epoch);
    bool need_train_metrics = should_log || should_validate;
    ClassificationMetrics metrics;
    float grad_norm = 0.0f;

#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
    bool ran_graph = false;
    if (!need_train_metrics && !graph_capture_disabled) {
      if (!train_graph.ready()) {
        Status capture_status = train_graph.Capture(ctx.stream(), [&]() {
          return RunCharTrainBody(ctx, model, optimizer, params, input_ids, target_ids, &logits,
                                  &loss_grad, cfg.grad_clip, nullptr, nullptr);
        });
        if (!capture_status.ok()) {
          graph_capture_disabled = true;
          train_graph.Reset();
        }
      }

      if (train_graph.ready()) {
        Status launch_status = train_graph.Launch(ctx.stream());
        if (launch_status.ok()) {
          ran_graph = true;
        } else {
          graph_capture_disabled = true;
          train_graph.Reset();
        }
      }
    }
#endif

#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
    if (!ran_graph)
#endif
    {
      DLCUDA_RETURN_IF_ERROR(RunCharTrainBody(
          ctx, model, optimizer, params, input_ids, target_ids, &logits, &loss_grad, cfg.grad_clip,
          need_train_metrics ? &metrics : nullptr, need_train_metrics ? &grad_norm : nullptr));
    }
    DLCUDA_RETURN_IF_ERROR(optimizer.Step(ctx, params, cfg.lr));
    completed_epoch = epoch + 1;

    bool has_validation_metrics = false;
    EvaluationSummary validation_summary;
    bool validation_improved = false;
    if (should_validate) {
      auto validation_result =
          EvaluateCharSplit(ctx, model, encoded_corpus_device, &input_ids, &target_ids, cfg.seq_len,
                            splits.validation, cfg.val_windows);
      if (!validation_result.ok()) {
        return validation_result.status();
      }
      validation_summary = validation_result.value();
      has_validation_metrics = true;
      validation_improved =
          !run_summary.has_best ||
          validation_summary.metrics.loss + cfg.min_delta < run_summary.best_val_loss;
      if (validation_improved) {
        run_summary.has_best = true;
        run_summary.best_epoch = completed_epoch;
        run_summary.best_val_loss = validation_summary.metrics.loss;
        no_improvement_evals = 0;
      } else {
        ++no_improvement_evals;
      }
    }

    if (need_train_metrics) {
      MetricSnapshot snapshot;
      snapshot.epoch = completed_epoch;
      snapshot.train = metrics;
      snapshot.grad_norm = grad_norm;
      snapshot.has_validation = has_validation_metrics;
      if (has_validation_metrics) {
        snapshot.validation = validation_summary.metrics;
      }
      metric_history.push_back(snapshot);
    }

    if (validation_improved && cfg.save && !best_checkpoint_path.empty()) {
      CheckpointMetadata metadata =
          BuildCharCheckpointMetadata(cfg, corpus, vocab, offset_rng, completed_epoch,
                                      optimizer.step_count(), splits, metric_history, run_summary);
      DLCUDA_RETURN_IF_ERROR(
          SaveCheckpoint(ctx, best_checkpoint_path, metadata, params, &optimizer));
      std::printf("Saved best checkpoint: %s (epoch=%lld val_loss=%.4f)\n",
                  best_checkpoint_path.c_str(), static_cast<long long>(completed_epoch),
                  validation_summary.metrics.loss);
    }

    if (should_log) {
      float ppl = std::exp(metrics.loss);
      float acc = metrics.accuracy * 100.0f;
      std::printf("Epoch %4lld | Train Loss: %.4f | PPL: %7.2f | Acc: %5.1f%% | "
                  "GradNorm: %.4f",
                  static_cast<long long>(completed_epoch), metrics.loss, ppl, acc, grad_norm);
      if (has_validation_metrics) {
        std::printf(" | Val Loss: %.4f | Val PPL: %7.2f | Val Acc: %5.1f%%",
                    validation_summary.metrics.loss, std::exp(validation_summary.metrics.loss),
                    validation_summary.metrics.accuracy * 100.0f);
      }
      std::printf("\n");
    } else if (has_validation_metrics) {
      std::printf("Epoch %4lld | Val Loss: %.4f | Val PPL: %7.2f | Val Acc: %5.1f%%\n",
                  static_cast<long long>(completed_epoch), validation_summary.metrics.loss,
                  std::exp(validation_summary.metrics.loss),
                  validation_summary.metrics.accuracy * 100.0f);
    }

    if (cfg.early_stop_patience > 0 && has_validation_metrics &&
        no_improvement_evals >= cfg.early_stop_patience) {
      run_summary.early_stopped = true;
      run_summary.stopped_epoch = completed_epoch;
      std::printf("Early stopping at epoch %lld after %d validation checks without improvement\n",
                  static_cast<long long>(completed_epoch), no_improvement_evals);
      break;
    }
  }

  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

  auto train_end = std::chrono::steady_clock::now();
  int64_t trained_epochs = completed_epoch - start_epoch;
  if (trained_epochs > 0) {
    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(train_end - train_start).count();
    double tokens = static_cast<double>(trained_epochs) * cfg.seq_len;
    double tok_per_sec = sec > 0.0 ? tokens / sec : 0.0;
    std::printf("Training throughput: %.2f tokens/s (%.3f s)\n", tok_per_sec, sec);
  }

  if (cfg.save) {
    CheckpointMetadata metadata =
        BuildCharCheckpointMetadata(cfg, corpus, vocab, offset_rng, completed_epoch,
                                    optimizer.step_count(), splits, metric_history, run_summary);
    DLCUDA_RETURN_IF_ERROR(SaveCheckpoint(ctx, cfg.checkpoint_path, metadata, params, &optimizer));
    std::printf("Saved checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  bool selected_best_checkpoint = false;
  int64_t selected_checkpoint_epoch = completed_epoch;
  if (run_summary.has_best && cfg.save && !best_checkpoint_path.empty()) {
    CheckpointMetadata best_metadata;
    DLCUDA_RETURN_IF_ERROR(LoadCheckpoint(ctx, best_checkpoint_path, kCharModelName, params,
                                          &optimizer, &best_metadata));
    DLCUDA_RETURN_IF_ERROR(
        ValidateCharCheckpointMetadata(best_metadata, cfg.seq_len, cfg.d_model, corpus, vocab));
    DLCUDA_RETURN_IF_ERROR(ValidateOptimizerStepMetadata(best_metadata, optimizer));
    DLCUDA_RETURN_IF_ERROR(RestoreMt19937State(best_metadata, "char.offset_rng", &offset_rng));
    selected_checkpoint_epoch = best_metadata.epoch;
    selected_best_checkpoint = true;
    std::printf("Selected best checkpoint for test/generation: %s (epoch=%lld val_loss=%.4f)\n",
                best_checkpoint_path.c_str(), static_cast<long long>(run_summary.best_epoch),
                run_summary.best_val_loss);
  } else if (run_summary.has_best) {
    std::printf("Best validation epoch: %lld val_loss=%.4f\n",
                static_cast<long long>(run_summary.best_epoch), run_summary.best_val_loss);
  }

  if (splits.test.count > 0) {
    auto test_result = EvaluateCharSplit(ctx, model, encoded_corpus_device, &input_ids, &target_ids,
                                         cfg.seq_len, splits.test, cfg.test_windows);
    if (!test_result.ok()) {
      return test_result.status();
    }
    run_summary.has_test = true;
    run_summary.test = test_result.value();
    std::printf("Test | windows=%lld | Loss: %.4f | PPL: %7.2f | Acc: %5.1f%%\n",
                static_cast<long long>(run_summary.test.windows), run_summary.test.metrics.loss,
                std::exp(run_summary.test.metrics.loss),
                run_summary.test.metrics.accuracy * 100.0f);

    if (cfg.save) {
      const std::string &metrics_checkpoint_path =
          selected_best_checkpoint ? best_checkpoint_path : cfg.checkpoint_path;
      CheckpointMetadata metadata =
          BuildCharCheckpointMetadata(cfg, corpus, vocab, offset_rng, selected_checkpoint_epoch,
                                      optimizer.step_count(), splits, metric_history, run_summary);
      DLCUDA_RETURN_IF_ERROR(
          SaveCheckpoint(ctx, metrics_checkpoint_path, metadata, params, &optimizer));
      std::printf("Updated checkpoint metrics: %s\n", metrics_checkpoint_path.c_str());
    }
  }

  auto generated = GenerateText(ctx, model, vocab, corpus, cfg.seq_len, cfg.gen_len,
                                cfg.temperature, cfg.top_p, cfg.sample_seed, cfg.prompt);
  if (!generated.ok()) {
    return generated.status();
  }
  std::printf("Generated text:\n  \"%s\"\n", generated.value().c_str());

  return Status::Ok();
}

} // namespace dlcuda
