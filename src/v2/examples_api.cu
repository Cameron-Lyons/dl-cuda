#include "dl_cuda/examples.hpp"

#include "dl_cuda/checkpoint.hpp"
#include "dl_cuda/data.hpp"
#include "dl_cuda/loss.hpp"
#include "dl_cuda/nn.hpp"
#include "dl_cuda/optim.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/trainer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace dlcuda {
namespace {

static const char *kCharCorpus =
    "To be, or not to be, that is the question. "
    "Whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles, "
    "and by opposing end them. To die, to sleep, "
    "no more, and by a sleep to say we end "
    "the heartache and the thousand natural shocks "
    "that flesh is heir to. Tis a consummation "
    "devoutly to be wished. To die, to sleep. "
    "To sleep, perchance to dream. Ay, there's the rub, "
    "for in that sleep of death what dreams may come "
    "when we have shuffled off this mortal coil, "
    "must give us pause. There's the respect "
    "that makes calamity of so long life. ";

RuntimeOptions OptionsFromXorConfig(const TrainXorConfig &cfg) {
  RuntimeOptions opts;
  opts.use_cublas = cfg.use_cublas;
  opts.tf32 = cfg.tf32;
  opts.seed = cfg.seed;
  opts.stream = 0;
  return opts;
}

RuntimeOptions OptionsFromCharConfig(bool use_cublas, bool tf32, uint64_t seed) {
  RuntimeOptions opts;
  opts.use_cublas = use_cublas;
  opts.tf32 = tf32;
  opts.seed = seed;
  opts.stream = 0;
  return opts;
}

Status ValidateXorConfig(const TrainXorConfig &cfg) {
  if (cfg.epochs < 0) {
    return Status::InvalidArgument("epochs must be >= 0");
  }
  if (cfg.print_every <= 0) {
    return Status::InvalidArgument("print_every must be > 0");
  }
  if (cfg.hidden_size <= 0) {
    return Status::InvalidArgument("hidden_size must be > 0");
  }
  if (!(cfg.lr > 0.0f)) {
    return Status::InvalidArgument("lr must be > 0");
  }
  if (!(cfg.grad_clip > 0.0f)) {
    return Status::InvalidArgument("grad_clip must be > 0");
  }
  return Status::Ok();
}

Status ValidateCharConfig(const TrainCharConfig &cfg) {
  if (cfg.seq_len <= 1) {
    return Status::InvalidArgument("seq_len must be > 1");
  }
  if (cfg.d_model <= 0) {
    return Status::InvalidArgument("d_model must be > 0");
  }
  if (cfg.epochs < 0) {
    return Status::InvalidArgument("epochs must be >= 0");
  }
  if (cfg.print_every <= 0) {
    return Status::InvalidArgument("print_every must be > 0");
  }
  if (!(cfg.lr > 0.0f)) {
    return Status::InvalidArgument("lr must be > 0");
  }
  if (!(cfg.grad_clip > 0.0f)) {
    return Status::InvalidArgument("grad_clip must be > 0");
  }
  if (!(cfg.temperature > 0.0f)) {
    return Status::InvalidArgument("temperature must be > 0");
  }
  if (!(cfg.top_p > 0.0f && cfg.top_p <= 1.0f)) {
    return Status::InvalidArgument("top_p must be in (0, 1]");
  }
  if (cfg.gen_len < 0) {
    return Status::InvalidArgument("gen_len must be >= 0");
  }
  return Status::Ok();
}

Status ValidateSampleCharConfig(const SampleCharConfig &cfg) {
  if (cfg.seq_len <= 1) {
    return Status::InvalidArgument("seq_len must be > 1");
  }
  if (cfg.d_model <= 0) {
    return Status::InvalidArgument("d_model must be > 0");
  }
  if (cfg.gen_len < 0) {
    return Status::InvalidArgument("gen_len must be >= 0");
  }
  if (!(cfg.temperature > 0.0f)) {
    return Status::InvalidArgument("temperature must be > 0");
  }
  if (!(cfg.top_p > 0.0f && cfg.top_p <= 1.0f)) {
    return Status::InvalidArgument("top_p must be in (0, 1]");
  }
  return Status::Ok();
}

void ApplyTopP(std::vector<float> &probs, float p) {
  std::vector<int> idx(static_cast<int>(probs.size()));
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    idx[i] = i;
  }
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

  float cum = 0.0f;
  int cutoff = static_cast<int>(idx.size());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    cum += probs[idx[i]];
    if (cum >= p) {
      cutoff = i + 1;
      break;
    }
  }

  for (int i = cutoff; i < static_cast<int>(idx.size()); ++i) {
    probs[idx[i]] = 0.0f;
  }
}

int SampleToken(const std::vector<float> &raw_probs, float temperature, float top_p,
                std::mt19937 &rng) {
  std::vector<float> probs = raw_probs;

  if (temperature != 1.0f) {
    float inv_t = 1.0f / temperature;
    for (float &p : probs) {
      p = std::pow(std::max(1e-8f, p), inv_t);
    }
  }

  if (top_p < 1.0f) {
    ApplyTopP(probs, top_p);
  }

  float sum = 0.0f;
  for (float p : probs) {
    sum += p;
  }
  if (sum <= 0.0f) {
    return 0;
  }

  std::uniform_real_distribution<float> dist(0.0f, sum);
  float r = dist(rng);
  float cum = 0.0f;
  for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
    cum += probs[i];
    if (r <= cum) {
      return i;
    }
  }
  return static_cast<int>(probs.size()) - 1;
}

Status BuildCharModel(Sequential *model, RuntimeContext &ctx, int vocab_size,
                      int d_model) {
  if (model == nullptr) {
    return Status::InvalidArgument("BuildCharModel requires a model pointer");
  }
  DLCUDA_RETURN_IF_ERROR(
      model->Add(std::make_unique<Embedding>(vocab_size, d_model, ctx)));
  DLCUDA_RETURN_IF_ERROR(
      model->Add(std::make_unique<Linear>(d_model, vocab_size, ctx)));
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Softmax>()));
  return Status::Ok();
}

Result<std::string> GenerateText(RuntimeContext &ctx, Sequential &model,
                                 const CharVocab &vocab, int seq_len,
                                 int gen_len, float temperature, float top_p,
                                 uint64_t sample_seed) {
  std::string text(kCharCorpus);
  if (static_cast<int>(text.size()) < seq_len + 1) {
    return Status::InvalidArgument("Corpus is too short for generation window");
  }

  std::vector<int32_t> context(static_cast<size_t>(seq_len));
  for (int i = 0; i < seq_len; ++i) {
    context[static_cast<size_t>(i)] = vocab.Encode(text[static_cast<size_t>(i)]);
  }

  auto context_tensor = Tensor::Allocate({seq_len}, DType::kInt32);
  if (!context_tensor.ok()) {
    return context_tensor.status();
  }
  auto input_ids = context_tensor.value();
  DLCUDA_RETURN_IF_ERROR(
      input_ids.CopyFromHost(context.data(), context.size() * sizeof(int32_t),
                             ctx.stream()));

  std::string generated;
  generated.reserve(static_cast<size_t>(seq_len + gen_len));
  for (int i = 0; i < seq_len; ++i) {
    generated.push_back(vocab.Decode(context[static_cast<size_t>(i)]));
  }

  std::mt19937 rng(static_cast<uint32_t>(sample_seed));

  for (int step = 0; step < gen_len; ++step) {
    Tensor probs;
    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, input_ids, &probs));

    int vocab_size = static_cast<int>(probs.dim(1));
    std::vector<float> host_probs(static_cast<size_t>(vocab_size));
    size_t offset = static_cast<size_t>(seq_len - 1) * static_cast<size_t>(vocab_size);
    cudaError_t copy_status = cudaMemcpyAsync(host_probs.data(),
                                              probs.data_as<float>() + offset,
                                              static_cast<size_t>(vocab_size) *
                                                  sizeof(float),
                                              cudaMemcpyDeviceToHost,
                                              ctx.stream());
    if (copy_status != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaMemcpyAsync failed in generation: ") +
                                  cudaGetErrorString(copy_status));
    }
    cudaError_t sync_status = cudaStreamSynchronize(ctx.stream());
    if (sync_status != cudaSuccess) {
      return Status::RuntimeError(std::string("cudaStreamSynchronize failed in generation: ") +
                                  cudaGetErrorString(sync_status));
    }

    int next_id = SampleToken(host_probs, temperature, top_p, rng);
    generated.push_back(vocab.Decode(next_id));

    for (int i = 0; i < seq_len - 1; ++i) {
      context[static_cast<size_t>(i)] = context[static_cast<size_t>(i + 1)];
    }
    context[static_cast<size_t>(seq_len - 1)] = static_cast<int32_t>(next_id);

    DLCUDA_RETURN_IF_ERROR(
        input_ids.CopyFromHost(context.data(), context.size() * sizeof(int32_t),
                               ctx.stream()));
  }

  return generated;
}

} // namespace

Status TrainXor(const TrainXorConfig &cfg) {
  DLCUDA_RETURN_IF_ERROR(ValidateXorConfig(cfg));

  RuntimeContext ctx(OptionsFromXorConfig(cfg));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<Linear>(2, cfg.hidden_size, ctx)));
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<ReLU>()));
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<Linear>(cfg.hidden_size, 1, ctx)));
  DLCUDA_RETURN_IF_ERROR(model.Add(std::make_unique<Sigmoid>()));

  AdamOptimizer optimizer;

  std::vector<float> host_x = {
      0.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
  };
  std::vector<float> host_y = {0.0f, 1.0f, 1.0f, 0.0f};

  auto x_tensor = Tensor::Allocate({4, 2}, DType::kFloat32);
  if (!x_tensor.ok()) {
    return x_tensor.status();
  }
  auto y_tensor = Tensor::Allocate({4, 1}, DType::kFloat32);
  if (!y_tensor.ok()) {
    return y_tensor.status();
  }

  Tensor x = x_tensor.value();
  Tensor y = y_tensor.value();
  DLCUDA_RETURN_IF_ERROR(
      x.CopyFromHost(host_x.data(), host_x.size() * sizeof(float), ctx.stream()));
  DLCUDA_RETURN_IF_ERROR(
      y.CopyFromHost(host_y.data(), host_y.size() * sizeof(float), ctx.stream()));

  if (cfg.resume) {
    Status load_status =
        LoadCheckpoint(ctx, cfg.checkpoint_path, "xor-mlp-v2", model.Parameters());
    if (!load_status.ok()) {
      return Status::RuntimeError("Failed to resume XOR checkpoint: " +
                                  load_status.message());
    }
    std::printf("Loaded checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  std::printf("XOR v2 | epochs=%d lr=%.4f hidden=%d | backend=%s | TF32=%s\n",
              cfg.epochs, cfg.lr, cfg.hidden_size,
              cfg.use_cublas ? "cuBLAS" : "kernels",
              cfg.tf32 ? "on" : "off");

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    auto params = model.Parameters();
    DLCUDA_RETURN_IF_ERROR(optimizer.ZeroGrad(ctx, params));

    Tensor predictions;
    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, x, &predictions));

    Tensor loss_grad;
    auto loss = BinaryCrossEntropy(ctx, y, predictions, &loss_grad);
    if (!loss.ok()) {
      return loss.status();
    }

    Tensor input_grad;
    DLCUDA_RETURN_IF_ERROR(model.Backward(ctx, loss_grad, &input_grad));

    auto grad_norm = ClipGradNorm(ctx, params, cfg.grad_clip);
    if (!grad_norm.ok()) {
      return grad_norm.status();
    }

    DLCUDA_RETURN_IF_ERROR(optimizer.Step(ctx, params, cfg.lr));

    if (epoch % cfg.print_every == 0) {
      std::printf("Epoch %4d | BCE: %.6f | GradNorm: %.4f\n", epoch,
                  loss.value(), grad_norm.value());
    }
  }

  Tensor final_predictions;
  DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, x, &final_predictions));

  std::vector<float> host_pred(4);
  DLCUDA_RETURN_IF_ERROR(
      final_predictions.CopyToHost(host_pred.data(), host_pred.size() * sizeof(float),
                                   ctx.stream()));
  cudaError_t sync_status = cudaStreamSynchronize(ctx.stream());
  if (sync_status != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(sync_status));
  }

  std::printf("Final predictions:\n");
  std::printf("  [0, 0] -> %.4f (expected 0)\n", host_pred[0]);
  std::printf("  [0, 1] -> %.4f (expected 1)\n", host_pred[1]);
  std::printf("  [1, 0] -> %.4f (expected 1)\n", host_pred[2]);
  std::printf("  [1, 1] -> %.4f (expected 0)\n", host_pred[3]);

  if (cfg.save) {
    CheckpointMetadata metadata;
    metadata.model_name = "xor-mlp-v2";
    metadata.format_version = 2;
    DLCUDA_RETURN_IF_ERROR(
        SaveCheckpoint(ctx, cfg.checkpoint_path, metadata, model.Parameters()));
    std::printf("Saved checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  return Status::Ok();
}

Status TrainChar(const TrainCharConfig &cfg) {
  DLCUDA_RETURN_IF_ERROR(ValidateCharConfig(cfg));

  std::string corpus(kCharCorpus);
  auto vocab_result = CharVocab::Build(corpus);
  if (!vocab_result.ok()) {
    return vocab_result.status();
  }
  CharVocab vocab = vocab_result.value();

  if (static_cast<int>(corpus.size()) <= cfg.seq_len + 1) {
    return Status::InvalidArgument("Corpus must be longer than seq_len + 1");
  }

  RuntimeContext ctx(OptionsFromCharConfig(cfg.use_cublas, cfg.tf32, cfg.seed));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(BuildCharModel(&model, ctx, vocab.size(), cfg.d_model));

  AdamOptimizer optimizer;

  auto input_ids_result = Tensor::Allocate({cfg.seq_len}, DType::kInt32);
  if (!input_ids_result.ok()) {
    return input_ids_result.status();
  }
  auto target_ids_result = Tensor::Allocate({cfg.seq_len}, DType::kInt32);
  if (!target_ids_result.ok()) {
    return target_ids_result.status();
  }

  Tensor input_ids = input_ids_result.value();
  Tensor target_ids = target_ids_result.value();

  if (cfg.resume) {
    Status load_status = LoadCheckpoint(ctx, cfg.checkpoint_path,
                                        "char-embed-softmax-v2", model.Parameters());
    if (!load_status.ok()) {
      return Status::RuntimeError("Failed to resume char checkpoint: " +
                                  load_status.message());
    }
    std::printf("Loaded checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  std::printf("Char v2 | vocab=%d seq_len=%d d_model=%d epochs=%d\n", vocab.size(),
              cfg.seq_len, cfg.d_model, cfg.epochs);
  std::printf("Optimizer: Adam | Grad clip: %.2f | temp=%.2f top_p=%.2f\n",
              cfg.grad_clip, cfg.temperature, cfg.top_p);

  std::mt19937 offset_rng(static_cast<uint32_t>(cfg.seed));
  auto train_start = std::chrono::steady_clock::now();

  std::vector<int32_t> host_input(static_cast<size_t>(cfg.seq_len));
  std::vector<int32_t> host_target(static_cast<size_t>(cfg.seq_len));

  int max_offset = static_cast<int>(corpus.size()) - cfg.seq_len - 1;

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    int offset = static_cast<int>(offset_rng() % static_cast<uint32_t>(max_offset + 1));
    for (int i = 0; i < cfg.seq_len; ++i) {
      host_input[static_cast<size_t>(i)] =
          vocab.Encode(corpus[static_cast<size_t>(offset + i)]);
      host_target[static_cast<size_t>(i)] =
          vocab.Encode(corpus[static_cast<size_t>(offset + i + 1)]);
    }

    DLCUDA_RETURN_IF_ERROR(
        input_ids.CopyFromHost(host_input.data(), host_input.size() * sizeof(int32_t),
                               ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(
        target_ids.CopyFromHost(host_target.data(),
                                host_target.size() * sizeof(int32_t),
                                ctx.stream()));

    auto params = model.Parameters();
    DLCUDA_RETURN_IF_ERROR(optimizer.ZeroGrad(ctx, params));

    Tensor probs;
    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, input_ids, &probs));

    Tensor loss_grad;
    auto metrics = CategoricalCrossEntropyFromIds(ctx, target_ids, probs, &loss_grad);
    if (!metrics.ok()) {
      return metrics.status();
    }

    Tensor input_grad;
    DLCUDA_RETURN_IF_ERROR(model.Backward(ctx, loss_grad, &input_grad));

    auto grad_norm = ClipGradNorm(ctx, params, cfg.grad_clip);
    if (!grad_norm.ok()) {
      return grad_norm.status();
    }

    DLCUDA_RETURN_IF_ERROR(optimizer.Step(ctx, params, cfg.lr));

    if (epoch % cfg.print_every == 0) {
      float ppl = std::exp(metrics.value().loss);
      float acc = metrics.value().accuracy * 100.0f;
      std::printf("Epoch %4d | Loss: %.4f | PPL: %7.2f | Acc: %5.1f%% | "
                  "GradNorm: %.4f\n",
                  epoch, metrics.value().loss, ppl, acc, grad_norm.value());
    }
  }

  cudaError_t sync_status = cudaStreamSynchronize(ctx.stream());
  if (sync_status != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(sync_status));
  }

  auto train_end = std::chrono::steady_clock::now();
  if (cfg.epochs > 0) {
    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(train_end - train_start)
            .count();
    double tokens = static_cast<double>(cfg.epochs) * cfg.seq_len;
    double tok_per_sec = sec > 0.0 ? tokens / sec : 0.0;
    std::printf("Training throughput: %.2f tokens/s (%.3f s)\n", tok_per_sec,
                sec);
  }

  if (cfg.save) {
    CheckpointMetadata metadata;
    metadata.model_name = "char-embed-softmax-v2";
    metadata.format_version = 2;
    DLCUDA_RETURN_IF_ERROR(
        SaveCheckpoint(ctx, cfg.checkpoint_path, metadata, model.Parameters()));
    std::printf("Saved checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  auto generated = GenerateText(ctx, model, vocab, cfg.seq_len, cfg.gen_len,
                                cfg.temperature, cfg.top_p, cfg.sample_seed);
  if (!generated.ok()) {
    return generated.status();
  }
  std::printf("Generated text:\n  \"%s\"\n", generated.value().c_str());

  return Status::Ok();
}

Result<std::string> SampleChar(const SampleCharConfig &cfg) {
  DLCUDA_RETURN_IF_ERROR(ValidateSampleCharConfig(cfg));

  std::string corpus(kCharCorpus);
  auto vocab_result = CharVocab::Build(corpus);
  if (!vocab_result.ok()) {
    return vocab_result.status();
  }
  CharVocab vocab = vocab_result.value();

  RuntimeContext ctx(OptionsFromCharConfig(cfg.use_cublas, cfg.tf32, cfg.seed));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(BuildCharModel(&model, ctx, vocab.size(), cfg.d_model));
  DLCUDA_RETURN_IF_ERROR(LoadCheckpoint(ctx, cfg.checkpoint_path,
                                        "char-embed-softmax-v2", model.Parameters()));

  return GenerateText(ctx, model, vocab, cfg.seq_len, cfg.gen_len,
                      cfg.temperature, cfg.top_p, cfg.sample_seed);
}

} // namespace dlcuda
