#include "dl_cuda/examples.hpp"

#include "dl_cuda/checkpoint.hpp"
#include "dl_cuda/data.hpp"
#include "dl_cuda/loss.hpp"
#include "dl_cuda/nn.hpp"
#include "dl_cuda/optim.hpp"
#include "dl_cuda/runtime.hpp"
#include "dl_cuda/trainer.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace dlcuda {
namespace {

static const char *kCharCorpus = "To be, or not to be, that is the question. "
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

constexpr int kExampleThreads = 256;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
#define DLCUDA_CAN_CAPTURE_CUDA_GRAPHS 1
#endif

#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
class CudaGraphExec {
public:
  CudaGraphExec() = default;
  CudaGraphExec(const CudaGraphExec &) = delete;
  CudaGraphExec &operator=(const CudaGraphExec &) = delete;

  ~CudaGraphExec() {
    Reset();
  }

  [[nodiscard]] bool ready() const {
    return exec_ != nullptr;
  }

  void Reset() {
    if (exec_ != nullptr) {
      cudaGraphExecDestroy(exec_);
      exec_ = nullptr;
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
  }

  template <typename Fn> Status Capture(cudaStream_t stream, Fn &&fn) {
    Reset();
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
      return UnsupportedGraphStatus("cudaStreamBeginCapture", err);
    }

    Status body_status = fn();
    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(stream, &captured_graph);
    if (!body_status.ok()) {
      if (captured_graph != nullptr) {
        cudaGraphDestroy(captured_graph);
      }
      return Status::Unsupported("CUDA graph capture body failed: " + body_status.ToString());
    }
    if (err != cudaSuccess) {
      if (captured_graph != nullptr) {
        cudaGraphDestroy(captured_graph);
      }
      return UnsupportedGraphStatus("cudaStreamEndCapture", err);
    }

    cudaGraphExec_t captured_exec = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    err = cudaGraphInstantiateWithFlags(&captured_exec, captured_graph, 0);
#else
    err = cudaGraphInstantiate(&captured_exec, captured_graph, nullptr, nullptr, 0);
#endif
    if (err != cudaSuccess) {
      cudaGraphDestroy(captured_graph);
      return UnsupportedGraphStatus("cudaGraphInstantiate", err);
    }

    graph_ = captured_graph;
    exec_ = captured_exec;
    return Status::Ok();
  }

  Status Launch(cudaStream_t stream) {
    if (exec_ == nullptr) {
      return Status::InvalidArgument("CUDA graph has not been captured");
    }
    cudaError_t err = cudaGraphLaunch(exec_, stream);
    return detail::CudaStatus(err, "cudaGraphLaunch");
  }

private:
  static Status UnsupportedGraphStatus(const char *context, cudaError_t err) {
    return Status::Unsupported(std::string(context) + ": " + cudaGetErrorString(err));
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};
#endif

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

Status ValidatePositiveFinite(float value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0f)) {
    return Status::InvalidArgument(std::string(name) + " must be finite and > 0");
  }
  return Status::Ok();
}

Status ValidateTopP(float value) {
  if (!std::isfinite(value) || !(value > 0.0f && value <= 1.0f)) {
    return Status::InvalidArgument("top_p must be finite and in (0, 1]");
  }
  return Status::Ok();
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
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.lr, "lr"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.grad_clip, "grad_clip"));
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
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.lr, "lr"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.grad_clip, "grad_clip"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.temperature, "temperature"));
  DLCUDA_RETURN_IF_ERROR(ValidateTopP(cfg.top_p));
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
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.temperature, "temperature"));
  DLCUDA_RETURN_IF_ERROR(ValidateTopP(cfg.top_p));
  return Status::Ok();
}

Status ValidateCorpusWindow(size_t corpus_size, int seq_len, const char *context) {
  if (seq_len <= 0) {
    return Status::InvalidArgument(std::string(context) + " seq_len must be > 0");
  }
  if (static_cast<size_t>(seq_len) + 1 > corpus_size) {
    return Status::InvalidArgument(std::string(context) + " corpus is too short for seq_len");
  }
  return Status::Ok();
}

void ApplyTopP(std::vector<float> &probs, float p) {
  std::vector<int> idx(static_cast<int>(probs.size()));
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    idx[i] = i;
  }
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

  float total = 0.0f;
  for (float prob : probs) {
    total += prob;
  }
  if (total <= 0.0f) {
    return;
  }
  float target_mass = p * total;

  float cum = 0.0f;
  int cutoff = static_cast<int>(idx.size());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    cum += probs[idx[i]];
    if (cum >= target_mass) {
      cutoff = i + 1;
      break;
    }
  }

  for (int i = cutoff; i < static_cast<int>(idx.size()); ++i) {
    probs[idx[i]] = 0.0f;
  }
}

int SampleFromWeights(const std::vector<float> &probs, std::mt19937 &rng) {
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

int SampleToken(const std::vector<float> &raw_probs, float temperature, float top_p,
                std::mt19937 &rng) {
  if (temperature == 1.0f && top_p >= 1.0f) {
    return SampleFromWeights(raw_probs, rng);
  }

  std::vector<float> probs = raw_probs;

  if (temperature != 1.0f) {
    float inv_t = 1.0f / temperature;
    for (float &p : probs) {
      p = p > 0.0f ? std::pow(p, inv_t) : 0.0f;
    }
  }

  if (top_p < 1.0f) {
    ApplyTopP(probs, top_p);
  }

  return SampleFromWeights(probs, rng);
}

__global__ void ShiftAppendTokenKernel(int32_t *context, int64_t seq_len, int32_t next_id) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int64_t i = 0; i + 1 < seq_len; ++i) {
      context[i] = context[i + 1];
    }
    context[seq_len - 1] = next_id;
  }
}

__global__ void FillTrainingWindowKernel(const int32_t *encoded_corpus, int32_t *input_ids,
                                         int32_t *target_ids, int64_t seq_len, int64_t offset) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < seq_len) {
    input_ids[idx] = encoded_corpus[offset + idx];
    target_ids[idx] = encoded_corpus[offset + idx + 1];
  }
}

Status FillTrainingWindow(RuntimeContext &ctx, const Tensor &encoded_corpus_device,
                          Tensor *input_ids, Tensor *target_ids, int seq_len, int offset) {
  if (input_ids == nullptr || target_ids == nullptr) {
    return Status::InvalidArgument("FillTrainingWindow received null tensor pointer");
  }
  auto window_blocks = detail::BlocksForElements(seq_len, kExampleThreads);
  if (!window_blocks.ok()) {
    return window_blocks.status();
  }
  if (window_blocks.value() > 0) {
    FillTrainingWindowKernel<<<window_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
        encoded_corpus_device.data_as<int32_t>(), input_ids->data_as<int32_t>(),
        target_ids->data_as<int32_t>(), seq_len, offset);
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("FillTrainingWindowKernel"));
  }
  return Status::Ok();
}

Status RunCharTrainBody(RuntimeContext &ctx, Sequential &model, AdamOptimizer &optimizer,
                        const std::vector<ParameterRef> &params, const Tensor &input_ids,
                        const Tensor &target_ids, Tensor *logits, Tensor *loss_grad,
                        float grad_clip, ClassificationMetrics *metrics, float *grad_norm) {
  DLCUDA_RETURN_IF_ERROR(optimizer.ZeroGrad(ctx, params));

  DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, input_ids, logits));
  if (metrics != nullptr) {
    auto metrics_result = CategoricalCrossEntropyMetricsFromLogits(ctx, target_ids, *logits);
    if (!metrics_result.ok()) {
      return metrics_result.status();
    }
    *metrics = metrics_result.value();
  }

  DLCUDA_RETURN_IF_ERROR(
      CategoricalCrossEntropyBackwardFromLogits(ctx, target_ids, *logits, loss_grad));
  DLCUDA_RETURN_IF_ERROR(model.Backward(ctx, *loss_grad, nullptr));
  DLCUDA_RETURN_IF_ERROR(ClipGradNorm(ctx, params, grad_clip, grad_norm));
  return Status::Ok();
}

Status BuildCharModel(Sequential *model, RuntimeContext &ctx, int vocab_size, int d_model) {
  if (model == nullptr) {
    return Status::InvalidArgument("BuildCharModel requires a model pointer");
  }
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Embedding>(vocab_size, d_model, ctx)));
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Linear>(d_model, vocab_size, ctx)));
  return Status::Ok();
}

Result<std::string> GenerateText(RuntimeContext &ctx, Sequential &model, const CharVocab &vocab,
                                 int seq_len, int gen_len, float temperature, float top_p,
                                 uint64_t sample_seed) {
  if (gen_len < 0) {
    return Status::InvalidArgument("gen_len must be >= 0");
  }
  std::string text(kCharCorpus);
  DLCUDA_RETURN_IF_ERROR(ValidateCorpusWindow(text.size(), seq_len, "Generation"));

  std::vector<int32_t> context(static_cast<size_t>(seq_len));
  for (int i = 0; i < seq_len; ++i) {
    context[static_cast<size_t>(i)] = vocab.Encode(text[static_cast<size_t>(i)]);
  }

  auto context_tensor = Tensor::AllocateAsync({seq_len}, DType::kInt32, ctx.stream());
  if (!context_tensor.ok()) {
    return context_tensor.status();
  }
  auto input_ids = context_tensor.value();
  DLCUDA_RETURN_IF_ERROR(
      input_ids.CopyFromHost(context.data(), context.size() * sizeof(int32_t), ctx.stream()));

  std::string generated;
  size_t reserve_size = static_cast<size_t>(seq_len);
  size_t gen_len_size = static_cast<size_t>(gen_len);
  if (gen_len_size > std::numeric_limits<size_t>::max() - reserve_size) {
    return Status::InvalidArgument("generated text length is too large");
  }
  generated.reserve(reserve_size + gen_len_size);
  for (int i = 0; i < seq_len; ++i) {
    generated.push_back(vocab.Decode(context[static_cast<size_t>(i)]));
  }

  std::mt19937 rng(static_cast<uint32_t>(sample_seed));
  std::vector<float> host_probs(static_cast<size_t>(vocab.size()));
  Softmax softmax;

  for (int step = 0; step < gen_len; ++step) {
    Tensor logits;
    Tensor probs;
    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, input_ids, &logits));
    DLCUDA_RETURN_IF_ERROR(softmax.Forward(ctx, logits, &probs));

    if (probs.rank() != 2 || probs.dim(0) != seq_len || probs.dim(1) != vocab.size()) {
      return Status::RuntimeError("Generation probability tensor shape mismatch");
    }
    int vocab_size = static_cast<int>(probs.dim(1));
    size_t offset = static_cast<size_t>(seq_len - 1) * static_cast<size_t>(vocab_size);
    size_t offset_bytes = offset * sizeof(float);
    size_t copy_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
    DLCUDA_RETURN_IF_ERROR(
        probs.CopyRangeToHost(host_probs.data(), offset_bytes, copy_bytes, ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

    int next_id = SampleToken(host_probs, temperature, top_p, rng);
    generated.push_back(vocab.Decode(next_id));

    ShiftAppendTokenKernel<<<1, 1, 0, ctx.stream()>>>(input_ids.data_as<int32_t>(), seq_len,
                                                      static_cast<int32_t>(next_id));
    DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("ShiftAppendTokenKernel"));
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

  if (cfg.resume) {
    Status load_status = LoadCheckpoint(ctx, cfg.checkpoint_path, "xor-mlp", params);
    if (!load_status.ok()) {
      return Status::RuntimeError("Failed to resume XOR checkpoint: " + load_status.message());
    }
    std::printf("Loaded checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  std::printf("XOR | epochs=%d lr=%.4f hidden=%d | backend=%s | TF32=%s\n", cfg.epochs, cfg.lr,
              cfg.hidden_size, cfg.use_cublas ? "cuBLAS" : "kernels", cfg.tf32 ? "on" : "off");

  Tensor predictions;
  Tensor loss_grad;
  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    DLCUDA_RETURN_IF_ERROR(optimizer.ZeroGrad(ctx, params));

    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, x, &predictions));
    bool should_log = (epoch % cfg.print_every) == 0;
    float loss_value = 0.0f;
    if (should_log) {
      auto loss = BinaryCrossEntropyLoss(ctx, y, predictions);
      if (!loss.ok()) {
        return loss.status();
      }
      loss_value = loss.value();
    }

    DLCUDA_RETURN_IF_ERROR(BinaryCrossEntropyBackward(ctx, y, predictions, &loss_grad));
    DLCUDA_RETURN_IF_ERROR(model.Backward(ctx, loss_grad, nullptr));

    float grad_norm = 0.0f;
    DLCUDA_RETURN_IF_ERROR(
        ClipGradNorm(ctx, params, cfg.grad_clip, should_log ? &grad_norm : nullptr));
    DLCUDA_RETURN_IF_ERROR(optimizer.Step(ctx, params, cfg.lr));

    if (should_log) {
      std::printf("Epoch %4d | BCE: %.6f | GradNorm: %.4f\n", epoch, loss_value, grad_norm);
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
    CheckpointMetadata metadata;
    metadata.model_name = "xor-mlp";
    metadata.format_version = 2;
    DLCUDA_RETURN_IF_ERROR(SaveCheckpoint(ctx, cfg.checkpoint_path, metadata, params));
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

  DLCUDA_RETURN_IF_ERROR(ValidateCorpusWindow(corpus.size(), cfg.seq_len, "Training"));

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

  if (cfg.resume) {
    Status load_status = LoadCheckpoint(ctx, cfg.checkpoint_path, "char-embed-softmax", params);
    if (!load_status.ok()) {
      return Status::RuntimeError("Failed to resume char checkpoint: " + load_status.message());
    }
    std::printf("Loaded checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  std::printf("Char | vocab=%d seq_len=%d d_model=%d epochs=%d\n", vocab.size(), cfg.seq_len,
              cfg.d_model, cfg.epochs);
  std::printf("Optimizer: Adam | Grad clip: %.2f | temp=%.2f top_p=%.2f\n", cfg.grad_clip,
              cfg.temperature, cfg.top_p);

  std::mt19937 offset_rng(static_cast<uint32_t>(cfg.seed));
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

  int max_offset = static_cast<int>(corpus.size()) - cfg.seq_len - 1;
  Tensor logits;
  Tensor loss_grad;
#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
  CudaGraphExec train_graph;
  bool graph_capture_disabled = false;
#endif

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    int offset = static_cast<int>(offset_rng() % static_cast<uint32_t>(max_offset + 1));
    DLCUDA_RETURN_IF_ERROR(FillTrainingWindow(ctx, encoded_corpus_device, &input_ids, &target_ids,
                                              cfg.seq_len, offset));

    bool should_log = (epoch % cfg.print_every) == 0;
    ClassificationMetrics metrics;
    float grad_norm = 0.0f;

#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
    bool ran_graph = false;
    if (!should_log && !graph_capture_disabled) {
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
          should_log ? &metrics : nullptr, should_log ? &grad_norm : nullptr));
    }
    DLCUDA_RETURN_IF_ERROR(optimizer.Step(ctx, params, cfg.lr));

    if (should_log) {
      float ppl = std::exp(metrics.loss);
      float acc = metrics.accuracy * 100.0f;
      std::printf("Epoch %4d | Loss: %.4f | PPL: %7.2f | Acc: %5.1f%% | "
                  "GradNorm: %.4f\n",
                  epoch, metrics.loss, ppl, acc, grad_norm);
    }
  }

  DLCUDA_RETURN_IF_ERROR(ctx.Synchronize());

  auto train_end = std::chrono::steady_clock::now();
  if (cfg.epochs > 0) {
    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(train_end - train_start).count();
    double tokens = static_cast<double>(cfg.epochs) * cfg.seq_len;
    double tok_per_sec = sec > 0.0 ? tokens / sec : 0.0;
    std::printf("Training throughput: %.2f tokens/s (%.3f s)\n", tok_per_sec, sec);
  }

  if (cfg.save) {
    CheckpointMetadata metadata;
    metadata.model_name = "char-embed-softmax";
    metadata.format_version = 2;
    DLCUDA_RETURN_IF_ERROR(SaveCheckpoint(ctx, cfg.checkpoint_path, metadata, params));
    std::printf("Saved checkpoint: %s\n", cfg.checkpoint_path.c_str());
  }

  auto generated = GenerateText(ctx, model, vocab, cfg.seq_len, cfg.gen_len, cfg.temperature,
                                cfg.top_p, cfg.sample_seed);
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
  DLCUDA_RETURN_IF_ERROR(ValidateCorpusWindow(corpus.size(), cfg.seq_len, "Sampling"));

  RuntimeContext ctx(OptionsFromCharConfig(cfg.use_cublas, cfg.tf32, cfg.seed));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(BuildCharModel(&model, ctx, vocab.size(), cfg.d_model));
  DLCUDA_RETURN_IF_ERROR(
      LoadCheckpoint(ctx, cfg.checkpoint_path, "char-embed-softmax", model.parameters()));

  return GenerateText(ctx, model, vocab, cfg.seq_len, cfg.gen_len, cfg.temperature, cfg.top_p,
                      cfg.sample_seed);
}

} // namespace dlcuda
