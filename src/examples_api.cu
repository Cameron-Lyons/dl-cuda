#include "dl_cuda/examples.hpp"

#include "dl_cuda/checkpoint.hpp"
#include "dl_cuda/data.hpp"
#include "dl_cuda/loss.hpp"
#include "dl_cuda/nn.hpp"
#include "dl_cuda/optim.hpp"
#include "dl_cuda/runtime.hpp"

#include "dl_cuda/detail/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <cstdio>
#include <limits>
#include <random>
#include <sstream>
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
constexpr int kCharContextWidth = 5;
constexpr const char *kCharModelName = "char-causal-conv-lm";

Result<std::string> LoadTextFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Status::IoError("Failed to open char corpus file: " + path);
  }
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (file.bad()) {
    return Status::IoError("Failed to read char corpus file: " + path);
  }
  return text;
}

Result<std::string> LoadCharCorpus(const std::string &data_path) {
  if (data_path.empty()) {
    return std::string(kCharCorpus);
  }
  return LoadTextFile(data_path);
}

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

Status ValidateFraction(float value, const char *name) {
  if (!std::isfinite(value) || value < 0.0f || value >= 1.0f) {
    return Status::InvalidArgument(std::string(name) + " must be finite and in [0, 1)");
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
  if (cfg.val_every <= 0) {
    return Status::InvalidArgument("val_every must be > 0");
  }
  if (cfg.val_windows <= 0) {
    return Status::InvalidArgument("val_windows must be > 0");
  }
  if (cfg.test_windows <= 0) {
    return Status::InvalidArgument("test_windows must be > 0");
  }
  if (cfg.early_stop_patience < 0) {
    return Status::InvalidArgument("early_stop_patience must be >= 0");
  }
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.lr, "lr"));
  DLCUDA_RETURN_IF_ERROR(ValidatePositiveFinite(cfg.grad_clip, "grad_clip"));
  DLCUDA_RETURN_IF_ERROR(ValidateFraction(cfg.val_fraction, "val_fraction"));
  DLCUDA_RETURN_IF_ERROR(ValidateFraction(cfg.test_fraction, "test_fraction"));
  if (cfg.val_fraction + cfg.test_fraction >= 1.0f) {
    return Status::InvalidArgument("val_fraction + test_fraction must be < 1");
  }
  if (cfg.early_stop_patience > 0 && cfg.val_fraction <= 0.0f) {
    return Status::InvalidArgument("early stopping requires val_fraction > 0");
  }
  if (!std::isfinite(cfg.min_delta) || cfg.min_delta < 0.0f) {
    return Status::InvalidArgument("min_delta must be finite and >= 0");
  }
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

std::string BoolString(bool value) {
  return value ? "true" : "false";
}

std::string IntString(int64_t value) {
  return std::to_string(value);
}

std::string UIntString(uint64_t value) {
  return std::to_string(value);
}

std::string FloatString(float value) {
  std::ostringstream out;
  out.precision(std::numeric_limits<float>::max_digits10);
  out << value;
  return out.str();
}

uint64_t HashBytes(const std::string &bytes) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  for (unsigned char byte : bytes) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

std::string HashString(const std::string &bytes) {
  return UIntString(HashBytes(bytes));
}

std::string VocabBytes(const CharVocab &vocab) {
  std::string bytes;
  bytes.reserve(static_cast<size_t>(vocab.size()));
  for (int i = 0; i < vocab.size(); ++i) {
    bytes.push_back(vocab.Decode(i));
  }
  return bytes;
}

std::string SerializeMt19937(const std::mt19937 &rng) {
  std::ostringstream out;
  out << rng;
  return out.str();
}

const std::string *FindCheckpointValue(const std::vector<CheckpointKeyValue> &values,
                                       const std::string &key) {
  for (const auto &value : values) {
    if (value.key == key) {
      return &value.value;
    }
  }
  return nullptr;
}

Status RequireCheckpointValue(const std::vector<CheckpointKeyValue> &values, const std::string &key,
                              const std::string &expected, const char *context) {
  const std::string *actual = FindCheckpointValue(values, key);
  if (actual == nullptr) {
    return Status::InvalidArgument(std::string("Checkpoint missing ") + context +
                                   " metadata: " + key);
  }
  if (*actual != expected) {
    return Status::InvalidArgument(std::string("Checkpoint ") + context + " mismatch for " + key +
                                   ": expected " + expected + " got " + *actual);
  }
  return Status::Ok();
}

Result<int64_t> ComputeEndEpoch(int64_t start_epoch, int epochs_to_run) {
  if (start_epoch < 0) {
    return Status::InvalidArgument("Checkpoint epoch must be non-negative");
  }
  if (start_epoch > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(epochs_to_run)) {
    return Status::InvalidArgument("Checkpoint epoch plus requested epochs overflows");
  }
  return start_epoch + static_cast<int64_t>(epochs_to_run);
}

Status ValidateOptimizerStepMetadata(const CheckpointMetadata &metadata,
                                     const Optimizer &optimizer) {
  if (metadata.format_version >= 3 && metadata.step != optimizer.step_count()) {
    return Status::InvalidArgument("Checkpoint optimizer step does not match metadata step");
  }
  return Status::Ok();
}

Status RestoreMt19937State(const CheckpointMetadata &metadata, const std::string &name,
                           std::mt19937 *rng) {
  if (rng == nullptr) {
    return Status::InvalidArgument("RNG restore destination is null");
  }
  const std::string *state = FindCheckpointValue(metadata.rng_states, name);
  if (state == nullptr) {
    return Status::NotFound("Checkpoint missing RNG state: " + name);
  }
  std::istringstream in(*state);
  in >> *rng;
  if (!in) {
    return Status::InvalidArgument("Checkpoint RNG state is invalid: " + name);
  }
  return Status::Ok();
}

std::vector<CheckpointKeyValue> ConstantSchedulerState(float lr, int64_t step) {
  return {{"type", "constant"}, {"base_lr", FloatString(lr)}, {"step", IntString(step)}};
}

std::vector<CheckpointKeyValue> BuildXorTrainingConfig(const TrainXorConfig &cfg) {
  return {{"command", "train-xor"},
          {"hidden_size", IntString(cfg.hidden_size)},
          {"epochs", IntString(cfg.epochs)},
          {"print_every", IntString(cfg.print_every)},
          {"lr", FloatString(cfg.lr)},
          {"grad_clip", FloatString(cfg.grad_clip)},
          {"use_cublas", BoolString(cfg.use_cublas)},
          {"tf32", BoolString(cfg.tf32)},
          {"seed", UIntString(cfg.seed)}};
}

CheckpointMetadata BuildXorCheckpointMetadata(const TrainXorConfig &cfg, int64_t completed_epoch,
                                              int64_t step) {
  CheckpointMetadata metadata;
  metadata.model_name = "xor-mlp";
  metadata.format_version = 3;
  metadata.epoch = completed_epoch;
  metadata.step = step;
  metadata.training_config = BuildXorTrainingConfig(cfg);
  metadata.scheduler_state = ConstantSchedulerState(cfg.lr, step);
  metadata.extra_metadata = {{"dataset", "xor"}, {"samples", "4"}};
  return metadata;
}

Status ValidateXorCheckpointMetadata(const CheckpointMetadata &metadata,
                                     const TrainXorConfig &cfg) {
  if (metadata.format_version < 3) {
    return Status::Ok();
  }
  return RequireCheckpointValue(metadata.training_config, "hidden_size", IntString(cfg.hidden_size),
                                "training_config");
}

std::vector<CheckpointKeyValue> BuildCharTrainingConfig(const TrainCharConfig &cfg) {
  return {{"command", "train-char"},
          {"seq_len", IntString(cfg.seq_len)},
          {"d_model", IntString(cfg.d_model)},
          {"epochs", IntString(cfg.epochs)},
          {"print_every", IntString(cfg.print_every)},
          {"val_every", IntString(cfg.val_every)},
          {"val_windows", IntString(cfg.val_windows)},
          {"test_windows", IntString(cfg.test_windows)},
          {"early_stop_patience", IntString(cfg.early_stop_patience)},
          {"lr", FloatString(cfg.lr)},
          {"grad_clip", FloatString(cfg.grad_clip)},
          {"val_fraction", FloatString(cfg.val_fraction)},
          {"test_fraction", FloatString(cfg.test_fraction)},
          {"min_delta", FloatString(cfg.min_delta)},
          {"temperature", FloatString(cfg.temperature)},
          {"top_p", FloatString(cfg.top_p)},
          {"gen_len", IntString(cfg.gen_len)},
          {"use_cublas", BoolString(cfg.use_cublas)},
          {"tf32", BoolString(cfg.tf32)},
          {"seed", UIntString(cfg.seed)},
          {"sample_seed", UIntString(cfg.sample_seed)}};
}

std::vector<CheckpointKeyValue> BuildCorpusMetadata(const TrainCharConfig &cfg,
                                                    const std::string &corpus) {
  return {{"source", cfg.data_path.empty() ? "embedded" : cfg.data_path},
          {"bytes", UIntString(static_cast<uint64_t>(corpus.size()))},
          {"hash_fnv1a64", HashString(corpus)}};
}

std::vector<CheckpointKeyValue> BuildVocabMetadata(const CharVocab &vocab) {
  return {{"size", IntString(vocab.size())}, {"hash_fnv1a64", HashString(VocabBytes(vocab))}};
}

struct WindowSplit {
  int64_t begin = 0;
  int64_t count = 0;
};

struct CharDatasetSplits {
  int64_t total_windows = 0;
  WindowSplit train;
  WindowSplit validation;
  WindowSplit test;
};

struct MetricSnapshot {
  int64_t epoch = 0;
  ClassificationMetrics train;
  bool has_validation = false;
  ClassificationMetrics validation;
  float grad_norm = 0.0f;
};

struct EvaluationSummary {
  ClassificationMetrics metrics;
  int64_t windows = 0;
};

struct CharRunSummary {
  bool has_best = false;
  int64_t best_epoch = 0;
  float best_val_loss = 0.0f;
  bool early_stopped = false;
  int64_t stopped_epoch = 0;
  bool has_test = false;
  EvaluationSummary test;
};

Result<int64_t> FractionToWindowCount(int64_t total_windows, float fraction, const char *name) {
  if (total_windows <= 0) {
    return Status::InvalidArgument(std::string(name) + " cannot split an empty window set");
  }
  if (fraction <= 0.0f) {
    return static_cast<int64_t>(0);
  }
  double raw_count = std::floor(static_cast<double>(total_windows) * static_cast<double>(fraction));
  int64_t count = static_cast<int64_t>(raw_count);
  if (count <= 0) {
    count = 1;
  }
  return count;
}

Result<CharDatasetSplits> BuildCharDatasetSplits(size_t corpus_size, const TrainCharConfig &cfg) {
  if (corpus_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return Status::InvalidArgument("corpus is too large to index");
  }
  int64_t corpus_len = static_cast<int64_t>(corpus_size);
  int64_t total_windows = corpus_len - static_cast<int64_t>(cfg.seq_len);
  if (total_windows <= 0) {
    return Status::InvalidArgument("Training corpus has no eligible windows");
  }

  auto val_count_result = FractionToWindowCount(total_windows, cfg.val_fraction, "val_fraction");
  if (!val_count_result.ok()) {
    return val_count_result.status();
  }
  auto test_count_result = FractionToWindowCount(total_windows, cfg.test_fraction, "test_fraction");
  if (!test_count_result.ok()) {
    return test_count_result.status();
  }
  int64_t val_count = val_count_result.value();
  int64_t test_count = test_count_result.value();
  if (val_count + test_count >= total_windows) {
    return Status::InvalidArgument("validation/test split leaves no training windows");
  }

  CharDatasetSplits splits;
  splits.total_windows = total_windows;
  splits.train = {0, total_windows - val_count - test_count};
  splits.validation = {splits.train.begin + splits.train.count, val_count};
  splits.test = {splits.validation.begin + splits.validation.count, test_count};
  return splits;
}

std::string SerializeMetricHistory(const std::vector<MetricSnapshot> &history) {
  std::ostringstream out;
  out.precision(std::numeric_limits<float>::max_digits10);
  out << "epoch,train_loss,train_ppl,train_accuracy,val_loss,val_ppl,val_accuracy,grad_norm";
  for (const auto &record : history) {
    out << '\n'
        << record.epoch << ',' << record.train.loss << ',' << std::exp(record.train.loss) << ','
        << record.train.accuracy << ',';
    if (record.has_validation) {
      out << record.validation.loss << ',' << std::exp(record.validation.loss) << ','
          << record.validation.accuracy;
    } else {
      out << ",,";
    }
    out << ',' << record.grad_norm;
  }
  return out.str();
}

std::vector<CheckpointKeyValue> BuildCharExtraMetadata(const CharDatasetSplits &splits,
                                                       const std::vector<MetricSnapshot> &history,
                                                       const CharRunSummary &summary) {
  std::vector<CheckpointKeyValue> extra = {{"split_strategy", "contiguous_offset_windows"},
                                           {"total_windows", IntString(splits.total_windows)},
                                           {"train_window_begin", IntString(splits.train.begin)},
                                           {"train_windows", IntString(splits.train.count)},
                                           {"val_window_begin", IntString(splits.validation.begin)},
                                           {"val_windows", IntString(splits.validation.count)},
                                           {"test_window_begin", IntString(splits.test.begin)},
                                           {"test_windows", IntString(splits.test.count)},
                                           {"metrics_history_csv", SerializeMetricHistory(history)},
                                           {"early_stopped", BoolString(summary.early_stopped)}};
  if (summary.has_best) {
    extra.push_back({"best_epoch", IntString(summary.best_epoch)});
    extra.push_back({"best_val_loss", FloatString(summary.best_val_loss)});
  }
  if (summary.early_stopped) {
    extra.push_back({"stopped_epoch", IntString(summary.stopped_epoch)});
  }
  if (summary.has_test) {
    extra.push_back({"test_eval_windows", IntString(summary.test.windows)});
    extra.push_back({"test_loss", FloatString(summary.test.metrics.loss)});
    extra.push_back(
        {"test_ppl", FloatString(static_cast<float>(std::exp(summary.test.metrics.loss)))});
    extra.push_back({"test_accuracy", FloatString(summary.test.metrics.accuracy)});
  }
  return extra;
}

std::string BestCheckpointPath(const TrainCharConfig &cfg) {
  if (!cfg.best_checkpoint_path.empty()) {
    return cfg.best_checkpoint_path;
  }
  if (cfg.checkpoint_path.empty()) {
    return std::string();
  }
  size_t slash = cfg.checkpoint_path.find_last_of("/\\");
  size_t basename = slash == std::string::npos ? 0 : slash + 1;
  size_t dot = cfg.checkpoint_path.find_last_of('.');
  if (dot == std::string::npos || dot < basename) {
    return cfg.checkpoint_path + ".best";
  }
  return cfg.checkpoint_path.substr(0, dot) + ".best" + cfg.checkpoint_path.substr(dot);
}

CheckpointMetadata BuildCharCheckpointMetadata(const TrainCharConfig &cfg,
                                               const std::string &corpus, const CharVocab &vocab,
                                               const std::mt19937 &offset_rng,
                                               int64_t completed_epoch, int64_t step,
                                               const CharDatasetSplits &splits,
                                               const std::vector<MetricSnapshot> &history,
                                               const CharRunSummary &summary) {
  CheckpointMetadata metadata;
  metadata.model_name = kCharModelName;
  metadata.format_version = 3;
  metadata.epoch = completed_epoch;
  metadata.step = step;
  metadata.training_config = BuildCharTrainingConfig(cfg);
  metadata.corpus_metadata = BuildCorpusMetadata(cfg, corpus);
  metadata.vocab_metadata = BuildVocabMetadata(vocab);
  metadata.scheduler_state = ConstantSchedulerState(cfg.lr, step);
  metadata.extra_metadata = BuildCharExtraMetadata(splits, history, summary);
  metadata.rng_states = {{"char.offset_rng", SerializeMt19937(offset_rng)}};
  return metadata;
}

Status ValidateCharCheckpointMetadata(const CheckpointMetadata &metadata, int seq_len, int d_model,
                                      const std::string &corpus, const CharVocab &vocab) {
  if (metadata.format_version < 3) {
    return Status::Ok();
  }
  DLCUDA_RETURN_IF_ERROR(RequireCheckpointValue(metadata.training_config, "seq_len",
                                                IntString(seq_len), "training_config"));
  DLCUDA_RETURN_IF_ERROR(RequireCheckpointValue(metadata.training_config, "d_model",
                                                IntString(d_model), "training_config"));
  DLCUDA_RETURN_IF_ERROR(RequireCheckpointValue(metadata.corpus_metadata, "bytes",
                                                UIntString(static_cast<uint64_t>(corpus.size())),
                                                "corpus"));
  DLCUDA_RETURN_IF_ERROR(RequireCheckpointValue(metadata.corpus_metadata, "hash_fnv1a64",
                                                HashString(corpus), "corpus"));
  DLCUDA_RETURN_IF_ERROR(
      RequireCheckpointValue(metadata.vocab_metadata, "size", IntString(vocab.size()), "vocab"));
  return RequireCheckpointValue(metadata.vocab_metadata, "hash_fnv1a64",
                                HashString(VocabBytes(vocab)), "vocab");
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

Status EnsureFloat2DTensor(Tensor *tensor, int64_t rows, int64_t cols, cudaStream_t stream) {
  if (tensor == nullptr) {
    return Status::InvalidArgument("EnsureFloat2DTensor received null tensor pointer");
  }
  if (tensor->defined() && tensor->dtype() == DType::kFloat32 && tensor->rank() == 2 &&
      tensor->dim(0) == rows && tensor->dim(1) == cols) {
    return Status::Ok();
  }
  auto allocated = Tensor::AllocateAsync({rows, cols}, DType::kFloat32, stream);
  if (!allocated.ok()) {
    return allocated.status();
  }
  *tensor = allocated.value();
  return Status::Ok();
}

__global__ void CausalConv1dForwardKernel(const float *input, const float *weight,
                                          const float *bias, float *output, int64_t seq_len,
                                          int64_t channels, int64_t kernel_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq_len * channels;
  if (idx >= total) {
    return;
  }

  int64_t time = idx / channels;
  int64_t out_ch = idx % channels;
  float sum = bias[out_ch];
  for (int64_t k = 0; k < kernel_width; ++k) {
    int64_t src_time = time - k;
    if (src_time < 0) {
      continue;
    }
    for (int64_t in_ch = 0; in_ch < channels; ++in_ch) {
      int64_t weight_idx = (k * channels + in_ch) * channels + out_ch;
      sum += input[src_time * channels + in_ch] * weight[weight_idx];
    }
  }
  output[idx] = sum;
}

__global__ void CausalConv1dBackwardInputKernel(const float *grad_output, const float *weight,
                                                float *grad_input, int64_t seq_len,
                                                int64_t channels, int64_t kernel_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq_len * channels;
  if (idx >= total) {
    return;
  }

  int64_t time = idx / channels;
  int64_t in_ch = idx % channels;
  float sum = 0.0f;
  int64_t max_future = time + kernel_width - 1;
  if (max_future >= seq_len) {
    max_future = seq_len - 1;
  }
  for (int64_t out_time = time; out_time <= max_future; ++out_time) {
    int64_t k = out_time - time;
    for (int64_t out_ch = 0; out_ch < channels; ++out_ch) {
      int64_t weight_idx = (k * channels + in_ch) * channels + out_ch;
      sum += grad_output[out_time * channels + out_ch] * weight[weight_idx];
    }
  }
  grad_input[idx] = sum;
}

__global__ void CausalConv1dBackwardWeightKernel(const float *input, const float *grad_output,
                                                 float *grad_weight, int64_t seq_len,
                                                 int64_t channels, int64_t kernel_width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = kernel_width * channels * channels;
  if (idx >= total) {
    return;
  }

  int64_t out_ch = idx % channels;
  int64_t in_ch = (idx / channels) % channels;
  int64_t k = idx / (channels * channels);
  float sum = 0.0f;
  for (int64_t time = k; time < seq_len; ++time) {
    sum += input[(time - k) * channels + in_ch] * grad_output[time * channels + out_ch];
  }
  grad_weight[idx] = sum;
}

__global__ void CausalConv1dBackwardBiasKernel(const float *grad_output, float *grad_bias,
                                               int64_t seq_len, int64_t channels) {
  int64_t out_ch = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_ch >= channels) {
    return;
  }

  float sum = 0.0f;
  for (int64_t time = 0; time < seq_len; ++time) {
    sum += grad_output[time * channels + out_ch];
  }
  grad_bias[out_ch] = sum;
}

class CausalConv1d : public Module {
public:
  CausalConv1d(int64_t channels, int64_t kernel_width, RuntimeContext &ctx)
      : channels_(channels), kernel_width_(kernel_width) {
    if (channels_ <= 0 || kernel_width_ <= 0) {
      init_status_ = Status::InvalidArgument("CausalConv1d dimensions must be positive");
      return;
    }

    auto weight =
        Tensor::AllocateAsync({kernel_width_, channels_, channels_}, DType::kFloat32, ctx.stream());
    if (!weight.ok()) {
      init_status_ = weight.status();
      return;
    }
    auto bias = Tensor::AllocateAsync({channels_}, DType::kFloat32, ctx.stream());
    if (!bias.ok()) {
      init_status_ = bias.status();
      return;
    }
    auto grad_weight =
        Tensor::AllocateAsync({kernel_width_, channels_, channels_}, DType::kFloat32, ctx.stream());
    if (!grad_weight.ok()) {
      init_status_ = grad_weight.status();
      return;
    }
    auto grad_bias = Tensor::AllocateAsync({channels_}, DType::kFloat32, ctx.stream());
    if (!grad_bias.ok()) {
      init_status_ = grad_bias.status();
      return;
    }

    weight_ = weight.value();
    bias_ = bias.value();
    grad_weight_ = grad_weight.value();
    grad_bias_ = grad_bias.value();

    std::mt19937 rng(static_cast<uint32_t>(ctx.NextInitSeed()));
    std::normal_distribution<float> dist(
        0.0f, std::sqrt(2.0f / static_cast<float>(channels_ * kernel_width_)));
    std::vector<float> host_weight(static_cast<size_t>(kernel_width_ * channels_ * channels_));
    for (float &value : host_weight) {
      value = dist(rng);
    }

    init_status_ =
        weight_.CopyFromHost(host_weight.data(), host_weight.size() * sizeof(float), ctx.stream());
    if (!init_status_.ok()) {
      return;
    }
    init_status_ = bias_.FillZero(ctx.stream());
    if (!init_status_.ok()) {
      return;
    }
    init_status_ = grad_weight_.FillZero(ctx.stream());
    if (!init_status_.ok()) {
      return;
    }
    init_status_ = grad_bias_.FillZero(ctx.stream());
  }

  Status Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) override {
    if (!init_status_.ok()) {
      return init_status_;
    }
    if (output == nullptr) {
      return Status::InvalidArgument("CausalConv1d::Forward output is null");
    }
    if (!input.defined()) {
      return Status::InvalidArgument("CausalConv1d input is undefined");
    }
    if (input.dtype() != DType::kFloat32) {
      return Status::InvalidArgument("CausalConv1d input must be float32");
    }
    if (input.rank() != 2) {
      return Status::InvalidArgument("CausalConv1d input must have rank 2");
    }
    if (input.dim(1) != channels_) {
      return Status::InvalidArgument("CausalConv1d input channel mismatch");
    }

    last_seq_len_ = input.dim(0);
    cached_input_ = input;
    DLCUDA_RETURN_IF_ERROR(
        EnsureFloat2DTensor(&forward_output_, last_seq_len_, channels_, ctx.stream()));

    int64_t total = last_seq_len_ * channels_;
    auto blocks = detail::BlocksForElements(total, kExampleThreads);
    if (!blocks.ok()) {
      return blocks.status();
    }
    if (blocks.value() > 0) {
      CausalConv1dForwardKernel<<<blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          input.data_as<float>(), weight_.data_as<float>(), bias_.data_as<float>(),
          forward_output_.data_as<float>(), last_seq_len_, channels_, kernel_width_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dForwardKernel"));
    }

    *output = forward_output_;
    return Status::Ok();
  }

  Status Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) override {
    if (!init_status_.ok()) {
      return init_status_;
    }
    if (!grad_output.defined()) {
      return Status::InvalidArgument("CausalConv1d grad_output is undefined");
    }
    if (grad_output.dtype() != DType::kFloat32) {
      return Status::InvalidArgument("CausalConv1d grad_output must be float32");
    }
    if (grad_output.rank() != 2 || grad_output.dim(0) != last_seq_len_ ||
        grad_output.dim(1) != channels_) {
      return Status::InvalidArgument("CausalConv1d grad_output shape mismatch");
    }
    if (!cached_input_.defined()) {
      return Status::RuntimeError("CausalConv1d backward called before forward");
    }

    if (grad_input != nullptr) {
      DLCUDA_RETURN_IF_ERROR(
          EnsureFloat2DTensor(&backward_output_, last_seq_len_, channels_, ctx.stream()));
    }

    int64_t input_total = last_seq_len_ * channels_;
    auto input_blocks = detail::BlocksForElements(input_total, kExampleThreads);
    if (!input_blocks.ok()) {
      return input_blocks.status();
    }
    if (grad_input != nullptr && input_blocks.value() > 0) {
      CausalConv1dBackwardInputKernel<<<input_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          grad_output.data_as<float>(), weight_.data_as<float>(), backward_output_.data_as<float>(),
          last_seq_len_, channels_, kernel_width_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dBackwardInputKernel"));
    }

    int64_t weight_total = kernel_width_ * channels_ * channels_;
    auto weight_blocks = detail::BlocksForElements(weight_total, kExampleThreads);
    if (!weight_blocks.ok()) {
      return weight_blocks.status();
    }
    if (weight_blocks.value() > 0) {
      CausalConv1dBackwardWeightKernel<<<weight_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          cached_input_.data_as<float>(), grad_output.data_as<float>(),
          grad_weight_.data_as<float>(), last_seq_len_, channels_, kernel_width_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dBackwardWeightKernel"));
    }

    auto bias_blocks = detail::BlocksForElements(channels_, kExampleThreads);
    if (!bias_blocks.ok()) {
      return bias_blocks.status();
    }
    if (bias_blocks.value() > 0) {
      CausalConv1dBackwardBiasKernel<<<bias_blocks.value(), kExampleThreads, 0, ctx.stream()>>>(
          grad_output.data_as<float>(), grad_bias_.data_as<float>(), last_seq_len_, channels_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("CausalConv1dBackwardBiasKernel"));
    }

    if (grad_input != nullptr) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }

  void AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) override {
    if (out == nullptr) {
      return;
    }
    std::string base = prefix.empty() ? std::string() : prefix + ".";
    out->push_back(ParameterRef{base + "weight", &weight_, &grad_weight_});
    out->push_back(ParameterRef{base + "bias", &bias_, &grad_bias_});
  }

private:
  Status init_status_;
  int64_t channels_ = 0;
  int64_t kernel_width_ = 0;
  int64_t last_seq_len_ = 0;
  Tensor weight_;
  Tensor bias_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  Tensor cached_input_;
  Tensor forward_output_;
  Tensor backward_output_;
};

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
                          Tensor *input_ids, Tensor *target_ids, int seq_len, int64_t offset) {
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

int64_t EvaluationOffset(const WindowSplit &split, int64_t index, int64_t windows) {
  if (windows <= 1 || split.count <= 1) {
    return split.begin + split.count / 2;
  }
  int64_t last = split.count - 1;
  return split.begin + (index * last) / (windows - 1);
}

Result<EvaluationSummary> EvaluateCharSplit(RuntimeContext &ctx, Sequential &model,
                                            const Tensor &encoded_corpus_device, Tensor *input_ids,
                                            Tensor *target_ids, int seq_len,
                                            const WindowSplit &split, int requested_windows) {
  if (input_ids == nullptr || target_ids == nullptr) {
    return Status::InvalidArgument("EvaluateCharSplit received null tensor pointer");
  }
  if (split.count <= 0) {
    return Status::InvalidArgument("EvaluateCharSplit requires a non-empty split");
  }
  if (requested_windows <= 0) {
    return Status::InvalidArgument("requested evaluation windows must be > 0");
  }

  int64_t windows = std::min<int64_t>(split.count, requested_windows);
  double loss_sum = 0.0;
  double accuracy_sum = 0.0;
  Tensor logits;
  for (int64_t i = 0; i < windows; ++i) {
    int64_t offset = EvaluationOffset(split, i, windows);
    DLCUDA_RETURN_IF_ERROR(
        FillTrainingWindow(ctx, encoded_corpus_device, input_ids, target_ids, seq_len, offset));
    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, *input_ids, &logits));
    auto metrics_result = CategoricalCrossEntropyMetricsFromLogits(ctx, *target_ids, logits);
    if (!metrics_result.ok()) {
      return metrics_result.status();
    }
    loss_sum += static_cast<double>(metrics_result.value().loss);
    accuracy_sum += static_cast<double>(metrics_result.value().accuracy);
  }

  EvaluationSummary summary;
  summary.windows = windows;
  summary.metrics.loss = static_cast<float>(loss_sum / static_cast<double>(windows));
  summary.metrics.accuracy = static_cast<float>(accuracy_sum / static_cast<double>(windows));
  return summary;
}

Status BuildCharModel(Sequential *model, RuntimeContext &ctx, int vocab_size, int d_model) {
  if (model == nullptr) {
    return Status::InvalidArgument("BuildCharModel requires a model pointer");
  }
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Embedding>(vocab_size, d_model, ctx)));
  DLCUDA_RETURN_IF_ERROR(
      model->Add(std::make_unique<CausalConv1d>(d_model, kCharContextWidth, ctx)));
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<GELU>()));
  DLCUDA_RETURN_IF_ERROR(model->Add(std::make_unique<Linear>(d_model, vocab_size, ctx)));
  return Status::Ok();
}

Result<std::vector<int32_t>> BuildGenerationContext(const CharVocab &vocab,
                                                    const std::string &corpus,
                                                    const std::string &prompt, int seq_len,
                                                    std::string *generated) {
  if (generated == nullptr) {
    return Status::InvalidArgument("BuildGenerationContext requires a generated output pointer");
  }
  DLCUDA_RETURN_IF_ERROR(ValidateCorpusWindow(corpus.size(), seq_len, "Generation"));

  std::vector<int32_t> context(static_cast<size_t>(seq_len));
  generated->clear();

  if (prompt.empty()) {
    generated->reserve(static_cast<size_t>(seq_len));
    for (int i = 0; i < seq_len; ++i) {
      int id = vocab.Encode(corpus[static_cast<size_t>(i)]);
      if (id < 0) {
        return Status::InvalidArgument("Generation corpus contains a character outside the vocab");
      }
      context[static_cast<size_t>(i)] = id;
      generated->push_back(vocab.Decode(id));
    }
    return context;
  }

  for (char ch : prompt) {
    if (vocab.Encode(ch) < 0) {
      return Status::InvalidArgument("Prompt contains a character outside the checkpoint vocab");
    }
  }

  if (prompt.size() >= static_cast<size_t>(seq_len)) {
    size_t start = prompt.size() - static_cast<size_t>(seq_len);
    for (int i = 0; i < seq_len; ++i) {
      context[static_cast<size_t>(i)] = vocab.Encode(prompt[start + static_cast<size_t>(i)]);
    }
  } else {
    size_t pad = static_cast<size_t>(seq_len) - prompt.size();
    for (size_t i = 0; i < pad; ++i) {
      int id = vocab.Encode(corpus[i]);
      if (id < 0) {
        return Status::InvalidArgument("Generation corpus contains a character outside the vocab");
      }
      context[i] = id;
    }
    for (size_t i = 0; i < prompt.size(); ++i) {
      context[pad + i] = vocab.Encode(prompt[i]);
    }
  }

  *generated = prompt;
  return context;
}

Result<std::string> GenerateText(RuntimeContext &ctx, Sequential &model, const CharVocab &vocab,
                                 const std::string &corpus, int seq_len, int gen_len,
                                 float temperature, float top_p, uint64_t sample_seed,
                                 const std::string &prompt) {
  if (gen_len < 0) {
    return Status::InvalidArgument("gen_len must be >= 0");
  }

  std::string generated;
  auto context_result = BuildGenerationContext(vocab, corpus, prompt, seq_len, &generated);
  if (!context_result.ok()) {
    return context_result.status();
  }
  std::vector<int32_t> context = context_result.value();

  auto context_tensor = Tensor::AllocateAsync({seq_len}, DType::kInt32, ctx.stream());
  if (!context_tensor.ok()) {
    return context_tensor.status();
  }
  auto input_ids = context_tensor.value();
  DLCUDA_RETURN_IF_ERROR(
      input_ids.CopyFromHost(context.data(), context.size() * sizeof(int32_t), ctx.stream()));

  size_t reserve_size = generated.size();
  size_t gen_len_size = static_cast<size_t>(gen_len);
  if (gen_len_size > std::numeric_limits<size_t>::max() - reserve_size) {
    return Status::InvalidArgument("generated text length is too large");
  }
  generated.reserve(reserve_size + gen_len_size);

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

  Tensor predictions;
  Tensor loss_grad;
  for (int64_t epoch = start_epoch; epoch < end_epoch; ++epoch) {
    DLCUDA_RETURN_IF_ERROR(optimizer.ZeroGrad(ctx, params));

    DLCUDA_RETURN_IF_ERROR(model.Forward(ctx, x, &predictions));
    bool should_log = ((epoch - start_epoch) % cfg.print_every) == 0;
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
      std::printf("Epoch %4lld | BCE: %.6f | GradNorm: %.4f\n", static_cast<long long>(epoch),
                  loss_value, grad_norm);
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

Result<std::string> SampleChar(const SampleCharConfig &cfg) {
  DLCUDA_RETURN_IF_ERROR(ValidateSampleCharConfig(cfg));

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
  DLCUDA_RETURN_IF_ERROR(ValidateCorpusWindow(corpus.size(), cfg.seq_len, "Sampling"));

  RuntimeContext ctx(OptionsFromCharConfig(cfg.use_cublas, cfg.tf32, cfg.seed));
  DLCUDA_RETURN_IF_ERROR(ctx.Initialize());

  Sequential model;
  DLCUDA_RETURN_IF_ERROR(BuildCharModel(&model, ctx, vocab.size(), cfg.d_model));
  CheckpointMetadata checkpoint_metadata;
  DLCUDA_RETURN_IF_ERROR(LoadCheckpoint(ctx, cfg.checkpoint_path, kCharModelName,
                                        model.parameters(), &checkpoint_metadata));
  DLCUDA_RETURN_IF_ERROR(
      ValidateCharCheckpointMetadata(checkpoint_metadata, cfg.seq_len, cfg.d_model, corpus, vocab));

  return GenerateText(ctx, model, vocab, corpus, cfg.seq_len, cfg.gen_len, cfg.temperature,
                      cfg.top_p, cfg.sample_seed, cfg.prompt);
}

} // namespace dlcuda
