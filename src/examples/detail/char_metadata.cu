#include "char_metadata.cuh"

namespace dlcuda::examples_detail {

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

const char kCharModelName[] = "char-causal-conv-lm";

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

} // namespace dlcuda::examples_detail
