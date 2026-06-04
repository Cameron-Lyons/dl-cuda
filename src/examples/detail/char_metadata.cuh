#pragma once

#include "common.cuh"

namespace dlcuda::examples_detail {

extern const char kCharModelName[];

Result<std::string> LoadTextFile(const std::string &path);
Result<std::string> LoadCharCorpus(const std::string &data_path);
std::vector<CheckpointKeyValue> BuildCharTrainingConfig(const TrainCharConfig &cfg);
std::vector<CheckpointKeyValue> BuildCorpusMetadata(const TrainCharConfig &cfg,
                                                    const std::string &corpus);
std::vector<CheckpointKeyValue> BuildVocabMetadata(const CharVocab &vocab);

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

Result<int64_t> FractionToWindowCount(int64_t total_windows, float fraction, const char *name);
Result<CharDatasetSplits> BuildCharDatasetSplits(size_t corpus_size, const TrainCharConfig &cfg);
std::string SerializeMetricHistory(const std::vector<MetricSnapshot> &history);
std::vector<CheckpointKeyValue> BuildCharExtraMetadata(const CharDatasetSplits &splits,
                                                       const std::vector<MetricSnapshot> &history,
                                                       const CharRunSummary &summary);
std::string BestCheckpointPath(const TrainCharConfig &cfg);
CheckpointMetadata BuildCharCheckpointMetadata(const TrainCharConfig &cfg,
                                               const std::string &corpus, const CharVocab &vocab,
                                               const std::mt19937 &offset_rng,
                                               int64_t completed_epoch, int64_t step,
                                               const CharDatasetSplits &splits,
                                               const std::vector<MetricSnapshot> &history,
                                               const CharRunSummary &summary);
Status ValidateCharCheckpointMetadata(const CheckpointMetadata &metadata, int seq_len, int d_model,
                                      const std::string &corpus, const CharVocab &vocab);

} // namespace dlcuda::examples_detail

namespace dlcuda {

using examples_detail::BestCheckpointPath;
using examples_detail::BuildCharCheckpointMetadata;
using examples_detail::BuildCharDatasetSplits;
using examples_detail::BuildCharExtraMetadata;
using examples_detail::BuildCharTrainingConfig;
using examples_detail::BuildCorpusMetadata;
using examples_detail::BuildVocabMetadata;
using examples_detail::CharDatasetSplits;
using examples_detail::CharRunSummary;
using examples_detail::EvaluationSummary;
using examples_detail::FractionToWindowCount;
using examples_detail::kCharModelName;
using examples_detail::LoadCharCorpus;
using examples_detail::LoadTextFile;
using examples_detail::MetricSnapshot;
using examples_detail::SerializeMetricHistory;
using examples_detail::ValidateCharCheckpointMetadata;
using examples_detail::WindowSplit;

} // namespace dlcuda
