#include "detail/char_sampling.cuh"

namespace dlcuda {

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
