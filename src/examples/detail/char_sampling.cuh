#pragma once

#include "char_model.cuh"

namespace dlcuda {
namespace {

inline void ApplyTopP(std::vector<float> &probs, float p) {
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

inline int SampleFromWeights(const std::vector<float> &probs, std::mt19937 &rng) {
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

inline int SampleToken(const std::vector<float> &raw_probs, float temperature, float top_p,
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

inline Result<std::vector<int32_t>> BuildGenerationContext(const CharVocab &vocab,
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

inline Result<std::string> GenerateText(RuntimeContext &ctx, Sequential &model,
                                        const CharVocab &vocab, const std::string &corpus,
                                        int seq_len, int gen_len, float temperature, float top_p,
                                        uint64_t sample_seed, const std::string &prompt) {
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
} // namespace dlcuda
