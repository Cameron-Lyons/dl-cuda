#include "activation.cuh"
#include "embedding.cuh"
#include "layers.cuh"
#include "loss.cuh"
#include "optimizer.cuh"
#include "positional_encoding.cuh"
#include "sampling.cuh"
#include "scheduler.cuh"
#include "sequential.cuh"
#include "transformer.cuh"
#include "../include/dl_cuda/examples.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace dlcuda {

__global__ void argmaxKernel(const float *logits, int *result, int num_rows, int row_width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const float *row_ptr = logits + row * row_width;
    int best = 0;
    float best_val = row_ptr[0];
    for (int j = 1; j < row_width; j++) {
      if (row_ptr[j] > best_val) {
        best_val = row_ptr[j];
        best = j;
      }
    }
    result[row] = best;
  }
}

int run_char_lm(const CharLMConfig &cfg) {
  set_global_init_seed(cfg.init_seed);

  if (cfg.seq_len <= 0) {
    std::fprintf(stderr, "seq_len must be > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.batch_size != 1) {
    std::fprintf(stderr, "Only batch_size=1 is currently supported.\n");
    return EXIT_FAILURE;
  }
  if (cfg.d_model <= 0 || cfg.d_ff <= 0 || cfg.num_heads <= 0 || cfg.num_layers <= 0) {
    std::fprintf(stderr, "Model dimensions and layer counts must be > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.epochs < 0 || cfg.gen_len < 0) {
    std::fprintf(stderr, "epochs and gen_len must be >= 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.print_every <= 0) {
    std::fprintf(stderr, "print_every must be > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.lr_max <= 0.0f || cfg.lr_min < 0.0f || cfg.lr_min > cfg.lr_max) {
    std::fprintf(stderr, "Learning rates must satisfy 0 <= lr_min <= lr_max and lr_max > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.warmup_steps < 0) {
    std::fprintf(stderr, "warmup_steps must be >= 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.grad_clip <= 0.0f) {
    std::fprintf(stderr, "grad_clip must be > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.temperature <= 0.0f) {
    std::fprintf(stderr, "temperature must be > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.top_p <= 0.0f || cfg.top_p > 1.0f) {
    std::fprintf(stderr, "top_p must be in (0, 1].\n");
    return EXIT_FAILURE;
  }

  const std::string text = "To be, or not to be, that is the question. "
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

  std::set<char> char_set(text.begin(), text.end());
  std::vector<char> vocab(char_set.begin(), char_set.end());
  std::sort(vocab.begin(), vocab.end());
  int vocab_size = static_cast<int>(vocab.size());

  std::vector<int> char_to_id(256, 0);
  std::vector<char> id_to_char(vocab_size);
  for (int i = 0; i < vocab_size; i++) {
    char_to_id[static_cast<unsigned char>(vocab[i])] = i;
    id_to_char[i] = vocab[i];
  }

  int num_tokens = cfg.seq_len;
  int text_len = static_cast<int>(text.size());
  if (text_len <= cfg.seq_len + 1) {
    std::fprintf(stderr, "Dataset text must be longer than seq_len + 1.\n");
    return EXIT_FAILURE;
  }

  std::vector<int> h_input_ids(num_tokens);
  std::vector<int> h_target_ids(num_tokens);
  for (int i = 0; i < num_tokens; i++) {
    h_input_ids[i] = char_to_id[static_cast<unsigned char>(text[i % text_len])];
    h_target_ids[i] = char_to_id[static_cast<unsigned char>(text[(i + 1) % text_len])];
  }

  EmbeddingLayer embedding(vocab_size, cfg.d_model, num_tokens);
  PositionalEncoding pos_enc(cfg.batch_size, cfg.seq_len, cfg.d_model);
  TransformerStack transformer(cfg.num_layers, cfg.batch_size, cfg.seq_len, cfg.d_model,
                               cfg.num_heads, 0.0f, cfg.d_ff, true);
  LinearLayer output_proj(num_tokens, cfg.d_model, vocab_size);
  SoftmaxActivation softmax(num_tokens, vocab_size);

  Sequential model;
  model.add(&embedding);
  model.add(&pos_enc);
  model.add(&transformer);
  model.add(&output_proj);
  model.add(&softmax);

  AdamWOptimizer adamw(0.9f, 0.999f, 1e-8f, 0.01f);
  model.set_optimizer(&adamw);

  CosineAnnealingScheduler cosine_sched(cfg.lr_max, cfg.lr_min, cfg.epochs);
  WarmupScheduler scheduler(cfg.lr_max, cfg.warmup_steps, &cosine_sched);

  int out_total = num_tokens * vocab_size;
  float *d_pred = nullptr;
  float *d_loss_grad = nullptr;
  float *d_error = nullptr;
  float *d_input_grad = nullptr;
  float *d_dummy_input = nullptr;
  float *d_gen_pred = nullptr;
  int *d_target_ids = nullptr;
  int *d_pred_ids = nullptr;
  auto cleanup = [&]() {
    if (d_pred)
      cudaFree(d_pred);
    if (d_loss_grad)
      cudaFree(d_loss_grad);
    if (d_error)
      cudaFree(d_error);
    if (d_input_grad)
      cudaFree(d_input_grad);
    if (d_target_ids)
      cudaFree(d_target_ids);
    if (d_pred_ids)
      cudaFree(d_pred_ids);
    if (d_dummy_input)
      cudaFree(d_dummy_input);
    if (d_gen_pred)
      cudaFree(d_gen_pred);
  };

  if (cfg.load_weights) {
    if (!model.load_weights(cfg.weights_path)) {
      std::fprintf(stderr, "Failed to load model weights from %s\n", cfg.weights_path.c_str());
      return EXIT_FAILURE;
    }
    std::printf("Loaded weights from %s\n", cfg.weights_path.c_str());
  }

  CUDA_CHECK(cudaMalloc(&d_pred, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss_grad, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_error, num_tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input_grad, num_tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target_ids, num_tokens * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pred_ids, num_tokens * sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_dummy_input, num_tokens * sizeof(float)));

  std::printf("Char-level LM | vocab=%d, seq_len=%d, d_model=%d, d_ff=%d, "
              "heads=%d, layers=%d\n",
              vocab_size, cfg.seq_len, cfg.d_model, cfg.d_ff, cfg.num_heads, cfg.num_layers);
  std::printf("Optimizer: AdamW (wd=0.01) | Grad clip: %.1f | Sampling: "
              "temp=%.2f, top_p=%.2f\n",
              cfg.grad_clip, cfg.temperature, cfg.top_p);
  std::printf("Training on %d chars for %d epochs\n\n", text_len, cfg.epochs);

  std::mt19937 offset_rng(static_cast<uint32_t>(cfg.init_seed));
  std::vector<float> h_error(num_tokens);
  std::vector<int> h_pred_ids(num_tokens);

  auto train_start = std::chrono::steady_clock::now();

  for (int epoch = 0; epoch < cfg.epochs; epoch++) {
    int max_offset = text_len - cfg.seq_len - 1;
    int offset = static_cast<int>(offset_rng() % (max_offset + 1));
    for (int i = 0; i < num_tokens; i++) {
      h_input_ids[i] = char_to_id[static_cast<unsigned char>(text[offset + i])];
      h_target_ids[i] = char_to_id[static_cast<unsigned char>(text[offset + i + 1])];
    }

    embedding.set_token_ids(h_input_ids.data());
    CUDA_CHECK(cudaMemcpy(d_target_ids, h_target_ids.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    model.forward(d_dummy_input, d_pred);
    float lr = scheduler.get_lr(epoch);

    if (epoch % cfg.print_every == 0) {
      computeCategoricalCrossEntropyFromIds(d_target_ids, d_pred, d_error, num_tokens, vocab_size);
      CUDA_CHECK(cudaGetLastError());

      CUDA_CHECK(
          cudaMemcpy(h_error.data(), d_error, num_tokens * sizeof(float), cudaMemcpyDeviceToHost));
      float total_loss = 0.0f;
      for (int i = 0; i < num_tokens; i++) {
        total_loss += h_error[i];
      }
      total_loss /= num_tokens;
      float perplexity = expf(total_loss);

      argmaxKernel<<<(num_tokens + 255) / 256, 256>>>(d_pred, d_pred_ids, num_tokens, vocab_size);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(h_pred_ids.data(), d_pred_ids, num_tokens * sizeof(int),
                            cudaMemcpyDeviceToHost));
      int correct = 0;
      for (int i = 0; i < num_tokens; i++) {
        if (h_pred_ids[i] == h_target_ids[i])
          correct++;
      }
      float accuracy = 100.0f * correct / num_tokens;

      std::printf("Epoch %4d | Loss: %.4f | PPL: %7.2f | Acc: %5.1f%% | LR: "
                  "%.6f\n",
                  epoch, total_loss, perplexity, accuracy, lr);
    }

    computeCategoricalCrossEntropyBackwardFromIds(d_target_ids, d_pred, d_loss_grad, num_tokens,
                                                  vocab_size);
    model.backward(d_loss_grad, d_input_grad);
    model.clip_grad_norm(cfg.grad_clip);

    model.update_weights(lr);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto train_end = std::chrono::steady_clock::now();
  if (cfg.epochs > 0) {
    double train_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(train_end - train_start).count();
    double tokens = static_cast<double>(cfg.epochs) * cfg.seq_len;
    double tok_per_sec = train_sec > 0.0 ? tokens / train_sec : 0.0;
    std::printf("Training throughput: %.2f tokens/s (%.3f s)\n", tok_per_sec, train_sec);
  }

  if (cfg.save_weights) {
    if (!model.save_weights(cfg.weights_path)) {
      std::fprintf(stderr, "Failed to save model weights to %s\n", cfg.weights_path.c_str());
      cleanup();
      return EXIT_FAILURE;
    }
    std::printf("\nWeights saved to %s\n", cfg.weights_path.c_str());
  }

  std::printf("\nGenerating text (temp=%.2f, top_p=%.2f, %d chars):\n", cfg.temperature, cfg.top_p,
              cfg.gen_len);

  std::mt19937 sample_rng(static_cast<uint32_t>(cfg.sample_seed));

  std::vector<int> context(cfg.seq_len);
  for (int i = 0; i < cfg.seq_len; i++) {
    context[i] = char_to_id[static_cast<unsigned char>(text[i])];
  }

  std::string generated;
  for (int i = 0; i < cfg.seq_len; i++) {
    generated += id_to_char[context[i]];
  }

  CUDA_CHECK(cudaMalloc(&d_gen_pred, num_tokens * vocab_size * sizeof(float)));
  std::vector<float> h_probs(vocab_size);

  for (int step = 0; step < cfg.gen_len; step++) {
    embedding.set_token_ids(context.data());

    model.forward(d_dummy_input, d_gen_pred);

    float *last_row = d_gen_pred + (num_tokens - 1) * vocab_size;
    CUDA_CHECK(
        cudaMemcpy(h_probs.data(), last_row, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    int next_id = sampleWithStrategy(h_probs, cfg.temperature, 0, cfg.top_p, sample_rng);
    generated += id_to_char[next_id];
    std::move(context.begin() + 1, context.end(), context.begin());
    context.back() = next_id;
  }

  std::printf("  \"%s\"\n", generated.c_str());

  cleanup();

  return 0;
}

int run_xor(const XorConfig &cfg) {
  set_global_init_seed(cfg.init_seed);

  if (cfg.epochs < 0) {
    std::fprintf(stderr, "epochs must be >= 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.print_every <= 0) {
    std::fprintf(stderr, "print_every must be > 0.\n");
    return EXIT_FAILURE;
  }
  if (cfg.lr <= 0.0f) {
    std::fprintf(stderr, "lr must be > 0.\n");
    return EXIT_FAILURE;
  }

  constexpr int N = 4;
  constexpr int IN = 2;
  constexpr int H = 8;
  constexpr int OUT = 1;

  std::vector<float> h_x = {
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
  };
  std::vector<float> h_y = {0.0f, 1.0f, 1.0f, 0.0f};

  LinearLayer fc1(N, IN, H);
  ReLUActivation relu(N * H);
  LinearLayer fc2(N, H, OUT);
  SigmoidActivation sigmoid(N * OUT);

  Sequential model;
  model.add(&fc1);
  model.add(&relu);
  model.add(&fc2);
  model.add(&sigmoid);

  AdamOptimizer adam(0.9f, 0.999f, 1e-8f);
  model.set_optimizer(&adam);

  float *d_x = nullptr;
  float *d_pred = nullptr;
  float *d_y = nullptr;
  float *d_loss = nullptr;
  float *d_loss_grad = nullptr;
  float *d_input_grad = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, N * IN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pred, N * OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N * OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss_grad, N * OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input_grad, N * IN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * IN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * OUT * sizeof(float), cudaMemcpyHostToDevice));

  std::printf("XOR | epochs=%d lr=%.4f\n", cfg.epochs, cfg.lr);
  for (int epoch = 0; epoch < cfg.epochs; epoch++) {
    model.forward(d_x, d_pred);

    dim3 blocks((N + 255) / 256);
    computeLoss(d_y, d_pred, d_loss, N, 1, BINARY_CROSS_ENTROPY, blocks, dim3(256));

    if (epoch % cfg.print_every == 0) {
      std::vector<float> h_loss(N);
      CUDA_CHECK(cudaMemcpy(h_loss.data(), d_loss, N * sizeof(float), cudaMemcpyDeviceToHost));
      float avg = 0.0f;
      for (float v : h_loss)
        avg += v;
      avg /= N;
      std::printf("Epoch %4d | BCE: %.6f\n", epoch, avg);
    }

    computeLossBackward(d_y, d_pred, d_loss_grad, N, 1, BINARY_CROSS_ENTROPY);
    model.backward(d_loss_grad, d_input_grad);
    model.update_weights(cfg.lr);
  }

  model.forward(d_x, d_pred);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_pred(N);
  CUDA_CHECK(cudaMemcpy(h_pred.data(), d_pred, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Final predictions:\n");
  std::printf("  [0, 0] -> %.4f (expected 0)\n", h_pred[0]);
  std::printf("  [0, 1] -> %.4f (expected 1)\n", h_pred[1]);
  std::printf("  [1, 0] -> %.4f (expected 1)\n", h_pred[2]);
  std::printf("  [1, 1] -> %.4f (expected 0)\n", h_pred[3]);

  cudaFree(d_x);
  cudaFree(d_pred);
  cudaFree(d_y);
  cudaFree(d_loss);
  cudaFree(d_loss_grad);
  cudaFree(d_input_grad);

  return 0;
}

} // namespace dlcuda
