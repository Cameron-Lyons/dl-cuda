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
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace dlcuda {

__global__ void argmaxKernel(const float *logits, int *result, int num_rows,
                             int row_width) {
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

  const std::string text =
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
    h_target_ids[i] =
        char_to_id[static_cast<unsigned char>(text[(i + 1) % text_len])];
  }

  EmbeddingLayer embedding(vocab_size, cfg.d_model, num_tokens);
  PositionalEncoding pos_enc(cfg.batch_size, cfg.seq_len, cfg.d_model);
  TransformerStack transformer(cfg.num_layers, cfg.batch_size, cfg.seq_len,
                               cfg.d_model, cfg.num_heads, 0.0f, cfg.d_ff,
                               true);
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
  int *d_target_ids = nullptr;
  int *d_pred_ids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pred, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss_grad, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_error, num_tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input_grad, num_tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target_ids, num_tokens * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pred_ids, num_tokens * sizeof(int)));

  float *d_dummy_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dummy_input, num_tokens * sizeof(float)));

  std::printf("Char-level LM | vocab=%d, seq_len=%d, d_model=%d, d_ff=%d, "
              "heads=%d, layers=%d\n",
              vocab_size, cfg.seq_len, cfg.d_model, cfg.d_ff, cfg.num_heads,
              cfg.num_layers);
  std::printf("Optimizer: AdamW (wd=0.01) | Grad clip: %.1f | Sampling: "
              "temp=%.2f, top_p=%.2f\n",
              cfg.grad_clip, cfg.temperature, cfg.top_p);
  std::printf("Training on %d chars for %d epochs\n\n", text_len,
              cfg.epochs);

  std::mt19937 offset_rng(static_cast<uint32_t>(cfg.init_seed));

  for (int epoch = 0; epoch < cfg.epochs; epoch++) {
    int max_offset = text_len - cfg.seq_len - 1;
    int offset = static_cast<int>(offset_rng() % (max_offset + 1));
    for (int i = 0; i < num_tokens; i++) {
      h_input_ids[i] =
          char_to_id[static_cast<unsigned char>(text[offset + i])];
      h_target_ids[i] =
          char_to_id[static_cast<unsigned char>(text[offset + i + 1])];
    }

    embedding.set_token_ids(h_input_ids.data());
    CUDA_CHECK(cudaMemcpy(d_target_ids, h_target_ids.data(),
                          num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    model.forward(d_dummy_input, d_pred);

    if (epoch % cfg.print_every == 0) {
      computeCategoricalCrossEntropyFromIds(d_target_ids, d_pred, d_error,
                                            num_tokens, vocab_size);
      CUDA_CHECK(cudaGetLastError());

      std::vector<float> h_error(num_tokens);
      CUDA_CHECK(cudaMemcpy(h_error.data(), d_error,
                            num_tokens * sizeof(float),
                            cudaMemcpyDeviceToHost));
      float total_loss = 0.0f;
      for (int i = 0; i < num_tokens; i++) {
        total_loss += h_error[i];
      }
      total_loss /= num_tokens;
      float perplexity = expf(total_loss);

      argmaxKernel<<<(num_tokens + 255) / 256, 256>>>(d_pred, d_pred_ids,
                                                       num_tokens, vocab_size);
      CUDA_CHECK(cudaGetLastError());
      std::vector<int> h_pred_ids(num_tokens);
      CUDA_CHECK(cudaMemcpy(h_pred_ids.data(), d_pred_ids,
                            num_tokens * sizeof(int), cudaMemcpyDeviceToHost));
      int correct = 0;
      for (int i = 0; i < num_tokens; i++) {
        if (h_pred_ids[i] == h_target_ids[i])
          correct++;
      }
      float accuracy = 100.0f * correct / num_tokens;

      float lr = scheduler.get_lr(epoch);
      std::printf("Epoch %4d | Loss: %.4f | PPL: %7.2f | Acc: %5.1f%% | LR: "
                  "%.6f\n",
                  epoch, total_loss, perplexity, accuracy, lr);
    }

    computeCategoricalCrossEntropyBackwardFromIds(d_target_ids, d_pred,
                                                  d_loss_grad, num_tokens,
                                                  vocab_size);
    model.backward(d_loss_grad, d_input_grad);
    model.clip_grad_norm(cfg.grad_clip);

    float lr = scheduler.get_lr(epoch);
    model.update_weights(lr);
  }

  if (!model.save_weights("model.bin")) {
    std::fprintf(stderr, "Failed to save model weights to model.bin\n");
    return EXIT_FAILURE;
  }
  std::printf("\nWeights saved to model.bin\n");

  std::printf("\nGenerating text (temp=%.2f, top_p=%.2f, %d chars):\n",
              cfg.temperature, cfg.top_p, cfg.gen_len);

  std::mt19937 sample_rng(static_cast<uint32_t>(cfg.sample_seed));

  std::vector<int> gen_ids(cfg.seq_len);
  for (int i = 0; i < cfg.seq_len; i++) {
    gen_ids[i] = char_to_id[static_cast<unsigned char>(text[i])];
  }

  std::string generated;
  for (int i = 0; i < cfg.seq_len; i++) {
    generated += id_to_char[gen_ids[i]];
  }

  float *d_gen_pred = nullptr;
  CUDA_CHECK(cudaMalloc(&d_gen_pred, num_tokens * vocab_size * sizeof(float)));

  for (int step = 0; step < cfg.gen_len; step++) {
    std::vector<int> context(gen_ids.end() - cfg.seq_len, gen_ids.end());
    embedding.set_token_ids(context.data());

    model.forward(d_dummy_input, d_gen_pred);

    float *last_row = d_gen_pred + (num_tokens - 1) * vocab_size;
    std::vector<float> h_probs(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_probs.data(), last_row, vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int next_id =
        sampleWithStrategy(h_probs, cfg.temperature, 0, cfg.top_p, sample_rng);
    gen_ids.push_back(next_id);
    generated += id_to_char[next_id];
  }

  std::printf("  \"%s\"\n", generated.c_str());

  cudaFree(d_pred);
  cudaFree(d_loss_grad);
  cudaFree(d_error);
  cudaFree(d_input_grad);
  cudaFree(d_target_ids);
  cudaFree(d_pred_ids);
  cudaFree(d_dummy_input);
  cudaFree(d_gen_pred);

  return 0;
}

int run_xor(const XorConfig &cfg) {
  set_global_init_seed(cfg.init_seed);

  constexpr int N = 4;
  constexpr int IN = 2;
  constexpr int H = 8;
  constexpr int OUT = 1;

  std::vector<float> h_x = {
      0.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
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
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * IN * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * OUT * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::printf("XOR | epochs=%d lr=%.4f\n", cfg.epochs, cfg.lr);
  for (int epoch = 0; epoch < cfg.epochs; epoch++) {
    model.forward(d_x, d_pred);

    dim3 blocks((N + 255) / 256);
    computeLoss(d_y, d_pred, d_loss, N, 1, BINARY_CROSS_ENTROPY, blocks,
                dim3(256));

    if (epoch % cfg.print_every == 0) {
      std::vector<float> h_loss(N);
      CUDA_CHECK(cudaMemcpy(h_loss.data(), d_loss, N * sizeof(float),
                            cudaMemcpyDeviceToHost));
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

  std::vector<float> h_pred(N);
  CUDA_CHECK(cudaMemcpy(h_pred.data(), d_pred, N * sizeof(float),
                        cudaMemcpyDeviceToHost));
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
