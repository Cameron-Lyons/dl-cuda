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
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

__global__ void buildOneHotTargetsKernel(const int *token_ids, float *targets,
                                         int num_tokens, int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tokens * vocab_size) {
    int t = idx / vocab_size;
    int c = idx % vocab_size;
    targets[idx] = (c == token_ids[t]) ? 1.0f : 0.0f;
  }
}

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

int main() {
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

  const int SEQ_LEN = 64;
  const int BATCH_SIZE = 1;
  const int D_MODEL = 64;
  const int D_FF = 256;
  const int NUM_HEADS = 4;
  const int NUM_LAYERS = 3;
  const int EPOCHS = 800;
  const int PRINT_EVERY = 50;
  const int GEN_LEN = 200;
  const float GRAD_CLIP = 1.0f;
  const float TEMPERATURE = 0.8f;
  const float TOP_P = 0.9f;

  int num_tokens = SEQ_LEN;
  int text_len = static_cast<int>(text.size());

  std::vector<int> h_input_ids(num_tokens);
  std::vector<int> h_target_ids(num_tokens);
  for (int i = 0; i < num_tokens; i++) {
    h_input_ids[i] = char_to_id[static_cast<unsigned char>(text[i % text_len])];
    h_target_ids[i] =
        char_to_id[static_cast<unsigned char>(text[(i + 1) % text_len])];
  }

  EmbeddingLayer embedding(vocab_size, D_MODEL, num_tokens);
  PositionalEncoding pos_enc(BATCH_SIZE, SEQ_LEN, D_MODEL);
  TransformerStack transformer(NUM_LAYERS, BATCH_SIZE, SEQ_LEN, D_MODEL,
                               NUM_HEADS, 0.0f, D_FF, true);
  LinearLayer output_proj(num_tokens, D_MODEL, vocab_size);
  SoftmaxActivation softmax(num_tokens, vocab_size);

  Sequential model;
  model.add(&embedding);
  model.add(&pos_enc);
  model.add(&transformer);
  model.add(&output_proj);
  model.add(&softmax);

  AdamWOptimizer adamw(0.9f, 0.999f, 1e-8f, 0.01f);
  model.set_optimizer(&adamw);

  CosineAnnealingScheduler cosine_sched(3e-3f, 1e-5f, EPOCHS);
  WarmupScheduler scheduler(3e-3f, 50, &cosine_sched);

  int out_total = num_tokens * vocab_size;
  float *d_pred, *d_targets, *d_loss_grad, *d_error, *d_input_grad;
  int *d_target_ids, *d_pred_ids;
  CUDA_CHECK(cudaMalloc(&d_pred, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_targets, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss_grad, out_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_error, num_tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input_grad, num_tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target_ids, num_tokens * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pred_ids, num_tokens * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_target_ids, h_target_ids.data(),
                         num_tokens * sizeof(int), cudaMemcpyHostToDevice));

  int oh_total = num_tokens * vocab_size;
  int oh_blocks = (oh_total + 255) / 256;
  buildOneHotTargetsKernel<<<oh_blocks, 256>>>(d_target_ids, d_targets,
                                                num_tokens, vocab_size);
  CUDA_CHECK(cudaDeviceSynchronize());

  float *d_dummy_input;
  CUDA_CHECK(cudaMalloc(&d_dummy_input, num_tokens * sizeof(float)));

  printf("Char-level LM | vocab=%d, seq_len=%d, d_model=%d, d_ff=%d, "
         "heads=%d, layers=%d\n",
         vocab_size, SEQ_LEN, D_MODEL, D_FF, NUM_HEADS, NUM_LAYERS);
  printf("Optimizer: AdamW (wd=0.01) | Grad clip: %.1f | Sampling: "
         "temp=%.1f, top_p=%.1f\n",
         GRAD_CLIP, TEMPERATURE, TOP_P);
  printf("Training on %d chars of Shakespeare for %d epochs\n\n", text_len,
         EPOCHS);

  std::mt19937 offset_rng(42);

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    int max_offset = text_len - SEQ_LEN - 1;
    int offset = offset_rng() % max_offset;
    for (int i = 0; i < num_tokens; i++) {
      h_input_ids[i] =
          char_to_id[static_cast<unsigned char>(text[offset + i])];
      h_target_ids[i] =
          char_to_id[static_cast<unsigned char>(text[offset + i + 1])];
    }
    embedding.set_token_ids(h_input_ids.data());
    CUDA_CHECK(cudaMemcpy(d_target_ids, h_target_ids.data(),
                           num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    buildOneHotTargetsKernel<<<oh_blocks, 256>>>(d_target_ids, d_targets,
                                                  num_tokens, vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    model.forward(d_dummy_input, d_pred);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (epoch % PRINT_EVERY == 0) {
      dim3 loss_blocks((num_tokens + 255) / 256);
      categoricalCrossEntropyKernel<<<loss_blocks, 256>>>(
          d_targets, d_pred, d_error, num_tokens, vocab_size, 1e-10f);
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<float> h_error(num_tokens);
      CUDA_CHECK(cudaMemcpy(h_error.data(), d_error,
                             num_tokens * sizeof(float),
                             cudaMemcpyDeviceToHost));
      float total_loss = 0.0f;
      for (int i = 0; i < num_tokens; i++)
        total_loss += h_error[i];
      total_loss /= num_tokens;
      float perplexity = expf(total_loss);

      argmaxKernel<<<(num_tokens + 255) / 256, 256>>>(d_pred, d_pred_ids,
                                                       num_tokens, vocab_size);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<int> h_pred_ids(num_tokens);
      CUDA_CHECK(cudaMemcpy(h_pred_ids.data(), d_pred_ids,
                             num_tokens * sizeof(int),
                             cudaMemcpyDeviceToHost));
      int correct = 0;
      for (int i = 0; i < num_tokens; i++) {
        if (h_pred_ids[i] == h_target_ids[i])
          correct++;
      }
      float accuracy = 100.0f * correct / num_tokens;

      float lr = scheduler.get_lr(epoch);
      printf("Epoch %3d | Loss: %.4f | PPL: %7.2f | Acc: %5.1f%% | LR: "
             "%.6f\n",
             epoch, total_loss, perplexity, accuracy, lr);
    }

    computeLossBackward(d_targets, d_pred, d_loss_grad, num_tokens, vocab_size,
                        CATEGORICAL_CROSS_ENTROPY);

    model.backward(d_loss_grad, d_input_grad);
    CUDA_CHECK(cudaDeviceSynchronize());

    model.clip_grad_norm(GRAD_CLIP);

    float lr = scheduler.get_lr(epoch);
    model.update_weights(lr);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  model.save_weights("model.bin");
  printf("\nWeights saved to model.bin\n");

  printf("\nGenerating text (temp=%.1f, top_p=%.1f, %d chars):\n", TEMPERATURE,
         TOP_P, GEN_LEN);

  std::mt19937 sample_rng(123);

  std::vector<int> gen_ids(SEQ_LEN);
  for (int i = 0; i < SEQ_LEN; i++) {
    gen_ids[i] = char_to_id[static_cast<unsigned char>(text[i])];
  }

  std::string generated;
  for (int i = 0; i < SEQ_LEN; i++) {
    generated += id_to_char[gen_ids[i]];
  }

  float *d_gen_pred;
  CUDA_CHECK(cudaMalloc(&d_gen_pred, num_tokens * vocab_size * sizeof(float)));

  for (int step = 0; step < GEN_LEN; step++) {
    std::vector<int> context(gen_ids.end() - SEQ_LEN, gen_ids.end());
    embedding.set_token_ids(context.data());

    model.forward(d_dummy_input, d_gen_pred);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *last_row = d_gen_pred + (num_tokens - 1) * vocab_size;
    std::vector<float> h_probs(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_probs.data(), last_row,
                           vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

    int next_id = sampleWithStrategy(h_probs, TEMPERATURE, 0, TOP_P,
                                     sample_rng);
    gen_ids.push_back(next_id);
    generated += id_to_char[next_id];
  }

  printf("  \"%s\"\n", generated.c_str());

  cudaFree(d_pred);
  cudaFree(d_targets);
  cudaFree(d_loss_grad);
  cudaFree(d_error);
  cudaFree(d_input_grad);
  cudaFree(d_target_ids);
  cudaFree(d_pred_ids);
  cudaFree(d_dummy_input);
  cudaFree(d_gen_pred);

  return 0;
}
