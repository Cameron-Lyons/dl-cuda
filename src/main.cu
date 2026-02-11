#include "activation.cuh"
#include "layers.cuh"
#include "loss.cuh"
#include "optimizer.cuh"
#include "sequential.cuh"
#include <cstdio>

int main() {
  const int N = 4;
  const int IN_FEATURES = 2;
  const int HIDDEN = 4;
  const int OUT_FEATURES = 1;
  const float LR = 0.01f;
  const int EPOCHS = 5000;

  float h_X[N * IN_FEATURES] = {0, 0, 0, 1, 1, 0, 1, 1};
  float h_Y[N * OUT_FEATURES] = {0, 1, 1, 0};

  float *d_X, *d_Y, *d_pred, *d_loss_grad;
  CUDA_CHECK(cudaMalloc(&d_X, N * IN_FEATURES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y, N * OUT_FEATURES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pred, N * OUT_FEATURES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss_grad, N * OUT_FEATURES * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_X, h_X, N * IN_FEATURES * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * OUT_FEATURES * sizeof(float),
                         cudaMemcpyHostToDevice));

  LinearLayer layer1(N, IN_FEATURES, HIDDEN);
  ReLUActivation relu1(N * HIDDEN);
  LinearLayer layer2(N, HIDDEN, OUT_FEATURES);

  Sequential model;
  model.add(&layer1);
  model.add(&relu1);
  model.add(&layer2);

  AdamOptimizer adam;
  model.set_optimizer(&adam);

  float *d_input_grad;
  CUDA_CHECK(cudaMalloc(&d_input_grad, N * IN_FEATURES * sizeof(float)));

  float *d_error;
  CUDA_CHECK(cudaMalloc(&d_error, N * sizeof(float)));

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    model.forward(d_X, d_pred);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (epoch % 500 == 0) {
      dim3 loss_blocks((N + 255) / 256);
      squaredErrorKernel<<<loss_blocks, 256>>>(d_Y, d_pred, d_error, N);
      CUDA_CHECK(cudaDeviceSynchronize());

      float h_error[N];
      CUDA_CHECK(cudaMemcpy(h_error, d_error, N * sizeof(float),
                             cudaMemcpyDeviceToHost));
      float total_loss = 0.0f;
      for (int i = 0; i < N; i++)
        total_loss += h_error[i];
      total_loss /= N;
      printf("Epoch %d, MSE Loss: %.6f\n", epoch, total_loss);
    }

    squaredErrorBackwardKernel<<<(N + 255) / 256, 256>>>(d_Y, d_pred,
                                                          d_loss_grad, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    model.backward(d_loss_grad, d_input_grad);
    CUDA_CHECK(cudaDeviceSynchronize());

    model.update_weights(LR);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  model.forward(d_X, d_pred);
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_pred[N * OUT_FEATURES];
  CUDA_CHECK(cudaMemcpy(h_pred, d_pred, N * OUT_FEATURES * sizeof(float),
                         cudaMemcpyDeviceToHost));

  printf("\nFinal predictions:\n");
  for (int i = 0; i < N; i++) {
    printf("  [%.0f, %.0f] -> %.4f (expected %.0f)\n", h_X[i * 2],
           h_X[i * 2 + 1], h_pred[i], h_Y[i]);
  }

  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_pred);
  cudaFree(d_loss_grad);
  cudaFree(d_input_grad);
  cudaFree(d_error);

  return 0;
}
