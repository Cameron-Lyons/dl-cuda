#include "activation.cuh"
#include "embedding.cuh"
#include "layers.cuh"
#include "loss.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool nearly_equal(float a, float b, float tol = 1e-4f) {
  return std::fabs(a - b) <= tol;
}

float relative_error(float a, float b) {
  float denom = std::max(1e-6f, std::fabs(a) + std::fabs(b));
  return std::fabs(a - b) / denom;
}

bool has_cuda_device() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

bool test_linear_forward_backward() {
  set_global_init_seed(1);

  LinearLayer layer(2, 3, 2);
  std::vector<float> w = {1.0f, -1.0f, 2.0f, 0.5f, -0.25f, 1.5f};
  std::vector<float> b = {0.1f, -0.2f};
  CUDA_CHECK(cudaMemcpy(layer.get_weights(), w.data(), w.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(layer.get_bias(), b.data(), b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> x = {1.0f, 2.0f, -1.0f,
                          0.5f, -3.0f, 2.0f};
  std::vector<float> dout = {0.3f, -0.7f,
                             1.2f, 0.4f};

  float *d_x = nullptr, *d_y = nullptr, *d_dout = nullptr, *d_dx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dout, dout.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dout, dout.data(), dout.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  layer.forward(d_x, d_y);
  layer.backward(d_dout, d_dx);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> y(4), dx(6), dw(w.size()), db(b.size());
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, dx.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dw.data(), layer.get_weight_grad(),
                        dw.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(db.data(), layer.get_bias_grad(),
                        db.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> y_expected(4, 0.0f);
  std::vector<float> dx_expected(6, 0.0f);
  std::vector<float> dw_expected(w.size(), 0.0f);
  std::vector<float> db_expected(b.size(), 0.0f);

  for (int n = 0; n < 2; n++) {
    for (int o = 0; o < 2; o++) {
      float sum = b[o];
      for (int i = 0; i < 3; i++) {
        sum += x[n * 3 + i] * w[i * 2 + o];
      }
      y_expected[n * 2 + o] = sum;
    }
  }

  for (int n = 0; n < 2; n++) {
    for (int i = 0; i < 3; i++) {
      float sum = 0.0f;
      for (int o = 0; o < 2; o++) {
        sum += dout[n * 2 + o] * w[i * 2 + o];
      }
      dx_expected[n * 3 + i] = sum;
    }
  }

  for (int i = 0; i < 3; i++) {
    for (int o = 0; o < 2; o++) {
      float sum = 0.0f;
      for (int n = 0; n < 2; n++) {
        sum += x[n * 3 + i] * dout[n * 2 + o];
      }
      dw_expected[i * 2 + o] = sum;
    }
  }
  for (int o = 0; o < 2; o++) {
    db_expected[o] = dout[o] + dout[2 + o];
  }

  bool ok = true;
  for (size_t i = 0; i < y.size(); i++)
    ok &= nearly_equal(y[i], y_expected[i]);
  for (size_t i = 0; i < dx.size(); i++)
    ok &= nearly_equal(dx[i], dx_expected[i]);
  for (size_t i = 0; i < dw.size(); i++)
    ok &= nearly_equal(dw[i], dw_expected[i]);
  for (size_t i = 0; i < db.size(); i++)
    ok &= nearly_equal(db[i], db_expected[i]);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_dout);
  cudaFree(d_dx);
  return ok;
}

float run_linear_loss(LinearLayer &layer, float *d_x, int out_size) {
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_y, out_size * sizeof(float)));
  layer.forward(d_x, d_y);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> y(out_size);
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, out_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_y);
  float loss = 0.0f;
  for (float v : y)
    loss += 0.5f * v * v;
  return loss;
}

bool test_linear_finite_difference() {
  set_global_init_seed(2);

  LinearLayer layer(2, 2, 2);
  std::vector<float> w = {0.3f, -0.2f, 0.7f, 0.1f};
  std::vector<float> b = {0.05f, -0.04f};
  CUDA_CHECK(cudaMemcpy(layer.get_weights(), w.data(), w.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(layer.get_bias(), b.data(), b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> x = {1.0f, 2.0f,
                          -1.0f, 0.5f};
  float *d_x = nullptr, *d_y = nullptr, *d_dout = nullptr, *d_dx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dout, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  layer.forward(d_x, d_y);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(d_dout, d_y, 4 * sizeof(float), cudaMemcpyDeviceToDevice));
  layer.backward(d_dout, d_dx);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> analytic(w.size());
  CUDA_CHECK(cudaMemcpy(analytic.data(), layer.get_weight_grad(),
                        analytic.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float eps = 1e-3f;
  bool ok = true;
  for (size_t i = 0; i < w.size(); i++) {
    std::vector<float> w_plus = w;
    std::vector<float> w_minus = w;
    w_plus[i] += eps;
    w_minus[i] -= eps;

    CUDA_CHECK(cudaMemcpy(layer.get_weights(), w_plus.data(), w_plus.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    float l_plus = run_linear_loss(layer, d_x, 4);

    CUDA_CHECK(cudaMemcpy(layer.get_weights(), w_minus.data(), w_minus.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    float l_minus = run_linear_loss(layer, d_x, 4);

    float numeric = (l_plus - l_minus) / (2.0f * eps);
    if (relative_error(numeric, analytic[i]) > 5e-2f) {
      std::fprintf(stderr,
                   "finite diff mismatch idx=%zu analytic=%f numeric=%f\n", i,
                   analytic[i], numeric);
      ok = false;
      break;
    }
  }

  CUDA_CHECK(cudaMemcpy(layer.get_weights(), w.data(), w.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_dout);
  cudaFree(d_dx);
  return ok;
}

bool test_embedding_forward_backward() {
  set_global_init_seed(3);

  EmbeddingLayer embedding(5, 3, 4);
  auto groups = embedding.get_param_groups();
  float *d_table = groups[0].params;
  float *d_grad = groups[0].grads;

  std::vector<float> table = {
      0, 1, 2,
      3, 4, 5,
      6, 7, 8,
      9, 10, 11,
      12, 13, 14,
  };
  CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<int> token_ids = {1, 3, 1, 4};
  embedding.set_token_ids(token_ids.data());

  std::vector<float> dout = {
      1, 1, 1,
      2, 2, 2,
      3, 3, 3,
      4, 4, 4,
  };

  float *d_out = nullptr, *d_dummy = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, dout.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dummy, 4 * sizeof(float)));

  embedding.forward(nullptr, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> out(dout.size());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(d_out, dout.data(), dout.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  embedding.backward(d_out, d_dummy);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> grad(table.size());
  CUDA_CHECK(cudaMemcpy(grad.data(), d_grad, grad.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<float> out_expected = {
      3, 4, 5,
      9, 10, 11,
      3, 4, 5,
      12, 13, 14,
  };
  std::vector<float> grad_expected(table.size(), 0.0f);
  for (int d = 0; d < 3; d++) {
    grad_expected[1 * 3 + d] = 1.0f + 3.0f;
    grad_expected[3 * 3 + d] = 2.0f;
    grad_expected[4 * 3 + d] = 4.0f;
  }

  bool ok = true;
  for (size_t i = 0; i < out.size(); i++)
    ok &= nearly_equal(out[i], out_expected[i]);
  for (size_t i = 0; i < grad.size(); i++)
    ok &= nearly_equal(grad[i], grad_expected[i]);

  cudaFree(d_out);
  cudaFree(d_dummy);
  return ok;
}

bool test_softmax_cross_entropy_grad() {
  SoftmaxActivation softmax(2, 3);

  std::vector<float> logits = {2.0f, 1.0f, 0.1f,
                               -1.0f, 0.0f, 3.0f};
  std::vector<int> targets = {2, 0};

  float *d_logits = nullptr, *d_probs = nullptr;
  int *d_targets = nullptr;
  float *d_dldp = nullptr, *d_dlogits = nullptr;
  CUDA_CHECK(cudaMalloc(&d_logits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probs, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_targets, targets.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dldp, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dlogits, logits.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  softmax.forward(d_logits, d_probs);
  computeCategoricalCrossEntropyBackwardFromIds(d_targets, d_probs, d_dldp, 2,
                                                3);
  softmax.backward(d_dldp, d_dlogits);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> probs(logits.size()), dlogits(logits.size());
  CUDA_CHECK(cudaMemcpy(probs.data(), d_probs, probs.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dlogits.data(), d_dlogits,
                        dlogits.size() * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 3; c++) {
      float expected = probs[r * 3 + c] / 2.0f;
      if (c == targets[r])
        expected -= 1.0f / 2.0f;
      ok &= nearly_equal(dlogits[r * 3 + c], expected, 2e-4f);
    }
  }

  cudaFree(d_logits);
  cudaFree(d_probs);
  cudaFree(d_targets);
  cudaFree(d_dldp);
  cudaFree(d_dlogits);
  return ok;
}

} // namespace

int main() {
  if (!has_cuda_device()) {
    std::printf("gpu_correctness_tests: SKIP (no CUDA device)\n");
    return 0;
  }

  if (!test_linear_forward_backward()) {
    std::fprintf(stderr, "test_linear_forward_backward failed\n");
    return 1;
  }
  if (!test_linear_finite_difference()) {
    std::fprintf(stderr, "test_linear_finite_difference failed\n");
    return 1;
  }
  if (!test_embedding_forward_backward()) {
    std::fprintf(stderr, "test_embedding_forward_backward failed\n");
    return 1;
  }
  if (!test_softmax_cross_entropy_grad()) {
    std::fprintf(stderr, "test_softmax_cross_entropy_grad failed\n");
    return 1;
  }

  std::printf("gpu_correctness_tests: PASS\n");
  return 0;
}
