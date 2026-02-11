#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct ParamGroup {
  float *params;
  float *grads;
  int size;
};

class Optimizer;

class Operation {
public:
  virtual void forward(float *d_input, float *d_output) = 0;
  virtual void backward(float *d_output_grad, float *d_input_grad) = 0;
  virtual void update_weights(float /*lr*/) {}
  virtual int input_size() const = 0;
  virtual int output_size() const = 0;
  virtual std::vector<ParamGroup> get_param_groups() { return {}; }
  virtual ~Operation() = default;
};

class Sequential {
private:
  std::vector<Operation *> operations;
  Optimizer *optimizer_ = nullptr;

public:
  void add(Operation *op) { operations.push_back(op); }

  void forward(float *d_input, float *d_output) {
    if (operations.empty())
      return;

    std::vector<float *> buffers;

    float *current_in = d_input;
    for (size_t i = 0; i < operations.size(); i++) {
      float *current_out;
      if (i == operations.size() - 1) {
        current_out = d_output;
      } else {
        CUDA_CHECK(cudaMalloc(&current_out,
                               operations[i]->output_size() * sizeof(float)));
        buffers.push_back(current_out);
      }
      operations[i]->forward(current_in, current_out);
      CUDA_CHECK(cudaDeviceSynchronize());
      current_in = current_out;
    }

    for (auto buf : buffers) {
      CUDA_CHECK(cudaFree(buf));
    }
  }

  void backward(float *d_output_grad, float *d_input_grad) {
    if (operations.empty())
      return;

    std::vector<float *> grad_buffers;

    float *current_grad = d_output_grad;
    for (int i = static_cast<int>(operations.size()) - 1; i >= 0; i--) {
      float *prev_grad;
      if (i == 0) {
        prev_grad = d_input_grad;
      } else {
        CUDA_CHECK(cudaMalloc(&prev_grad,
                               operations[i]->input_size() * sizeof(float)));
        grad_buffers.push_back(prev_grad);
      }
      operations[i]->backward(current_grad, prev_grad);
      CUDA_CHECK(cudaDeviceSynchronize());
      current_grad = prev_grad;
    }

    for (auto buf : grad_buffers) {
      CUDA_CHECK(cudaFree(buf));
    }
  }

  inline void set_optimizer(Optimizer *opt);
  inline void update_weights(float lr);
  inline float clip_grad_norm(float max_norm);

  std::vector<Operation *> &get_operations() { return operations; }

  bool save_weights(const std::string &path) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f)
      return false;
    char magic[4] = {'D', 'L', 'C', 'U'};
    fwrite(magic, 1, 4, f);
    std::vector<ParamGroup> all_groups;
    for (auto *op : operations) {
      auto groups = op->get_param_groups();
      all_groups.insert(all_groups.end(), groups.begin(), groups.end());
    }
    int num_groups = static_cast<int>(all_groups.size());
    fwrite(&num_groups, sizeof(int), 1, f);
    for (auto &pg : all_groups) {
      fwrite(&pg.size, sizeof(int), 1, f);
      std::vector<float> h_data(pg.size);
      CUDA_CHECK(cudaMemcpy(h_data.data(), pg.params, pg.size * sizeof(float),
                             cudaMemcpyDeviceToHost));
      fwrite(h_data.data(), sizeof(float), pg.size, f);
    }
    fclose(f);
    return true;
  }

  bool load_weights(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
      return false;
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "DLCU", 4) != 0) {
      fclose(f);
      return false;
    }
    std::vector<ParamGroup> all_groups;
    for (auto *op : operations) {
      auto groups = op->get_param_groups();
      all_groups.insert(all_groups.end(), groups.begin(), groups.end());
    }
    int num_groups;
    fread(&num_groups, sizeof(int), 1, f);
    if (num_groups != static_cast<int>(all_groups.size())) {
      fclose(f);
      return false;
    }
    for (auto &pg : all_groups) {
      int size;
      fread(&size, sizeof(int), 1, f);
      if (size != pg.size) {
        fclose(f);
        return false;
      }
      std::vector<float> h_data(size);
      fread(h_data.data(), sizeof(float), size, f);
      CUDA_CHECK(cudaMemcpy(pg.params, h_data.data(), size * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
    fclose(f);
    return true;
  }
};
