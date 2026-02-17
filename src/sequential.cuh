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
  static constexpr int kWeightsFormatVersion = 1;
  std::vector<Operation *> operations;
  Optimizer *optimizer_ = nullptr;
  std::vector<float *> forward_buffers_;
  std::vector<float *> backward_buffers_;
  std::vector<int> forward_buffer_sizes_;
  std::vector<int> backward_buffer_sizes_;

  void clear_buffers(std::vector<float *> &buffers) {
    for (auto *buf : buffers) {
      if (buf)
        CUDA_CHECK(cudaFree(buf));
    }
    buffers.clear();
  }

  void release_workspaces() {
    clear_buffers(forward_buffers_);
    clear_buffers(backward_buffers_);
    forward_buffer_sizes_.clear();
    backward_buffer_sizes_.clear();
  }

  void ensure_forward_workspace() {
    size_t needed = operations.size() > 1 ? operations.size() - 1 : 0;
    bool sizes_match = needed == forward_buffers_.size();
    if (sizes_match) {
      for (size_t i = 0; i < needed; i++) {
        if (forward_buffer_sizes_[i] != operations[i]->output_size()) {
          sizes_match = false;
          break;
        }
      }
    }
    if (sizes_match)
      return;
    clear_buffers(forward_buffers_);
    forward_buffer_sizes_.clear();
    forward_buffers_.resize(needed, nullptr);
    forward_buffer_sizes_.resize(needed, 0);
    for (size_t i = 0; i < needed; i++) {
      int size = operations[i]->output_size();
      CUDA_CHECK(cudaMalloc(&forward_buffers_[i], size * sizeof(float)));
      forward_buffer_sizes_[i] = size;
    }
  }

  void ensure_backward_workspace() {
    size_t needed = operations.size() > 1 ? operations.size() - 1 : 0;
    bool sizes_match = needed == backward_buffers_.size();
    if (sizes_match) {
      for (size_t i = 0; i < needed; i++) {
        if (backward_buffer_sizes_[i] != operations[i + 1]->input_size()) {
          sizes_match = false;
          break;
        }
      }
    }
    if (sizes_match)
      return;
    clear_buffers(backward_buffers_);
    backward_buffer_sizes_.clear();
    backward_buffers_.resize(needed, nullptr);
    backward_buffer_sizes_.resize(needed, 0);
    for (size_t i = 0; i < needed; i++) {
      int size = operations[i + 1]->input_size();
      CUDA_CHECK(cudaMalloc(&backward_buffers_[i], size * sizeof(float)));
      backward_buffer_sizes_[i] = size;
    }
  }

public:
  ~Sequential() { release_workspaces(); }

  void add(Operation *op) {
    operations.push_back(op);
    release_workspaces();
  }

  void forward(float *d_input, float *d_output) {
    if (operations.empty())
      return;
    ensure_forward_workspace();

    float *current_in = d_input;
    for (size_t i = 0; i < operations.size(); i++) {
      float *current_out;
      if (i == operations.size() - 1) {
        current_out = d_output;
      } else {
        current_out = forward_buffers_[i];
      }
      operations[i]->forward(current_in, current_out);
      current_in = current_out;
    }
  }

  void backward(float *d_output_grad, float *d_input_grad) {
    if (operations.empty())
      return;
    ensure_backward_workspace();

    float *current_grad = d_output_grad;
    for (int i = static_cast<int>(operations.size()) - 1; i >= 0; i--) {
      float *prev_grad;
      if (i == 0) {
        prev_grad = d_input_grad;
      } else {
        prev_grad = backward_buffers_[static_cast<size_t>(i - 1)];
      }
      operations[i]->backward(current_grad, prev_grad);
      current_grad = prev_grad;
    }
  }

  inline void set_optimizer(Optimizer *opt);
  inline void update_weights(float lr);
  inline float clip_grad_norm(float max_norm);

  std::vector<Operation *> &get_operations() { return operations; }

  bool save_weights(const std::string &path) {
    auto write_exact = [](const void *ptr, size_t size, size_t count,
                          FILE *f) -> bool {
      return fwrite(ptr, size, count, f) == count;
    };

    FILE *f = fopen(path.c_str(), "wb");
    if (!f)
      return false;
    char magic[4] = {'D', 'L', 'C', 'U'};
    if (!write_exact(magic, 1, 4, f)) {
      fclose(f);
      return false;
    }
    int version = kWeightsFormatVersion;
    if (!write_exact(&version, sizeof(int), 1, f)) {
      fclose(f);
      return false;
    }
    std::vector<ParamGroup> all_groups;
    for (auto *op : operations) {
      auto groups = op->get_param_groups();
      all_groups.insert(all_groups.end(), groups.begin(), groups.end());
    }
    int num_groups = static_cast<int>(all_groups.size());
    if (!write_exact(&num_groups, sizeof(int), 1, f)) {
      fclose(f);
      return false;
    }
    for (auto &pg : all_groups) {
      if (!write_exact(&pg.size, sizeof(int), 1, f)) {
        fclose(f);
        return false;
      }
      std::vector<float> h_data(pg.size);
      CUDA_CHECK(cudaMemcpy(h_data.data(), pg.params, pg.size * sizeof(float),
                             cudaMemcpyDeviceToHost));
      if (!write_exact(h_data.data(), sizeof(float), pg.size, f)) {
        fclose(f);
        return false;
      }
    }
    fclose(f);
    return true;
  }

  bool load_weights(const std::string &path) {
    auto read_exact = [](void *ptr, size_t size, size_t count, FILE *f) -> bool {
      return fread(ptr, size, count, f) == count;
    };

    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
      return false;
    char magic[4];
    if (!read_exact(magic, 1, 4, f)) {
      fclose(f);
      return false;
    }
    if (memcmp(magic, "DLCU", 4) != 0) {
      fclose(f);
      return false;
    }
    int version = 0;
    if (!read_exact(&version, sizeof(int), 1, f)) {
      fclose(f);
      return false;
    }
    if (version != kWeightsFormatVersion) {
      fclose(f);
      return false;
    }
    std::vector<ParamGroup> all_groups;
    for (auto *op : operations) {
      auto groups = op->get_param_groups();
      all_groups.insert(all_groups.end(), groups.begin(), groups.end());
    }
    int num_groups;
    if (!read_exact(&num_groups, sizeof(int), 1, f)) {
      fclose(f);
      return false;
    }
    if (num_groups != static_cast<int>(all_groups.size())) {
      fclose(f);
      return false;
    }
    for (auto &pg : all_groups) {
      int size;
      if (!read_exact(&size, sizeof(int), 1, f)) {
        fclose(f);
        return false;
      }
      if (size != pg.size) {
        fclose(f);
        return false;
      }
      std::vector<float> h_data(size);
      if (!read_exact(h_data.data(), sizeof(float), size, f)) {
        fclose(f);
        return false;
      }
      CUDA_CHECK(cudaMemcpy(pg.params, h_data.data(), size * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
    fclose(f);
    return true;
  }
};
