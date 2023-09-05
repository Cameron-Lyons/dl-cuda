class Operation {
public:
  virtual void forward(float *input, float *output) = 0;
  virtual ~Operation() = default;
};

class Sequential {
private:
  std::vector<Operation *> operations;

public:
  void add(Operation *op) { operations.push_back(op); }

  void forward(float *h_input, float *h_output, size_t input_size,
               size_t output_size) {
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    CUDA_CHECK(
        cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    float *temp_in = d_input;
    float *temp_out = d_output;

    for (auto &op : operations) {
      op->forward(temp_in, temp_out);

      CUDA_CHECK(cudaDeviceSynchronize());

      temp_in = temp_out;
    }

    CUDA_CHECK(
        cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
  }
};
