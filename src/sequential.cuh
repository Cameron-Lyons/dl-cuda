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

  void forward(float *h_input, float *h_output) {
    float *d_input, *d_output;
    size_t input_size = ...;  // Size of input data
    size_t output_size = ...; // Size of output data

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    float *temp_in = d_input;
    float *temp_out = d_output;

    for (auto &op : operations) {
      op->forward(temp_in, temp_out);

      // Synchronize after each operation
      cudaDeviceSynchronize();

      temp_in =
          temp_out; // output of current operation is the input to the next
    }

    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
  }
};
