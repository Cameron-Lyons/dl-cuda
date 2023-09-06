#include <cmath>
#include <cuda_runtime.h>

const short NUM_THREADS = 256;

__global__ void linearLayerKernel(float *d_X, float *d_W, float *d_b, float *d_Y,
                                  int n, int in_features, int out_features) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < out_features && row < n) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += d_X[row * in_features + i] * d_W[i * out_features + col];
        }
        d_Y[row * out_features + col] = sum + d_b[col];
    }
}

class LinearLayer {
private:
    float *d_X, *d_W, *d_b, *d_Y;
    int n, in_features, out_features;

    void allocateDeviceMemory() {
        cudaMalloc((void **)&d_X, n * in_features * sizeof(float));
        cudaMalloc((void **)&d_W, in_features * out_features * sizeof(float));
        cudaMalloc((void **)&d_b, out_features * sizeof(float));
        cudaMalloc((void **)&d_Y, n * out_features * sizeof(float));
    }

    void freeDeviceMemory() {
        cudaFree(d_X);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_Y);
    }

public:
    LinearLayer(int n, int in_features, int out_features)
        : n(n), in_features(in_features), out_features(out_features) {
        allocateDeviceMemory();
    }

    ~LinearLayer() { freeDeviceMemory(); }

    void forward(float *h_X, float *h_W, float *h_b, float *h_Y) {
        cudaMemcpy(d_X, h_X, n * in_features * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_W, h_W, in_features * out_features * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, out_features * sizeof(float),
                   cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(32, 32);
        dim3 blocks((out_features + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

        linearLayerKernel<<<blocks, threadsPerBlock>>>(d_X, d_W, d_b, d_Y, n,
                                                       in_features, out_features);

        cudaMemcpy(h_Y, d_Y, n * out_features * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
};__global__ void lstmKernel(float *x, float *h_prev, float *c_prev, float *Wf,
                           float *Wi, float *Wc, float *Wo, float *bf,
                           float *bi, float *bc, float *bo, float *h, float *c,
                           int batch_size, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    float combinedVec[2 * hidden_size];
    float ft = sigmoid(dotProduct(Wf, combinedVec, hidden_size) + bf[idx]);
    float it = sigmoid(dotProduct(Wi, combinedVec, hidden_size) + bi[idx]);
    float c_tilde = tanh(dotProduct(Wc, combinedVec, hidden_size) + bc[idx]);
    c[idx] = ft * c_prev[idx] + it * c_tilde;
    float ot = sigmoid(dotProduct(Wo, combinedVec, hidden_size) + bo[idx]);
    h[idx] = ot * tanh(c[idx]);
  }
}

class LSTMLayer {
private:
  float *Wf, *Wi, *Wc, *Wo, *bf, *bi, *bc, *bo;

  float *h, *c;

  int hidden_size, batch_size;

public:
  LSTMLayer(int hidden_size, int batch_size)
      : hidden_size(hidden_size), batch_size(batch_size) {

    int weightSize =
        2 * hidden_size * hidden_size; // '2' due to concatenation of h and x

    cudaMalloc(&Wf, weightSize * sizeof(float));
    cudaMalloc(&Wi, weightSize * sizeof(float));
    cudaMalloc(&Wc, weightSize * sizeof(float));
    cudaMalloc(&Wo, weightSize * sizeof(float));

    cudaMalloc(&bf, hidden_size * sizeof(float));
    cudaMalloc(&bi, hidden_size * sizeof(float));
    cudaMalloc(&bc, hidden_size * sizeof(float));
    cudaMalloc(&bo, hidden_size * sizeof(float));

    cudaMalloc(&h, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&c, batch_size * hidden_size * sizeof(float));
  }

  ~LSTMLayer() {
    cudaFree(Wf);
    cudaFree(Wi);
    cudaFree(Wc);
    cudaFree(Wo);

    cudaFree(bf);
    cudaFree(bi);
    cudaFree(bc);
    cudaFree(bo);

    cudaFree(h);
    cudaFree(c);
  }

  void forward(float *x, int sequence_length, float *output) {
    int num_threads = NUM_THREADS;
    int num_blocks = (batch_size + num_threads - 1) / num_threads;

    cudaMemset(h, 0, batch_size * hidden_size * sizeof(float));
    cudaMemset(c, 0, batch_size * hidden_size * sizeof(float));

    for (int t = 0; t < sequence_length; t++) {
      lstmKernel<<<num_blocks, num_threads>>>(
          /*... pass the necessary arguments including x[t] ...*/);

      cudaMemcpy(&output[t * batch_size * hidden_size], h,
                 batch_size * hidden_size * sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }
  }
};

__global__ void elmanRnnKernel(float *x, float *h_prev, float *Wxh, float *Whh,
                               float *b_h, float *Why, float *b_y, float *h,
                               float *y, int sequence_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < sequence_length) {
    float hidden_sum = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      hidden_sum += Wxh[idx * HIDDEN_SIZE + i] * x[idx] + Whh[i] * h_prev[idx];
    }
    h[idx] = tanhf(hidden_sum + b_h[idx]);

    float output_sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      output_sum += Why[idx * OUTPUT_SIZE + i] * h[idx];
    }
    y[idx] = output_sum + b_y[idx];
  }
}

class ElmanRNNLayer {
private:
  float *Wxh, *Whh, *b_h, *Why, *b_y;
  float *h;

  const int hidden_size = 16;
  const int output_size = 16;
  const int batch_size = 256;

public:
  ElmanRNNLayer() {
    cudaMalloc(&Wxh, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&Whh, hidden_size * hidden_size * sizeof(float));
    cudaMalloc(&b_h, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&Why, hidden_size * output_size * sizeof(float));
    cudaMalloc(&b_y, batch_size * output_size * sizeof(float));
    cudaMalloc(&h, batch_size * hidden_size * sizeof(float));

    cudaMemset(h, 0, batch_size * hidden_size * sizeof(float));
  }

  ~ElmanRNNLayer() {
    cudaFree(Wxh);
    cudaFree(Whh);
    cudaFree(b_h);
    cudaFree(Why);
    cudaFree(b_y);
    cudaFree(h);
  }

  void forward(float *x, int sequence_length, float *output) {
    int num_threads = NUM_THREADS;
    int num_blocks = (batch_size + num_threads - 1) / num_threads;

    for (int t = 0; t < sequence_length; t++) {
      elmanRnnKernel<<<num_blocks, num_threads>>>(
          &x[t * batch_size], // Input at current time step
          h,                  // Current hidden state
          Wxh, Whh, b_h, Why, b_y,
          h, // Updated hidden state (output from kernel)
          &output[t * batch_size * output_size], // Output at current time step
          batch_size);

      cudaMemcpy(&output[t * batch_size * output_size], y,
                 batch_size * output_size * sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }
  }
};

__global__ void conv1dKernel(float *input, float *kernel, float *output,
                             int inputSize, int kernelSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int halfK = kernelSize / 2;

  if (idx < inputSize) {
    float sum = 0.0f;
    for (int i = -halfK; i <= halfK; i++) {
      if (idx + i >= 0 && idx + i < inputSize) {
        sum += input[idx + i] * kernel[halfK + i];
      }
    }
    output[idx] = sum;
  }
}

class Conv1DLayer {
private:
  float *d_input;  // Device input
  float *d_kernel; // Device kernel/filter
  float *d_output; // Device output

  int inputSize;
  int kernelSize;

public:
  Conv1DLayer(int inputSize, int kernelSize)
      : inputSize(inputSize), kernelSize(kernelSize) {
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
  }

  ~Conv1DLayer() {
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
  }

  void forward(float *h_input, float *h_kernel, float *h_output) {
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float),
               cudaMemcpyHostToDevice);

    int num_threads = 256;
    int num_blocks = (inputSize + num_threads - 1) / num_threads;

    conv1dKernel<<<num_blocks, num_threads>>>(d_input, d_kernel, d_output,
                                              inputSize, kernelSize);

    cudaMemcpy(h_output, d_output, inputSize * sizeof(float),
               cudaMemcpyDeviceToHost);
  }
};

__global__ void conv2dKernel(float *input, int inputWidth, int inputHeight,
                             float *kernel, int kernelWidth, int kernelHeight,
                             float *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int halfKernelWidth = kernelWidth / 2;
  int halfKernelHeight = kernelHeight / 2;

  float value = 0.0f;

  if (x < inputWidth && y < inputHeight) {
    for (int ky = -halfKernelHeight; ky <= halfKernelHeight; ky++) {
      for (int kx = -halfKernelWidth; kx <= halfKernelWidth; kx++) {
        int inX = x + kx;
        int inY = y + ky;
        if (inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
          value += input[inY * inputWidth + inX] *
                   kernel[(ky + halfKernelHeight) * kernelWidth +
                          (kx + halfKernelWidth)];
        }
      }
    }
    output[y * inputWidth + x] = value;
  }
}

class Conv2DLayer {
private:
  float *d_input;  // Device input
  float *d_kernel; // Device kernel/filter
  float *d_output; // Device output

  int inputWidth, inputHeight;
  int kernelWidth, kernelHeight;

public:
  Conv2DLayer(int inputWidth, int inputHeight, int kernelWidth,
              int kernelHeight)
      : inputWidth(inputWidth), inputHeight(inputHeight),
        kernelWidth(kernelWidth), kernelHeight(kernelHeight) {
    cudaMalloc(&d_input, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&d_kernel, kernelWidth * kernelHeight * sizeof(float));
    cudaMalloc(&d_output, inputWidth * inputHeight *
                              sizeof(float)); // Assuming "same" padding
  }

  ~Conv2DLayer() {
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
  }

  void forward(float *h_input, float *h_kernel, float *h_output) {
    cudaMemcpy(d_input, h_input, inputWidth * inputHeight * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelWidth * kernelHeight * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((inputWidth + block_size.x - 1) / block_size.x,
                   (inputHeight + block_size.y - 1) / block_size.y);

    conv2dKernel<<<grid_size, block_size>>>(d_input, inputWidth, inputHeight,
                                            d_kernel, kernelWidth, kernelHeight,
                                            d_output);

    cudaMemcpy(h_output, d_output, inputWidth * inputHeight * sizeof(float),
               cudaMemcpyDeviceToHost);
  }
};
