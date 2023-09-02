#include <cmath>
#include <cuda_runtime.h>

__global__ void linearLayerKernel(float *X, float *W, float *b, float *Y, int n,
                                  int in_features, int out_features) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
      sum += X[row * in_features + i] * W[i * out_features + col];
    }
    Y[row * out_features + col] = sum + b[col];
  }
}

void linearLayer(float *h_X, float *h_W, float *h_b, float *h_Y, int n,
                 int in_features, int out_features) {
  float *d_X, *d_W, *d_b, *d_Y;

  cudaMalloc((void **)&d_X, n * in_features * sizeof(float));
  cudaMalloc((void **)&d_W, in_features * out_features * sizeof(float));
  cudaMalloc((void **)&d_b, out_features * sizeof(float));
  cudaMalloc((void **)&d_Y, n * out_features * sizeof(float));

  cudaMemcpy(d_X, h_X, n * in_features * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W, in_features * out_features * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, out_features * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(32, 32); // Change this based on GPU
  dim3 blocks((out_features + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

  linearLayerKernel<<<blocks, threadsPerBlock>>>(d_X, d_W, d_b, d_Y, n,
                                                 in_features, out_features);

  cudaMemcpy(h_Y, d_Y, n * out_features * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_X);
  cudaFree(d_W);
  cudaFree(d_b);
  cudaFree(d_Y);
}

__global__ void lstmKernel(float *x, float *h_prev, float *c_prev, float *Wf,
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
    int num_threads = 256; // Adjust based on GPU capabilities
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
