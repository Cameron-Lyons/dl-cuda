#include <cuda_runtime.h>

__global__ void tanh(float *data, int n, int elements_per_thread) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;

  for (int i = 0; i < elements_per_thread && (idx + i) < n; i++) {
    data[idx + i] = tanhf(data[idx + i]);
  }
}

__global__ void sigmoid(float *input, float *output, int n) {
  __shared__ float shared_input[256];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    shared_input[threadIdx.x] = input[idx];
    __syncthreads();

    output[idx] = 1.0f / (1.0f + expf(-shared_input[threadIdx.x]));
  }
}

__global__ void relu(float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
  }
}
