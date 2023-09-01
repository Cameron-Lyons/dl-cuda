#include <curand_kernel.h>

const int THREADS_PER_BLOCK = 256;

__global__ void initializeCurandStates(curandState_t *states,
                                       unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &states[idx]);
}

__global__ void dropoutKernel(float *input, float *output, float *mask,
                              float dropout_prob, int size,
                              curandState_t *states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    float rand_val = curand_uniform(&states[idx]);

    mask[idx] = (rand_val > dropout_prob) ? 1.0f : 0.0f;

    output[idx] = input[idx] * mask[idx];
  }
}
