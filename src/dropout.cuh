#include <curand_kernel.h>

const int THREADS_PER_BLOCK = 256;

__global__ void initializeCurandStates(curandState_t *states,
                                       unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed + idx, idx, 0, &states[idx]); // Add idx to seed for variety
}

const int ELEMENTS_PER_THREAD = 2;

__global__ void dropoutKernel(float *input, float *output, float *mask,
                              float dropout_prob, int size,
                              curandState_t *states) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

  for (int i = 0; i < ELEMENTS_PER_THREAD && (idx + i) < size; i++) {
    float rand_val = curand_uniform(&states[idx + i]);

    mask[idx + i] = (rand_val > dropout_prob) ? 1.0f : 0.0f;
    output[idx + i] = input[idx + i] * mask[idx + i];
  }
}
