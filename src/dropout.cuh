__global__ void dropoutKernel(float *input, float *output, float *mask,
                              float dropout_prob, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    float rand_val = (float)(idx * 1234) / INT_MAX;

    mask[idx] = (rand_val > dropout_prob) ? 1.0f : 0.0f;

    output[idx] = input[idx] * mask[idx];
  }
}
