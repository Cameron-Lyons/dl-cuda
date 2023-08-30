#include <cuda_runtime.h>

__global__ void tanh(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = tanhf(data[idx]);
    }
}

__global__ void sigmoid(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void relu(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}
