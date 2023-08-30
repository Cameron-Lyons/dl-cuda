#include <cuda_runtime.h>

__global__ void linearLayerKernel(float *X, float *W, float *b, float *Y, int n, int in_features, int out_features)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < out_features)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++)
        {
            sum += X[row * in_features + i] * W[i * out_features + col];
        }
        Y[row * out_features + col] = sum + b[col];
    }
}
