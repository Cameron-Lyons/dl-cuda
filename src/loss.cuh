#include <cuda_runtime.h>

__global__ void squaredErrorKernel(float *y, float *y_pred, float *error, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float diff = y[idx] - y_pred[idx];
        error[idx] = diff * diff;
    }
}

__global__ void absoluteErrorKernel(float *y, float *y_pred, float *error, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        error[idx] = fabsf(y[idx] - y_pred[idx]);
    }
}
