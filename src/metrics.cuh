#include <cuda_runtime.h>

__global__ void r2Kernel(float *y, float *y_pred, float *numerator, float *denominator, float y_mean, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float diff = y[idx] - y_pred[idx];
        numerator[idx] = diff * diff;

        float diff_mean = y[idx] - y_mean;
        denominator[idx] = diff_mean * diff_mean;
    }
}
