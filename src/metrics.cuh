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

__global__ void accuracyKernel(int *y, int *y_pred, int *correct_preds, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        correct_preds[idx] = (y[idx] == y_pred[idx]) ? 1 : 0;
    }
}

__global__ void f1Kernel(int *y, int *y_pred, int *TP, int *FP, int *FN, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        TP[idx] = (y[idx] == 1 && y_pred[idx] == 1) ? 1 : 0;
        FP[idx] = (y[idx] == 0 && y_pred[idx] == 1) ? 1 : 0;
        FN[idx] = (y[idx] == 1 && y_pred[idx] == 0) ? 1 : 0;
    }
}

__global__ void mccKernel(int *y, int *y_pred, int *TP, int *FP, int *FN, int *TN, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        TP[idx] = (y[idx] == 1 && y_pred[idx] == 1) ? 1 : 0;
        FP[idx] = (y[idx] == 0 && y_pred[idx] == 1) ? 1 : 0;
        FN[idx] = (y[idx] == 1 && y_pred[idx] == 0) ? 1 : 0;
        TN[idx] = (y[idx] == 0 && y_pred[idx] == 0) ? 1 : 0;
    }
}
