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

__global__ void lstmKernel(float *x, float *h_prev, float *c_prev,
                           float *Wf, float *Wi, float *Wc, float *Wo,
                           float *bf, float *bi, float *bc, float *bo,
                           float *h, float *c, int sequence_length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sequence_length)
    {
        // Forget gate
        float ft = sigmoid(dot(Wf, concat(h_prev, x[idx])) + bf);

        // Input gate & Cell state update
        float it = sigmoid(dot(Wi, concat(h_prev, x[idx])) + bi);
        float c_tilde = tanh(dot(Wc, concat(h_prev, x[idx])) + bc);
        c[idx] = ft * c_prev[idx] + it * c_tilde;

        // Output gate
        float ot = sigmoid(dot(Wo, concat(h_prev, x[idx])) + bo);

        // Hidden state
        h[idx] = ot * tanh(c[idx]);
    }
}
