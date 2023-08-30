__global__ void tanhKernel(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = tanhf(data[idx]);
    }
}