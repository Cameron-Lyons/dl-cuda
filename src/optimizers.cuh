__global__ void computeGradients(float *d_x, float *d_y, float *d_w, float *d_b,
                                 float *d_dw, float *d_db, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n) {
    float error = (*d_w) * d_x[idx] + (*d_b) - d_y[idx];
    d_dw[idx] = error * d_x[idx];
    d_db[idx] = error;
  }
}
