__global__ void computeGradients(float *d_x, float *d_y, float *d_w, float *d_b,
                                 float *d_dw, float *d_db, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n) {
    float error = (*d_w) * d_x[idx] + (*d_b) - d_y[idx];
    d_dw[idx] = error * d_x[idx];
    d_db[idx] = error;
  }
}

void SGD(float *h_x, float *h_y, float *h_w, float *h_b, float learning_rate,
         int n, int num_epochs) {
  float *d_x, *d_y, *d_w, *d_b, *d_dw, *d_db;

  cudaMalloc((void **)&d_x, n * sizeof(float));
  cudaMalloc((void **)&d_y, n * sizeof(float));
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_w, sizeof(float));
  cudaMalloc((void **)&d_b, sizeof(float));
  cudaMemcpy(d_w, h_w, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_dw, n * sizeof(float));
  cudaMalloc((void **)&d_db, n * sizeof(float));

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int i = 0; i < n; i++) {
      computeGradients<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_w, d_b,
                                                           d_dw, d_db, n);

      float h_dw[n], h_db[n];
      cudaMemcpy(h_dw, d_dw, n * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_db, d_db, n * sizeof(float), cudaMemcpyDeviceToHost);

      float dw_avg = 0.0f, db_avg = 0.0f;
      for (int j = 0; j < n; j++) {
        dw_avg += h_dw[j];
        db_avg += h_db[j];
      }
      dw_avg /= n;
      db_avg /= n;

      // Update weights and biases
      *h_w -= learning_rate * dw_avg;
      *h_b -= learning_rate * db_avg;
    }
  }

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_w);
  cudaFree(d_b);
  cudaFree(d_dw);
  cudaFree(d_db);
}

__global__ void updateRMSprop(float *d_g, float *d_s, float *d_theta,
                              float learning_rate, float decay_rate,
                              float epsilon, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_s[idx] = decay_rate * d_s[idx] + (1 - decay_rate) * d_g[idx] * d_g[idx];

    d_theta[idx] -= learning_rate * d_g[idx] / (sqrtf(d_s[idx]) + epsilon);
  }
}

void RMSprop(float *d_g, float *d_s, float *d_theta, float learning_rate,
             float decay_rate, float epsilon, int n) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  updateRMSprop<<<blocksPerGrid, threadsPerBlock>>>(
      d_g, d_s, d_theta, learning_rate, decay_rate, epsilon, n);
}

__global__ void updateAdam(float *d_g, float *d_m, float *d_v, float *d_theta,
                           float alpha, float beta1, float beta2, float epsilon,
                           int t, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    d_m[idx] = beta1 * d_m[idx] + (1.0f - beta1) * d_g[idx];
    d_v[idx] = beta2 * d_v[idx] + (1.0f - beta2) * d_g[idx] * d_g[idx];

    float m_hat = d_m[idx] / (1.0f - powf(beta1, t));
    float v_hat = d_v[idx] / (1.0f - powf(beta2, t));

    d_theta[idx] -= alpha * m_hat / (sqrtf(v_hat) + epsilon);
  }
}

void Adam(float *d_g, float *d_m, float *d_v, float *d_theta, float alpha,
          float beta1, float beta2, float epsilon, int t, int n) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  updateAdam<<<blocksPerGrid, threadsPerBlock>>>(d_g, d_m, d_v, d_theta, alpha,
                                                 beta1, beta2, epsilon, t, n);
}
