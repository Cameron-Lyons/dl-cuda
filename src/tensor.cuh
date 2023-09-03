const short N_THREADS = 256;

struct Tensor {
  float *data;
  int *shape;    // pointer to an array on device
  int *strides;  // pointer to an array on device
  int shapeSize; // size of shape array

  Tensor(int shapeSize) : shapeSize(shapeSize) {
    cudaMalloc(&shape, sizeof(int) * shapeSize);
    cudaMalloc(&strides, sizeof(int) * shapeSize);
  }

  ~Tensor() {
    cudaFree(shape);
    cudaFree(strides);
  }

  void calculateStrides() {
    int *hostStrides = new int[shapeSize];
    int *hostShape = new int[shapeSize];

    hostStrides[shapeSize - 1] = 1;
    for (int i = shapeSize - 2; i >= 0; --i) {
      hostStrides[i] = hostStrides[i + 1] * hostShape[i + 1];
    }

    cudaMemcpy(strides, hostStrides, sizeof(int) * shapeSize,
               cudaMemcpyHostToDevice);
    delete[] hostStrides;
    delete[] hostShape;
  }

  __host__ __device__ int getLinearIndex(const int *indices) const {
    int index = 0;
    for (int i = 0; i < shapeSize; ++i) {
      index += indices[i] * strides[i];
    }
    return index;
  }
  __global__ void addKernel(const float *A, const float *B, float *C,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      C[idx] = A[idx] + B[idx];
    }
  }

  Tensor operator+(const Tensor &other) {
    Tensor result(shapeSize);

    float *resultData;
    cudaMalloc(&resultData, sizeof(float) * shapeSize);

    int blocks = (shapeSize + N_THREADS - 1) / threadsPerBlock;

    addKernel<<<blocks, threadsPerBlock>>>(data, other.data, resultData,
                                           shapeSize);

    result.data = resultData;
    return result;
  }
  __global__ void scalarMulKernel(float *A, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      A[idx] *= scalar;
    }
  }

  void operator*=(float scalar) {
    int threadsPerBlock = 256;
    int blocks = (shapeSize + threadsPerBlock - 1) / threadsPerBlock;

    scalarMulKernel<<<blocks, threadsPerBlock>>>(data, scalar, shapeSize);
  }
  __global__ void matMulKernel(const float *A, const float *B, float *C,
                               int ARows, int ACols, int BCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if (row < ARows && col < BCols) {
      for (int k = 0; k < ACols; ++k) {
        sum += A[row * ACols + k] * B[k * BCols + col];
      }
      C[row * BCols + col] = sum;
    }
  }

  Tensor matMul(const Tensor &other) {
    int ARows = shape[0];
    int ACols = shape[1];
    int BCols = other.shape[1];

    Tensor result(2);

    float *resultData;
    cudaMalloc(&resultData, sizeof(float) * ARows * BCols);

    dim3 threadsPerBlock(16, 16); // 2D block
    dim3 blocks((BCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (ARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matMulKernel<<<blocks, threadsPerBlock>>>(data, other.data, resultData,
                                              ARows, ACols, BCols);

    result.data = resultData;
    return result;
  }
};
}
;
