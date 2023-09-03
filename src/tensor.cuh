
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
};
}
;
