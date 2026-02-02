#pragma once

#include <cuda_runtime.h>

static const short N_THREADS = 256;

__global__ void addKernel(const float *A, const float *B, float *C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void scalarMulKernel(float *A, float scalar, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    A[idx] *= scalar;
  }
}

__global__ void matMulKernel(const float *A, const float *B, float *C,
                             int ARows, int ACols, int BCols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < ARows && col < BCols) {
    float sum = 0;
    for (int k = 0; k < ACols; ++k) {
      sum += A[row * ACols + k] * B[k * BCols + col];
    }
    C[row * BCols + col] = sum;
  }
}

struct Tensor {
  float *data;
  int *shape;
  int *strides;
  int shapeSize;

  Tensor(int shapeSize) : data(nullptr), shapeSize(shapeSize) {
    cudaMalloc(&shape, sizeof(int) * shapeSize);
    cudaMalloc(&strides, sizeof(int) * shapeSize);
  }

  ~Tensor() {
    cudaFree(shape);
    cudaFree(strides);
    if (data)
      cudaFree(data);
  }

  int numElements() const {
    int *hostShape = new int[shapeSize];
    cudaMemcpy(hostShape, shape, sizeof(int) * shapeSize,
               cudaMemcpyDeviceToHost);
    int total = 1;
    for (int i = 0; i < shapeSize; i++) {
      total *= hostShape[i];
    }
    delete[] hostShape;
    return total;
  }

  void calculateStrides() {
    int *hostStrides = new int[shapeSize];
    int *hostShape = new int[shapeSize];

    cudaMemcpy(hostShape, shape, sizeof(int) * shapeSize,
               cudaMemcpyDeviceToHost);

    hostStrides[shapeSize - 1] = 1;
    for (int i = shapeSize - 2; i >= 0; --i) {
      hostStrides[i] = hostStrides[i + 1] * hostShape[i + 1];
    }

    cudaMemcpy(strides, hostStrides, sizeof(int) * shapeSize,
               cudaMemcpyHostToDevice);
    delete[] hostStrides;
    delete[] hostShape;
  }

  Tensor operator+(const Tensor &other) {
    int n = numElements();
    Tensor result(shapeSize);
    cudaMemcpy(result.shape, shape, sizeof(int) * shapeSize,
               cudaMemcpyDeviceToDevice);

    float *resultData;
    cudaMalloc(&resultData, sizeof(float) * n);

    int blocks = (n + N_THREADS - 1) / N_THREADS;
    addKernel<<<blocks, N_THREADS>>>(data, other.data, resultData, n);

    result.data = resultData;
    result.calculateStrides();
    return result;
  }

  void operator*=(float scalar) {
    int n = numElements();
    int blocks = (n + N_THREADS - 1) / N_THREADS;
    scalarMulKernel<<<blocks, N_THREADS>>>(data, scalar, n);
  }

  Tensor matMul(const Tensor &other) {
    int hostShapeA[2], hostShapeB[2];
    cudaMemcpy(hostShapeA, shape, sizeof(int) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostShapeB, other.shape, sizeof(int) * 2,
               cudaMemcpyDeviceToHost);

    int ARows = hostShapeA[0];
    int ACols = hostShapeA[1];
    int BCols = hostShapeB[1];

    Tensor result(2);
    int resultShape[2] = {ARows, BCols};
    cudaMemcpy(result.shape, resultShape, sizeof(int) * 2,
               cudaMemcpyHostToDevice);

    float *resultData;
    cudaMalloc(&resultData, sizeof(float) * ARows * BCols);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((BCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (ARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matMulKernel<<<blocks, threadsPerBlock>>>(data, other.data, resultData,
                                              ARows, ACols, BCols);

    result.data = resultData;
    result.calculateStrides();
    return result;
  }
};
