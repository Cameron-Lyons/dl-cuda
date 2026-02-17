#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

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
    CUDA_CHECK(cudaMalloc(&shape, sizeof(int) * shapeSize));
    CUDA_CHECK(cudaMalloc(&strides, sizeof(int) * shapeSize));
  }

  ~Tensor() {
    if (shape)
      cudaFree(shape);
    if (strides)
      cudaFree(strides);
    if (data)
      cudaFree(data);
  }

  Tensor(const Tensor &other) : data(nullptr), shape(nullptr), strides(nullptr),
                                shapeSize(other.shapeSize) {
    CUDA_CHECK(cudaMalloc(&shape, sizeof(int) * shapeSize));
    CUDA_CHECK(cudaMalloc(&strides, sizeof(int) * shapeSize));
    CUDA_CHECK(cudaMemcpy(shape, other.shape, sizeof(int) * shapeSize,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(strides, other.strides, sizeof(int) * shapeSize,
                          cudaMemcpyDeviceToDevice));

    if (other.data) {
      int n = other.numElements();
      CUDA_CHECK(cudaMalloc(&data, sizeof(float) * n));
      CUDA_CHECK(cudaMemcpy(data, other.data, sizeof(float) * n,
                            cudaMemcpyDeviceToDevice));
    }
  }

  Tensor(Tensor &&other) noexcept
      : data(other.data), shape(other.shape), strides(other.strides),
        shapeSize(other.shapeSize) {
    other.data = nullptr;
    other.shape = nullptr;
    other.strides = nullptr;
    other.shapeSize = 0;
  }

  Tensor &operator=(const Tensor &other) {
    if (this == &other)
      return *this;

    Tensor tmp(other);
    swap(tmp);
    return *this;
  }

  Tensor &operator=(Tensor &&other) noexcept {
    if (this == &other)
      return *this;

    if (shape)
      cudaFree(shape);
    if (strides)
      cudaFree(strides);
    if (data)
      cudaFree(data);

    data = other.data;
    shape = other.shape;
    strides = other.strides;
    shapeSize = other.shapeSize;

    other.data = nullptr;
    other.shape = nullptr;
    other.strides = nullptr;
    other.shapeSize = 0;
    return *this;
  }

  void swap(Tensor &other) noexcept {
    std::swap(data, other.data);
    std::swap(shape, other.shape);
    std::swap(strides, other.strides);
    std::swap(shapeSize, other.shapeSize);
  }

  int numElements() const {
    int *hostShape = new int[shapeSize];
    CUDA_CHECK(cudaMemcpy(hostShape, shape, sizeof(int) * shapeSize,
                          cudaMemcpyDeviceToHost));
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

    CUDA_CHECK(cudaMemcpy(hostShape, shape, sizeof(int) * shapeSize,
                          cudaMemcpyDeviceToHost));

    hostStrides[shapeSize - 1] = 1;
    for (int i = shapeSize - 2; i >= 0; --i) {
      hostStrides[i] = hostStrides[i + 1] * hostShape[i + 1];
    }

    CUDA_CHECK(cudaMemcpy(strides, hostStrides, sizeof(int) * shapeSize,
                          cudaMemcpyHostToDevice));
    delete[] hostStrides;
    delete[] hostShape;
  }

  Tensor operator+(const Tensor &other) {
    int n = numElements();
    Tensor result(shapeSize);
    CUDA_CHECK(cudaMemcpy(result.shape, shape, sizeof(int) * shapeSize,
                          cudaMemcpyDeviceToDevice));

    float *resultData;
    CUDA_CHECK(cudaMalloc(&resultData, sizeof(float) * n));

    int blocks = (n + N_THREADS - 1) / N_THREADS;
    addKernel<<<blocks, N_THREADS>>>(data, other.data, resultData, n);
    CUDA_CHECK(cudaGetLastError());

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
    CUDA_CHECK(
        cudaMemcpy(hostShapeA, shape, sizeof(int) * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostShapeB, other.shape, sizeof(int) * 2,
                          cudaMemcpyDeviceToHost));

    int ARows = hostShapeA[0];
    int ACols = hostShapeA[1];
    int BCols = hostShapeB[1];

    Tensor result(2);
    int resultShape[2] = {ARows, BCols};
    CUDA_CHECK(cudaMemcpy(result.shape, resultShape, sizeof(int) * 2,
                          cudaMemcpyHostToDevice));

    float *resultData;
    CUDA_CHECK(cudaMalloc(&resultData, sizeof(float) * ARows * BCols));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((BCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (ARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matMulKernel<<<blocks, threadsPerBlock>>>(data, other.data, resultData,
                                              ARows, ACols, BCols);
    CUDA_CHECK(cudaGetLastError());

    result.data = resultData;
    result.calculateStrides();
    return result;
  }
};
