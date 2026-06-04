#pragma once

struct CUstream_st;
using cudaStream_t = CUstream_st *;

struct cublasContext;
using cublasHandle_t = cublasContext *;

#if defined(DLCUDA_HAS_CUBLASLT)
struct cublasLtContext;
using cublasLtHandle_t = cublasLtContext *;
#endif
