#pragma once

#include "common.cuh"

namespace dlcuda {
namespace {

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
#define DLCUDA_CAN_CAPTURE_CUDA_GRAPHS 1
#endif

#if defined(DLCUDA_CAN_CAPTURE_CUDA_GRAPHS)
class CudaGraphExec {
public:
  CudaGraphExec() = default;
  CudaGraphExec(const CudaGraphExec &) = delete;
  CudaGraphExec &operator=(const CudaGraphExec &) = delete;

  ~CudaGraphExec() {
    Reset();
  }

  [[nodiscard]] bool ready() const {
    return exec_ != nullptr;
  }

  void Reset() {
    if (exec_ != nullptr) {
      cudaGraphExecDestroy(exec_);
      exec_ = nullptr;
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
  }

  template <typename Fn> Status Capture(cudaStream_t stream, Fn &&fn) {
    Reset();
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
      return UnsupportedGraphStatus("cudaStreamBeginCapture", err);
    }

    Status body_status = fn();
    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(stream, &captured_graph);
    if (!body_status.ok()) {
      if (captured_graph != nullptr) {
        cudaGraphDestroy(captured_graph);
      }
      return Status::Unsupported("CUDA graph capture body failed: " + body_status.ToString());
    }
    if (err != cudaSuccess) {
      if (captured_graph != nullptr) {
        cudaGraphDestroy(captured_graph);
      }
      return UnsupportedGraphStatus("cudaStreamEndCapture", err);
    }

    cudaGraphExec_t captured_exec = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    err = cudaGraphInstantiateWithFlags(&captured_exec, captured_graph, 0);
#else
    err = cudaGraphInstantiate(&captured_exec, captured_graph, nullptr, nullptr, 0);
#endif
    if (err != cudaSuccess) {
      cudaGraphDestroy(captured_graph);
      return UnsupportedGraphStatus("cudaGraphInstantiate", err);
    }

    graph_ = captured_graph;
    exec_ = captured_exec;
    return Status::Ok();
  }

  Status Launch(cudaStream_t stream) {
    if (exec_ == nullptr) {
      return Status::InvalidArgument("CUDA graph has not been captured");
    }
    cudaError_t err = cudaGraphLaunch(exec_, stream);
    return detail::CudaStatus(err, "cudaGraphLaunch");
  }

private:
  static Status UnsupportedGraphStatus(const char *context, cudaError_t err) {
    return Status::Unsupported(std::string(context) + ": " + cudaGetErrorString(err));
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};
#endif

} // namespace
} // namespace dlcuda
