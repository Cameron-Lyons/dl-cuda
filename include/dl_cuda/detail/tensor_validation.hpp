#pragma once

#include "dl_cuda/status.hpp"
#include "dl_cuda/tensor.hpp"

#include <sstream>
#include <string>

namespace dlcuda::detail {

inline Status ValidateDefinedTensor(const Tensor &tensor, const char *name) {
  if (!tensor.defined()) {
    return Status::InvalidArgument(std::string(name) + " is undefined");
  }
  return Status::Ok();
}

inline Status ValidateFloatingTensor(const Tensor &tensor, const char *name) {
  DLCUDA_RETURN_IF_ERROR(ValidateDefinedTensor(tensor, name));
  if (!IsFloatingPointDType(tensor.dtype())) {
    return Status::InvalidArgument(std::string(name) + " must be floating point");
  }
  return Status::Ok();
}

inline Status ValidateIntTensor(const Tensor &tensor, const char *name) {
  DLCUDA_RETURN_IF_ERROR(ValidateDefinedTensor(tensor, name));
  if (tensor.dtype() != DType::kInt32) {
    return Status::InvalidArgument(std::string(name) + " must be int32");
  }
  return Status::Ok();
}

inline Status ValidateRank(const Tensor &tensor, int64_t rank, const char *name) {
  if (tensor.rank() != rank) {
    std::ostringstream oss;
    oss << name << " must have rank " << rank << ", got " << tensor.rank();
    return Status::InvalidArgument(oss.str());
  }
  return Status::Ok();
}

inline Status EnsureSameShapeAndType(const Tensor &a, const Tensor &b, const char *a_name,
                                     const char *b_name) {
  if (a.dtype() != b.dtype()) {
    return Status::InvalidArgument(std::string(a_name) + " and " + b_name + " dtype mismatch");
  }
  if (a.shape() != b.shape()) {
    return Status::InvalidArgument(std::string(a_name) + " and " + b_name + " shape mismatch");
  }
  return Status::Ok();
}

inline Status EnsureDType(const Tensor &tensor, DType dtype, const char *name) {
  if (tensor.dtype() != dtype) {
    return Status::InvalidArgument(std::string(name) + " dtype mismatch: expected " +
                                   DTypeName(dtype) + " got " + DTypeName(tensor.dtype()));
  }
  return Status::Ok();
}

} // namespace dlcuda::detail
