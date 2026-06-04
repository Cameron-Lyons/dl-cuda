#include "detail/common.cuh"

namespace dlcuda {

Status Sequential::Add(std::unique_ptr<Module> module) {
  if (!module) {
    return Status::InvalidArgument("Sequential::Add received null module");
  }
  modules_.push_back(std::move(module));
  RebuildParameterCache();
  return Status::Ok();
}

Status Sequential::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Forward output pointer is null");
  }

  Tensor current = input;
  for (size_t i = 0; i < modules_.size(); ++i) {
    Tensor next;
    Status status = modules_[i]->Forward(ctx, current, &next);
    if (!status.ok()) {
      return Status::RuntimeError("Forward failed in module " + std::to_string(i) + ": " +
                                  status.message());
    }
    if (!next.defined()) {
      return Status::RuntimeError("Forward output became undefined in module " + std::to_string(i));
    }
    current = next;
  }

  *output = current;
  return Status::Ok();
}

Status Sequential::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (modules_.empty()) {
    return Status::InvalidArgument("Sequential has no modules");
  }

  Tensor current = grad_output;
  for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
    Tensor next;
    Tensor *next_out = (i == 0 && grad_input == nullptr) ? nullptr : &next;
    Status status = modules_[static_cast<size_t>(i)]->Backward(ctx, current, next_out);
    if (!status.ok()) {
      return Status::RuntimeError("Backward failed in module " + std::to_string(i) + ": " +
                                  status.message());
    }
    if (i > 0 && !next.defined()) {
      return Status::RuntimeError("Backward gradient became undefined before first module");
    }
    current = next;
  }

  if (grad_input != nullptr) {
    *grad_input = current;
  }
  return Status::Ok();
}

void Sequential::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  for (size_t i = 0; i < modules_.size(); ++i) {
    std::string child_name = "layers." + std::to_string(i);
    std::string child_prefix = prefix.empty() ? child_name : prefix + "." + child_name;
    modules_[i]->AppendParameters(child_prefix, out);
  }
}

void Sequential::RebuildParameterCache() {
  parameter_cache_.clear();
  AppendParameters("", &parameter_cache_);
}

Residual::Residual(std::unique_ptr<Module> branch) : branch_(std::move(branch)) {}

Status Residual::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!branch_) {
    return Status::InvalidArgument("Residual branch is null");
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Residual::Forward output is null");
  }
  DLCUDA_RETURN_IF_ERROR(branch_->Forward(ctx, input, &branch_output_));
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(input, branch_output_, "Residual input", "Residual branch output"));
  DLCUDA_RETURN_IF_ERROR(TensorAdd(ctx, input, branch_output_, &forward_output_));
  *output = forward_output_;
  return Status::Ok();
}

Status Residual::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!branch_) {
    return Status::InvalidArgument("Residual branch is null");
  }
  if (grad_input == nullptr) {
    return branch_->Backward(ctx, grad_output, nullptr);
  }
  DLCUDA_RETURN_IF_ERROR(branch_->Backward(ctx, grad_output, &branch_grad_));
  DLCUDA_RETURN_IF_ERROR(
      EnsureSameShapeAndType(grad_output, branch_grad_, "Residual grad_output", "branch grad"));
  DLCUDA_RETURN_IF_ERROR(TensorAdd(ctx, grad_output, branch_grad_, &backward_output_));
  *grad_input = backward_output_;
  return Status::Ok();
}

void Residual::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (branch_ == nullptr || out == nullptr) {
    return;
  }
  branch_->AppendParameters(JoinParameterName(prefix, "branch"), out);
}

} // namespace dlcuda
