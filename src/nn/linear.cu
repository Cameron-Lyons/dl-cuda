#include "detail/linear_kernels.cuh"

namespace dlcuda {

Linear::Linear(int64_t in_features, int64_t out_features, RuntimeContext &ctx, DType dtype)
    : in_features_(in_features), out_features_(out_features), dtype_(dtype) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    init_status_ = Status::InvalidArgument("Linear dimensions must be positive");
    return;
  }
  init_status_ = ValidateFloatingDType(dtype_, "Linear");
  if (!init_status_.ok()) {
    return;
  }

  auto weight = Tensor::AllocateAsync({in_features_, out_features_}, dtype_, ctx.stream());
  if (!weight.ok()) {
    init_status_ = weight.status();
    return;
  }
  auto bias = Tensor::AllocateAsync({out_features_}, dtype_, ctx.stream());
  if (!bias.ok()) {
    init_status_ = bias.status();
    return;
  }
  auto grad_weight =
      Tensor::AllocateAsync({in_features_, out_features_}, DType::kFloat32, ctx.stream());
  if (!grad_weight.ok()) {
    init_status_ = grad_weight.status();
    return;
  }
  auto grad_bias = Tensor::AllocateAsync({out_features_}, DType::kFloat32, ctx.stream());
  if (!grad_bias.ok()) {
    init_status_ = grad_bias.status();
    return;
  }

  weight_ = weight.value();
  bias_ = bias.value();
  grad_weight_ = grad_weight.value();
  grad_bias_ = grad_bias.value();

  init_status_ = InitializeWeightBiasAndGradients(ctx, &weight_, &bias_, &grad_weight_, &grad_bias_,
                                                  static_cast<float>(in_features_));
}

Linear::~Linear() = default;

Status Linear::Forward(RuntimeContext &ctx, const Tensor &input, Tensor *output) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  if (output == nullptr) {
    return Status::InvalidArgument("Linear::Forward output is null");
  }

  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(input, "Linear input"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(input, dtype_, "Linear input"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(input, 2, "Linear input"));

  int64_t batch = input.dim(0);
  int64_t in_features = input.dim(1);
  if (in_features != in_features_) {
    std::ostringstream oss;
    oss << "Linear input feature mismatch: expected " << in_features_ << " got " << in_features;
    return Status::InvalidArgument(oss.str());
  }

  DLCUDA_RETURN_IF_ERROR(
      EnsureTensorAsync(&forward_output_, {batch, out_features_}, dtype_, ctx.stream()));
  cached_input_ = input;
  last_batch_ = batch;
  if (batch == 0) {
    *output = forward_output_;
    return Status::Ok();
  }

  bool used_accelerated = false;
  if (ctx.use_cublas()) {
    auto out_features_int = detail::CheckedInt(out_features_, "out_features");
    if (!out_features_int.ok()) {
      return out_features_int.status();
    }
    auto batch_int = detail::CheckedInt(batch, "batch");
    if (!batch_int.ok()) {
      return batch_int.status();
    }
    auto in_features_int = detail::CheckedInt(in_features_, "in_features");
    if (!in_features_int.ok()) {
      return in_features_int.status();
    }

#if defined(DLCUDA_HAS_CUBLASLT)
    if (cublaslt_forward_plan_ == nullptr) {
      cublaslt_forward_plan_ = std::make_unique<LinearCublasLtForwardPlan>();
    }
    Status lt_status = LinearForwardCublasLt(
        ctx, input, weight_, bias_, &forward_output_, out_features_int.value(), batch_int.value(),
        in_features_int.value(), dtype_, cublaslt_forward_plan_.get());
    if (lt_status.ok()) {
      used_accelerated = true;
    } else if (lt_status.code() != StatusCode::kUnsupported) {
      return lt_status;
    }
#endif
    if (!used_accelerated && dtype_ == DType::kFloat32) {
      DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
      cublasHandle_t handle = ctx.cublas_handle();
      const float alpha = 1.0f;
      const float beta = 0.0f;
      DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features_int.value(), batch_int.value(),
                      in_features_int.value(), &alpha, weight_.data_as<float>(),
                      out_features_int.value(), input.data_as<float>(), in_features_int.value(),
                      &beta, forward_output_.data_as<float>(), out_features_int.value()),
          "Linear forward cublasSgemm"));

      int64_t total = batch * out_features_;
      auto blocks = detail::BlocksForElements(total, kCudaThreads);
      if (!blocks.ok()) {
        return blocks.status();
      }
      AddBiasKernel<<<blocks.value(), kCudaThreads, 0, ctx.stream()>>>(
          forward_output_.data_as<float>(), bias_.data_as<float>(), batch, out_features_);
      DLCUDA_RETURN_IF_ERROR(detail::CheckKernelLaunch("Linear add-bias kernel"));
      used_accelerated = true;
    }
  }

  if (!used_accelerated) {
    auto x_blocks = detail::BlocksForElements(out_features_, kLinearTile);
    if (!x_blocks.ok()) {
      return x_blocks.status();
    }
    auto y_blocks = detail::BlocksForElements(batch, kLinearTile);
    if (!y_blocks.ok()) {
      return y_blocks.status();
    }
    dim3 threads(kLinearTile, kLinearTile);
    dim3 blocks(static_cast<unsigned int>(x_blocks.value()),
                static_cast<unsigned int>(y_blocks.value()));
    DLCUDA_RETURN_IF_ERROR(LaunchLinearForwardKernel(ctx, dtype_, input, weight_, bias_,
                                                     &forward_output_, blocks, threads, batch,
                                                     in_features_, out_features_));
  }

  *output = forward_output_;
  return Status::Ok();
}

Status Linear::Backward(RuntimeContext &ctx, const Tensor &grad_output, Tensor *grad_input) {
  if (!init_status_.ok()) {
    return init_status_;
  }
  bool need_grad_input = grad_input != nullptr;

  DLCUDA_RETURN_IF_ERROR(ValidateFloatingTensor(grad_output, "Linear grad_output"));
  DLCUDA_RETURN_IF_ERROR(EnsureDType(grad_output, dtype_, "Linear grad_output"));
  DLCUDA_RETURN_IF_ERROR(ValidateRank(grad_output, 2, "Linear grad_output"));
  if (grad_output.dim(0) != last_batch_ || grad_output.dim(1) != out_features_) {
    return Status::InvalidArgument("Linear grad_output shape mismatch");
  }
  if (!cached_input_.defined()) {
    return Status::RuntimeError("Linear backward called before forward");
  }

  if (need_grad_input) {
    DLCUDA_RETURN_IF_ERROR(
        EnsureTensorAsync(&backward_output_, {last_batch_, in_features_}, dtype_, ctx.stream()));
  }
  if (last_batch_ == 0) {
    DLCUDA_RETURN_IF_ERROR(grad_weight_.FillZero(ctx.stream()));
    DLCUDA_RETURN_IF_ERROR(grad_bias_.FillZero(ctx.stream()));
    if (need_grad_input) {
      *grad_input = backward_output_;
    }
    return Status::Ok();
  }

  bool used_accelerated = false;
  if (ctx.use_cublas()) {
    auto in_features_int = detail::CheckedInt(in_features_, "in_features");
    if (!in_features_int.ok()) {
      return in_features_int.status();
    }
    auto batch_int = detail::CheckedInt(last_batch_, "batch");
    if (!batch_int.ok()) {
      return batch_int.status();
    }
    auto out_features_int = detail::CheckedInt(out_features_, "out_features");
    if (!out_features_int.ok()) {
      return out_features_int.status();
    }

    DLCUDA_RETURN_IF_ERROR(ctx.EnsureCublas());
    cublasHandle_t handle = ctx.cublas_handle();

    if (dtype_ == DType::kFloat32) {
      const float alpha = 1.0f;
      const float beta = 0.0f;

      if (need_grad_input) {
        DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features_int.value(),
                        batch_int.value(), out_features_int.value(), &alpha,
                        weight_.data_as<float>(), out_features_int.value(),
                        grad_output.data_as<float>(), out_features_int.value(), &beta,
                        backward_output_.data_as<float>(), in_features_int.value()),
            "Linear backward-input cublasSgemm"));
      }

      DLCUDA_RETURN_IF_ERROR(detail::CublasStatus(
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features_int.value(),
                      in_features_int.value(), batch_int.value(), &alpha,
                      grad_output.data_as<float>(), out_features_int.value(),
                      cached_input_.data_as<float>(), in_features_int.value(), &beta,
                      grad_weight_.data_as<float>(), out_features_int.value()),
          "Linear backward-weight cublasSgemm"));
      used_accelerated = true;
    }

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    if (!used_accelerated && (dtype_ == DType::kFloat16 || dtype_ == DType::kBFloat16)) {
      bool gemm_supported = true;
      if (need_grad_input) {
        Status input_status = LinearCublasGemmEx(
            ctx, CUBLAS_OP_T, CUBLAS_OP_N, in_features_int.value(), batch_int.value(),
            out_features_int.value(), weight_, dtype_, out_features_int.value(), grad_output,
            dtype_, out_features_int.value(), &backward_output_, dtype_, in_features_int.value(),
            "Linear backward-input cublasGemmEx");
        if (!input_status.ok()) {
          if (input_status.code() == StatusCode::kUnsupported) {
            gemm_supported = false;
          } else {
            return input_status;
          }
        }
      }

      if (gemm_supported) {
        Status weight_status = LinearCublasGemmEx(
            ctx, CUBLAS_OP_N, CUBLAS_OP_T, out_features_int.value(), in_features_int.value(),
            batch_int.value(), grad_output, dtype_, out_features_int.value(), cached_input_, dtype_,
            in_features_int.value(), &grad_weight_, DType::kFloat32, out_features_int.value(),
            "Linear backward-weight cublasGemmEx");
        if (weight_status.ok()) {
          used_accelerated = true;
        } else if (weight_status.code() != StatusCode::kUnsupported) {
          return weight_status;
        }
      }
    }
#endif

    if (used_accelerated) {
      auto rows = detail::RowsForGrid(out_features_, "linear bias");
      if (!rows.ok()) {
        return rows.status();
      }
      DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardBiasKernel(
          ctx, dtype_, grad_output, &grad_bias_, rows.value(), last_batch_, out_features_));
    }
  }

  if (!used_accelerated) {
    dim3 threads(kLinearTile, kLinearTile);
    if (need_grad_input) {
      auto input_x_blocks = detail::BlocksForElements(in_features_, kLinearTile);
      if (!input_x_blocks.ok()) {
        return input_x_blocks.status();
      }
      auto input_y_blocks = detail::BlocksForElements(last_batch_, kLinearTile);
      if (!input_y_blocks.ok()) {
        return input_y_blocks.status();
      }
      dim3 blocks_input(static_cast<unsigned int>(input_x_blocks.value()),
                        static_cast<unsigned int>(input_y_blocks.value()));
      DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardInputKernel(
          ctx, dtype_, grad_output, weight_, &backward_output_, blocks_input, threads, last_batch_,
          in_features_, out_features_));
    }

    auto weight_x_blocks = detail::BlocksForElements(out_features_, kLinearTile);
    if (!weight_x_blocks.ok()) {
      return weight_x_blocks.status();
    }
    auto weight_y_blocks = detail::BlocksForElements(in_features_, kLinearTile);
    if (!weight_y_blocks.ok()) {
      return weight_y_blocks.status();
    }
    dim3 blocks_weight(static_cast<unsigned int>(weight_x_blocks.value()),
                       static_cast<unsigned int>(weight_y_blocks.value()));
    DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardWeightKernel(
        ctx, dtype_, cached_input_, grad_output, &grad_weight_, blocks_weight, threads, last_batch_,
        in_features_, out_features_));

    auto bias_rows = detail::RowsForGrid(out_features_, "linear bias");
    if (!bias_rows.ok()) {
      return bias_rows.status();
    }
    DLCUDA_RETURN_IF_ERROR(LaunchLinearBackwardBiasKernel(
        ctx, dtype_, grad_output, &grad_bias_, bias_rows.value(), last_batch_, out_features_));
  }

  if (need_grad_input) {
    *grad_input = backward_output_;
  }
  return Status::Ok();
}

void Linear::AppendParameters(const std::string &prefix, std::vector<ParameterRef> *out) {
  if (out == nullptr) {
    return;
  }
  out->push_back(ParameterRef{JoinParameterName(prefix, "weight"), &weight_, &grad_weight_});
  out->push_back(ParameterRef{JoinParameterName(prefix, "bias"), &bias_, &grad_bias_});
}

} // namespace dlcuda
