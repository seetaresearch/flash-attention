#ifdef USE_CUDA

#include <dragon/core/workspace.h>
#include "../operators/fused_dense_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CUDAFusedDenseOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  auto has_bias = InputSize() > 2, save_aux = OutputSize() > 1;

  auto Y_dims = X.dims();
  Y_dims.back() = W.dim(0);

  if (col_parallel_ && seq_parallel_) {
    LOG(FATAL) << "Unsupported sequence parallelism."; // X = AllGather(X)
  }

  if (epilogue_.empty()) {
    Y_impl_.SetEpilogueAndBias(
        has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT,
        has_bias ? (T*)Input(2).template data<T, Context>() : nullptr);
  } else if (epilogue_ == "GELU" && !save_aux) {
    Y_impl_.SetEpilogueAndBias(
        has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU,
        has_bias ? (T*)Input(2).template data<T, Context>() : nullptr);
  } else if (epilogue_ == "RELU" && !save_aux) {
    Y_impl_.SetEpilogueAndBias(
        has_bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU,
        has_bias ? (T*)Input(2).template data<T, Context>() : nullptr);
  } else if (epilogue_ == "GELU" && save_aux) {
    Y_impl_.SetEpilogueAndBias(
        has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX,
        has_bias ? (T*)Input(2).template data<T, Context>() : nullptr,
        Output(1)->Reshape(Y_dims)->template mutable_data<T, Context>(),
        Y_dims.back());
  } else if (epilogue_ == "RELU" && save_aux) {
    auto nbytes = math::utils::DivUp(math::utils::Prod(Y_dims), int64_t(8));
    Y_impl_.SetEpilogueAndBias(
        has_bias ? CUBLASLT_EPILOGUE_RELU_AUX_BIAS : CUBLASLT_EPILOGUE_RELU_AUX,
        has_bias ? (T*)Input(2).template data<T, Context>() : nullptr,
        Output(1)->Reshape({nbytes})->template mutable_data<uint8_t, Context>(),
        Y_dims.back());
  } else {
    LOG(FATAL) << "Unsupported epilogue: " << epilogue_;
  }
  Y_impl_.Setup<T>(0, 1, 1.f, 0.f, X.dims(), W.dims(), ctx());
  Y_impl_.Compute<T>(
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(Y_impl_.scratch_size()),
      ctx());

  if (row_parallel_ && seq_parallel_) {
    LOG(FATAL) << "Unsupported sequence parallelism."; // Y = ReduceScatter(Y)
  } else if (row_parallel_ && !seq_parallel_) {
    coll_impl_.AllReduce(
        Y->template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        Y->count(),
        ctx()->cuda_stream());
  }
}

template <class Context>
template <typename T>
void CUDAFusedDenseGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(2), &X_aux = Input(3);
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto has_bgrad = dB->has_name(), has_aux = X_aux.has_name();

  auto N = dY.dim(-1), K = X.dim(-1), M = dY.count() / dY.dim(-1);

  if (row_parallel_ && seq_parallel_) {
    LOG(FATAL) << "Unsupported sequence parallelism."; // X = AllGather(X)
  }

  if (dX->has_name()) {
    if (has_aux) {
      auto* aux = (void*)X_aux.template raw_data<Context>();
#if CUBLAS_VERSION >= 11800 // CUDA >= 11.6
      auto epilogue = CUBLASLT_EPILOGUE_DRELU;
      epilogue = X_aux.ndim() > 1 ? CUBLASLT_EPILOGUE_DGELU : epilogue;
      dX_impl_.SetEpilogueAndBias(epilogue, (T*)nullptr, aux, K);
#else // CUDA < 11.6
      auto* buffer = ctx()->workspace()->template data<T, Context>(K, "Buffer");
      auto epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;
      epilogue = X_aux.ndim() > 1 ? CUBLASLT_EPILOGUE_DGELU_BGRAD : epilogue;
      dX_impl_.SetEpilogueAndBias(epilogue, buffer, aux, K);
#endif
    } else {
      dX_impl_.SetEpilogueAndBias(CUBLASLT_EPILOGUE_DEFAULT, (T*)nullptr);
    }
    dX_impl_.Setup<T>(0, 0, 1.f, 0.f, dY.dims(), W.dims(), ctx());
    dX_impl_.Compute<T>(
        dY.template data<T, Context>(),
        W.template data<T, Context>(),
        dX->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(dX_impl_.scratch_size()),
        ctx());
  }

  if (dW->has_name()) {
    dW_impl_.SetEpilogueAndBias(CUBLASLT_EPILOGUE_DEFAULT, (T*)nullptr);
    if (has_bgrad) {
#if CUBLAS_VERSION >= 11600 // CUDA >= 11.4.2
      has_bgrad = false;
      dW_impl_.SetEpilogueAndBias(
          CUBLASLT_EPILOGUE_BGRADB,
          dB->Reshape({N})->template mutable_data<T, Context>());
#endif
    }
    dW_impl_.Setup<T>(1, 0, 1.f, 0.f, dY.dims(), X.dims(), ctx());
    dW_impl_.Compute<T>(
        dY.template data<T, Context>(),
        X.template data<T, Context>(),
        dW->ReshapeLike(W)->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(dX_impl_.scratch_size()),
        ctx());
  }

  if (has_bgrad) {
    auto* Z = ctx()->workspace()->CreateTensor("Ones");
    if (Z->count() < M) {
      math::Set(
          M,
          convert::To<T>(1.f),
          Z->Reshape({M})->template mutable_data<T, Context>(),
          ctx());
    }
    math::Gemv(
        CblasTrans,
        M,
        N,
        1.f,
        dY.template data<T, Context>(),
        Z->template mutable_data<T, Context>(),
        0.f,
        dB->Reshape({N})->template mutable_data<T, Context>(),
        ctx());
  }

  if (col_parallel_ && seq_parallel_) {
    LOG(FATAL) << "Unsupported sequence parallelism."; // dX = ReduceScatter(dX)
  } else if (col_parallel_ && !seq_parallel_) {
    coll_impl_.AllReduce(
        dX->template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        dX->count(),
        ctx()->cuda_stream());
  }
}

INSTANTIATE_OPERATOR(CUDAFusedDense, CUDAContext);
INSTANTIATE_OPERATOR(CUDAFusedDenseGradient, CUDAContext);
REGISTER_CUDA_OPERATOR(FusedDense, CUDAFusedDenseOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    FusedDenseGradient,
    CUDAFusedDenseGradientOp<CUDAContext>);

OPERATOR_SCHEMA(FusedDense)
    /* X, W, B, X_Aux */
    .NumInputs(2, 4)
    /* Y, Y_aux */
    .NumOutputs(1, 2);

OPERATOR_SCHEMA(FusedDenseGradient)
    /* X, W, dY, X_Aux */
    .NumInputs(4)
    /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0), I(3)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(FusedDense, GradientMaker);

} // namespace dragon

#endif // USE_CUDA
