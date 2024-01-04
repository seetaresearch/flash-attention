#include "../operators/swiglu_op.h"
#include "../kernels/swiglu/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SwiGLUOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto Y_dims = X.dims();
  Y_dims.back() /= 2;
  kernels::SwiGLU(
      X.count(0, X.axis(-1)),
      X.dim(-1),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SwiGLUGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::SwiGLUGrad(
      X.count(0, X.axis(-1)),
      X.dim(-1),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SwiGLU);
DEPLOY_CUDA_OPERATOR(SwiGLUGradient);
#endif

OPERATOR_SCHEMA(SwiGLU)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SwiGLUGradient)
    /* X, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(SwiGLU, GenericGradientMaker);

} // namespace dragon
