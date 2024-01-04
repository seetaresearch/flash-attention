#include "../operators/rms_norm_op.h"
#include "../kernels/rms_norm/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RMSNormOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto N = X.count(0, axis), C = X.count(axis);
  INITIALIZE_TENSOR_VIA_SPEC(W, vec64_t({C}), T);

  kernels::RMSNorm(
      N,
      C,
      epsilon_,
      X.template data<T, Context>(),
      W.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RMSNorm);
#endif

OPERATOR_SCHEMA(RMSNorm).NumInputs(2).NumOutputs(1);
REGISTER_GRADIENT(RMSNorm, GenericGradientMaker);

} // namespace dragon
