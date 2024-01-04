#include "../operators/kv_cache_op.h"
#include "../kernels/kv_cache/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void KVCacheOp<Context>::DoRunWithType() {
  auto &XK = Input(0), &XV = Input(1);
  auto &K_cache = Input(2), &V_cache = Input(3), &L = Input(4);
  auto *YK = Output(0), *YV = Output(1);

  auto Y_dims = XK.dims();
  const auto x_seqlen = XK.dim(data_format() == "NLHD" ? 1 : -2);
  const auto y_seqlen = L.template data<int, CPUContext>()[0] + x_seqlen;
  Y_dims[XK.axis(data_format() == "NLHD" ? 1 : -2)] = y_seqlen;
  YK->Reshape(Y_dims), YV->Reshape(Y_dims);

  T *xk = nullptr, *xv = nullptr, *yk = nullptr, *yv = nullptr;
  auto compute_format = data_format();

  kernels::KVCache(
      YK->dim(0), // N
      YK->dim(data_format() == "NLHD" ? 1 : -2), // L
      YK->dim(data_format() == "NLHD" ? -2 : 1), // H
      YK->dim(-1), // D
      y_seqlen - x_seqlen, // seqlen
      K_cache.dim(data_format() == "NLHD" ? 1 : -2), // max_seqlen
      compute_format,
      xk != nullptr ? xk : XK.template data<T, Context>(),
      xv != nullptr ? xv : XV.template data<T, Context>(),
      K_cache.template mutable_data<T, Context>(),
      V_cache.template mutable_data<T, Context>(),
      yk != nullptr ? yk : YK->template mutable_data<T, Context>(),
      yv != nullptr ? yv : YV->template mutable_data<T, Context>(),
      ctx());
}

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(KVCache);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(KVCache);
#endif

OPERATOR_SCHEMA(KVCache).NumInputs(5).NumOutputs(2);
NO_GRADIENT(KVCache);

} // namespace dragon
