#ifdef USE_MLU

#include <dragon/core/workspace.h>
#include "../operators/flash_attn_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLFlashAttnOp<Context>::DoRunWithType() {
  auto &Q = Input(0), &K = Input(1), &V = Input(2);
  auto *O = Output(0)->ReshapeLike(Q), *A_philox = Output("A_philox");
  if (A_philox->empty()) {
    auto* philox_args = reinterpret_cast<size_t*>(
        A_philox->Reshape({2})->template mutable_data<int64_t, CPUContext>());
    philox_args[0] = size_t(ctx()->random_seed()), philox_args[1] = size_t(0);
  }
  impl_.Setup<T>(
      Q.dim(0), // batch_size
      Q.dim(1), // seqlen_q
      K.dim(1), // seqlen_k
      Q.dim(2), // num_heads_q
      K.dim(2), // num_heads_k
      Q.dim(3), // head_dim
      causal_,
      softmax_scale_,
      dropout_,
      Output("A_lse"),
      A_philox,
      Output("QK_seqlens"),
      ctx());
  impl_.Compute<T>(
      Q.template data<T, MLUContext>(),
      K.template data<T, MLUContext>(),
      V.template data<T, MLUContext>(),
      O->template mutable_data<T, MLUContext>(),
      ctx()->workspace()->template data<MLUContext>(impl_.workspace_size()),
      ctx());
}

template <class Context>
template <typename T>
void CNNLFlashAttnGradientOp<Context>::DoRunWithType() {
  auto &Q = Input(0), &K = Input(1), &V = Input(2);
  auto &O = Input(3), &dO = Input(4);
  impl_.Setup<T>(
      Q.dim(0), // batch_size
      Q.dim(1), // seqlen_q
      K.dim(1), // seqlen_k
      Q.dim(2), // num_heads_q
      K.dim(2), // num_heads_k
      Q.dim(3), // head_dim
      causal_,
      softmax_scale_,
      dropout_,
      Input("A_lse"),
      Input("A_philox"),
      Input("QK_seqlens"),
      ctx());
  impl_.Compute<T>(
      Q.template data<T, MLUContext>(),
      K.template data<T, MLUContext>(),
      V.template data<T, MLUContext>(),
      O.template data<T, MLUContext>(),
      dO.template data<T, MLUContext>(),
      Output(0)->ReshapeLike(Q)->template mutable_data<T, MLUContext>(),
      Output(1)->ReshapeLike(K)->template mutable_data<T, MLUContext>(),
      Output(2)->ReshapeLike(V)->template mutable_data<T, MLUContext>(),
      ctx()->workspace()->template data<MLUContext>(impl_.workspace_size()),
      ctx());
}

INSTANTIATE_OPERATOR(CNNLFlashAttn, MLUContext);
INSTANTIATE_OPERATOR(CNNLFlashAttnGradient, MLUContext);
REGISTER_CNNL_OPERATOR(FlashAttn, CNNLFlashAttnOp<MLUContext>);
REGISTER_CNNL_OPERATOR(FlashAttnGradient, CNNLFlashAttnGradientOp<MLUContext>);

} // namespace dragon

#endif // USE_MLU
