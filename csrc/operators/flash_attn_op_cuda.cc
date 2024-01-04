#ifdef USE_CUDA

#include <dragon/core/workspace.h>
#include "../operators/flash_attn_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CUDAFlashAttnOp<Context>::DoRunWithType() {
  auto &Q = Input(0), &K = Input(1), &V = Input(2);
  auto* O = Output(0)->ReshapeLike(Q);
  CUDASetFlashAttnParams<T>(
      Q.dim(0), // batch_size
      Q.dim(1), // seqlen_q
      K.dim(1), // seqlen_k
      Q.dim(2), // num_heads_q
      K.dim(2), // num_heads_k
      Q.dim(3), // head_dim
      causal_,
      window_size_left_,
      window_size_right_,
      softmax_scale_,
      dropout_,
      Q,
      K,
      V,
      O,
      Output("A_lse"),
      Output("A_philox"),
      nullptr, // cu_seqlen_q
      nullptr, // cu_seqlen_k
      params_);
  if (InputSize() > 3) { // KVCache inference.
    auto &K_new = Input(3), &V_new = Input(4), &L = Input(5);
    CUDASetFlashAttnKVCacheParams<T>(K_new.dim(1), L, K_new, V_new, params_);
  }
  if (dropout_ > 0.f) {
    auto counter_offset = params_.b * params_.h * 32;
    philox_state_.offset_.val += ((counter_offset + 3) / 4) * 4;
    params_.philox_args = philox_state_;
  } else {
    size_t workspace_size;
    const auto& dprop = CUDAGetDeviceProp(ctx()->device());
    CUDASetFlashAttnHeuristicSplits(params_, dprop.multiProcessorCount, 128);
    CUDAGetFlashAttnWorkspaceSize(params_, &workspace_size);
    CUDASetFlashAttnWorkspace(
        params_, ctx()->workspace()->template data<Context>(workspace_size));
  }
  kernels::FlashAttn(params_, ctx());
}

template <class Context>
template <typename T>
void CUDAFlashAttnGradientOp<Context>::DoRunWithType() {
  auto &Q = Input(0), &K = Input(1), &V = Input(2);
  auto &O = Input(3), &dO = Input(4);
  CUDASetFlashAttnGradParams<T>(
      Q.dim(0), // batch_size
      Q.dim(1), // seqlen_q
      K.dim(1), // seqlen_k
      Q.dim(2), // num_heads_q
      K.dim(2), // num_heads_k
      Q.dim(3), // head_dim
      causal_,
      window_size_left_,
      window_size_right_,
      softmax_scale_,
      dropout_,
      Q,
      K,
      V,
      O,
      dO,
      Output(0)->ReshapeLike(Q),
      Output(1)->ReshapeLike(K),
      Output(2)->ReshapeLike(V),
      &Input("A_lse"),
      &Input("A_philox"),
      nullptr, // cu_seqlen_q
      nullptr, // cu_seqlen_k
      params_);
  size_t workspace_size;
  CUDAGetFlashAttnGradWorkspaceSize(params_, &workspace_size);
  CUDASetFlashAttnGradWorkspace(
      params_, ctx()->workspace()->template data<Context>(workspace_size));
  kernels::FlashAttnGrad(params_, ctx());
}

INSTANTIATE_OPERATOR(CUDAFlashAttn, CUDAContext);
INSTANTIATE_OPERATOR(CUDAFlashAttnGradient, CUDAContext);
REGISTER_CUDA_OPERATOR(FlashAttn, CUDAFlashAttnOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(FlashAttnGradient, CUDAFlashAttnGradientOp<CUDAContext>);

} // namespace dragon

#endif // USE_CUDA
