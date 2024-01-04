/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_FLASH_ATTN_OP_IMPL_CUDA_H_
#define DRAGON_OPERATORS_FLASH_ATTN_OP_IMPL_CUDA_H_

#ifdef USE_CUDA

#include <dragon/core/tensor.h>
#include "../kernels/flash_attn/op_kernels.h"

namespace dragon {

template <typename T>
void CUDASetFlashAttnParams(
    const int64_t batch_size,
    const int64_t seqlen_q,
    const int64_t seqlen_k,
    const int64_t num_heads_q,
    const int64_t num_heads_k,
    const int64_t head_dim,
    const int64_t causal,
    const int64_t window_size_left,
    const int64_t window_size_right,
    const float softmax_scale,
    const float dropout,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor* O,
    Tensor* A_lse,
    Tensor* A_philox,
    void* cu_seqlens_q,
    void* cu_seqlens_k,
    Flash_fwd_params& params) {
  params.is_bf16 = TypeMeta::Id<T>() != TypeMeta::Id<float16>();
  params.q_ptr = const_cast<T*>(Q.data<T, CUDAContext>());
  params.k_ptr = const_cast<T*>(K.data<T, CUDAContext>());
  params.v_ptr = const_cast<T*>(V.data<T, CUDAContext>());
  params.q_row_stride = Q.stride(-3);
  params.k_row_stride = K.stride(-3);
  params.v_row_stride = V.stride(-3);
  params.q_head_stride = Q.stride(-2);
  params.k_head_stride = K.stride(-2);
  params.v_head_stride = V.stride(-2);
  params.o_ptr = O->mutable_data<T, CUDAContext>();
  params.o_row_stride = O->stride(-3);
  params.o_head_stride = O->stride(-2);
  if (cu_seqlens_q == nullptr) {
    params.q_batch_stride = Q.stride(0);
    params.k_batch_stride = K.stride(0);
    params.v_batch_stride = V.stride(0);
    params.o_batch_stride = O->stride(0);
  }
  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k);
  // P = softmax(QK^T)
  params.p_ptr = nullptr;
  // Softmax sum
  A_lse->Reshape({batch_size, num_heads_q, seqlen_q});
  params.softmax_lse_ptr = A_lse->mutable_data<float, CUDAContext>();
  // Set the dimensions.
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  params.b = batch_size;
  params.h = num_heads_q;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = num_heads_q / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_dim;
  params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
  params.seqlen_k_rounded = round_multiple(seqlen_k, 128);
  params.d_rounded = round_multiple(head_dim, 32);
  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - dropout;
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  params.is_seqlens_k_cumulative = true;
  params.is_causal = causal && seqlen_q > 1;
  params.window_size_left = params.is_causal ? seqlen_k : window_size_left;
  params.window_size_right = params.is_causal ? 0 : window_size_right;
  if (window_size_left < 0 && window_size_right >= 0) {
    params.window_size_left = seqlen_k;
  }
  if (window_size_right < 0 && window_size_left >= 0) {
    params.window_size_right = seqlen_k;
  }
  params.num_splits = 1;
  // Set philox state.
  A_philox->Reshape({2});
  params.rng_state = (uint64_t*)A_philox->mutable_data<int64_t, CUDAContext>();
}

template <typename T>
void CUDASetFlashAttnKVCacheParams(
    const int64_t seqlen_new,
    Tensor& seqlen_cur,
    Tensor& K_new,
    Tensor& V_new,
    Flash_fwd_params& params) {
  CHECK_GE(seqlen_cur.count(), K_new.dim(0)) << "\nInsufficient seqlens.";
  params.seqlen_knew = K_new.dim(1);
  params.knew_ptr = const_cast<T*>(K_new.data<T, CUDAContext>());
  params.vnew_ptr = const_cast<T*>(V_new.data<T, CUDAContext>());
  params.knew_batch_stride = K_new.stride(0);
  params.vnew_batch_stride = V_new.stride(0);
  params.knew_row_stride = K_new.stride(-3);
  params.vnew_row_stride = V_new.stride(-3);
  params.knew_head_stride = K_new.stride(-2);
  params.vnew_head_stride = V_new.stride(-2);
  params.cu_seqlens_k = const_cast<int*>(seqlen_cur.data<int, CUDAContext>());
  params.is_seqlens_k_cumulative = false;
  params.rotary_dim = 0;
}

template <typename T>
void CUDASetFlashAttnGradParams(
    const int64_t batch_size,
    const int64_t seqlen_q,
    const int64_t seqlen_k,
    const int64_t num_heads_q,
    const int64_t num_heads_k,
    const int64_t head_dim,
    const int64_t causal,
    const int64_t window_size_left,
    const int64_t window_size_right,
    const float softmax_scale,
    const float dropout,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& O,
    Tensor& dO,
    Tensor* dQ,
    Tensor* dK,
    Tensor* dV,
    Tensor* A_lse,
    Tensor* A_philox,
    void* cu_seqlens_q,
    void* cu_seqlens_k,
    Flash_bwd_params& params) {
  CUDASetFlashAttnParams<T>(
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      causal,
      window_size_left,
      window_size_right,
      softmax_scale,
      dropout,
      Q,
      K,
      V,
      &O,
      A_lse,
      A_philox,
      cu_seqlens_q,
      cu_seqlens_k,
      params);
  // Fix stride if “O” is reshaped for projection.
  params.o_row_stride = dO.stride(-3);
  params.o_head_stride = dO.stride(-2);
  // Set the pointers and strides.
  params.do_ptr = const_cast<T*>(dO.data<T, CUDAContext>());
  params.do_row_stride = dO.stride(-3);
  params.do_head_stride = dO.stride(-2);
  params.dq_ptr = dQ->mutable_data<T, CUDAContext>();
  params.dk_ptr = dK->mutable_data<T, CUDAContext>();
  params.dv_ptr = dV->mutable_data<T, CUDAContext>();
  params.dq_row_stride = dQ->stride(-3);
  params.dk_row_stride = dK->stride(-3);
  params.dv_row_stride = dV->stride(-3);
  params.dq_head_stride = dQ->stride(-2);
  params.dk_head_stride = dK->stride(-2);
  params.dv_head_stride = dV->stride(-2);
  if (cu_seqlens_q == nullptr) {
    params.do_batch_stride = dO.stride(0);
    params.dq_batch_stride = dQ->stride(0);
    params.dk_batch_stride = dK->stride(0);
    params.dv_batch_stride = dV->stride(0);
  }
}

inline int CUDASetFlashAttnHeuristicSplits(
    Flash_fwd_params& params,
    int num_SMs,
    int max_splits) {
  const int block_n = params.d <= 64 ? 256 : (params.d <= 128 ? 128 : 64);
  const int num_n_blocks = (params.seqlen_k + block_n - 1) / block_n;
  const int num_m_blocks = (params.seqlen_q + 64 - 1) / 64;
  const auto batch_nheads_mblocks = params.b * params.h * num_m_blocks;
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) return params.num_splits = 1;
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 ||
        ceildiv(num_n_blocks, num_splits) !=
        ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      if (eff > max_efficiency) max_efficiency = eff;
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) continue;
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      return params.num_splits = num_splits;
    }
  }
  return params.num_splits = 1;
}

inline void CUDAGetFlashAttnWorkspaceSize(
    Flash_fwd_params& params,
    size_t* size) {
  const auto lse_count = params.b * params.h * params.seqlen_q_rounded;
  const auto o_accum_count = lse_count * params.d_rounded;
  const auto num_storages = params.num_splits > 1 ? params.num_splits : 0;
  *size = sizeof(float) * size_t(lse_count) * num_storages;
  *size += sizeof(float) * size_t(o_accum_count) * num_storages;
}

inline void CUDAGetFlashAttnGradWorkspaceSize(
    Flash_bwd_params& params,
    size_t* size) {
  const auto dlse_count = params.b * params.h * params.seqlen_q_rounded;
  const auto dq_accum_count = dlse_count * params.d_rounded;
  *size = sizeof(float) * size_t(dlse_count);
  *size += sizeof(float) * size_t(dq_accum_count);
}

inline void CUDASetFlashAttnWorkspace(
    Flash_fwd_params& params,
    void* workspace) {
  const auto lse_count = params.b * params.h * params.seqlen_q_rounded;
  const auto num_storages = params.num_splits > 1 ? params.num_splits : 0;
  params.oaccum_ptr = static_cast<float*>(workspace) + lse_count * num_storages;
  params.softmax_lseaccum_ptr = workspace;
}

inline void CUDASetFlashAttnGradWorkspace(
    Flash_bwd_params& params,
    void* workspace) {
  const auto dlse_count = params.b * params.h * params.seqlen_q_rounded;
  params.dq_accum_ptr = static_cast<float*>(workspace) + dlse_count;
  params.dk_accum_ptr = params.dv_accum_ptr = nullptr;
  params.dsoftmax_sum = workspace;
}

} // namespace dragon

#endif // USE_CUDA

#endif // DRAGON_OPERATORS_FLASH_ATTN_OP_IMPL_CUDA_H_
