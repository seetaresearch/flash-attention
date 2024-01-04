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

#ifndef DRAGON_OPERATORS_FLASH_ATTN_OP_IMPL_CNNL_H_
#define DRAGON_OPERATORS_FLASH_ATTN_OP_IMPL_CNNL_H_

#ifdef USE_MLU

#include <cnnl_extra.h>
#include <dragon/core/context_mlu.h>
#include <dragon/core/tensor.h>

namespace dragon {

template <typename T>
vector<T> GetCumLens(int64_t batch_size, int64_t seqlen_q, int64_t seqlen_k) {
  vector<T> qk_seqlens(2 * (batch_size + 1));
  auto* ptr = qk_seqlens.data(); // clang-format off
  for (int v = 0, i = 0; i <= batch_size; v += seqlen_q, ++i) *(ptr++) = v;
  for (int v = 0, i = 0; i <= batch_size; v += seqlen_k, ++i) *(ptr++) = v;
  return qk_seqlens; // clang-format on
}

class CNNLFlashAttnImpl {
 public:
  CNNLFlashAttnImpl() : workspace_size_(0) {
    CNNLCreateTensorDesc(&q_desc_);
    CNNLCreateTensorDesc(&k_desc_);
    CNNLCreateTensorDesc(&v_desc_);
    CNNLCreateTensorDesc(&o_desc_);
    CNNLCreateTensorDesc(&seq_desc_);
    CNNLCreateTensorDesc(&lse_desc_);
    CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&attn_desc_));
  }

  ~CNNLFlashAttnImpl() {
    CNNLDestroyTensorDesc(q_desc_);
    CNNLDestroyTensorDesc(k_desc_);
    CNNLDestroyTensorDesc(v_desc_);
    CNNLDestroyTensorDesc(o_desc_);
    CNNLDestroyTensorDesc(seq_desc_);
    CNNLDestroyTensorDesc(lse_desc_);
    CNNL_CHECK(cnnlDestroyFlashAttentionDescriptor(attn_desc_));
  }

  template <typename T>
  void Setup(
      const int64_t batch_size,
      const int64_t seqlen_q,
      const int64_t seqlen_k,
      const int64_t num_heads_q,
      const int64_t num_heads_k,
      const int64_t head_dim,
      const int64_t causal,
      const float softmax_scale,
      const float dropout,
      Tensor* A_lse,
      Tensor* A_philox,
      Tensor* QK_seqlens,
      MLUContext* ctx) {
    const bool is_causal = causal > 0 && seqlen_q > 1;
    const auto counter_offset = size_t(batch_size * num_heads_q * 32);
    auto qk_seqlens = GetCumLens<int>(batch_size, seqlen_q, seqlen_k);
    CNNLSetTensorDesc<T>(q_desc_, {seqlen_q, num_heads_q, head_dim});
    CNNLSetTensorDesc<T>(k_desc_, {seqlen_k, num_heads_k, head_dim});
    CNNLSetTensorDesc<T>(v_desc_, {seqlen_k, num_heads_k, head_dim});
    CNNLSetTensorDesc<T>(o_desc_, {seqlen_q, num_heads_q, head_dim});
    CNNLSetTensorDesc<int>(seq_desc_, {batch_size + 1});
    CNNLSetTensorDesc<float>(lse_desc_, {num_heads_q, seqlen_q});
    A_lse->Reshape({num_heads_q, seqlen_q}); // clang-format off
    softmax_lse_ptr_ = A_lse->mutable_data<float, MLUContext>();
    rng_state_ptr_ = (size_t*)A_philox->mutable_data<int64_t, CPUContext>();
    if (dropout > 0.f) rng_state_ptr_[1] += ((counter_offset + 3) / 4) * 4;
    q_seqlens_ptr_ = QK_seqlens->CopyFrom<int>(qk_seqlens)->data<int, MLUContext>();
    k_seqlens_ptr_ = q_seqlens_ptr_ + batch_size + 1; // clang-format on
    CNNL_CHECK(cnnlSetFlashAttentionDescriptor(
        attn_desc_,
        CNNL_DTYPE_FLOAT,
        CNNL_ACTIVATION_HIGH_PRECISION,
        is_causal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE,
        true, // is_pack_mode
        false, // is_out_zero
        false, // return_softmax
        seqlen_q, // max_seqlen_q
        seqlen_k, // max_seqlen_k
        dropout,
        softmax_scale));
    CNNL_CHECK(cnnlGetFlashAttentionForwardWorkspaceSize(
        ctx->cnnl_handle(),
        attn_desc_,
        q_desc_,
        k_desc_,
        v_desc_,
        &workspace_size_));
  }

  template <typename T>
  void Compute(
      const T* q,
      const T* k,
      const T* v,
      T* o,
      void* workspace,
      MLUContext* ctx) {
    CNNL_CHECK(cnnlFlashAttentionForward(
        ctx->cnnl_handle(),
        attn_desc_,
        q_desc_,
        q,
        k_desc_,
        k,
        v_desc_,
        v,
        seq_desc_,
        q_seqlens_ptr_,
        seq_desc_,
        k_seqlens_ptr_,
        rng_state_ptr_,
        workspace_size_ > 0 ? workspace : nullptr,
        workspace_size_,
        nullptr,
        nullptr,
        lse_desc_,
        softmax_lse_ptr_,
        o_desc_,
        o));
  }

  size_t workspace_size() {
    return workspace_size_;
  }

 private:
  size_t workspace_size_;
  float* softmax_lse_ptr_;
  size_t* rng_state_ptr_;
  const int *q_seqlens_ptr_, *k_seqlens_ptr_;
  cnnlFlashAttentionDescriptor_t attn_desc_;
  cnnlTensorDescriptor_t q_desc_, k_desc_, v_desc_, o_desc_;
  cnnlTensorDescriptor_t seq_desc_, lse_desc_;
};

class CNNLFlashAttnGradImpl {
 public:
  CNNLFlashAttnGradImpl() : workspace_size_(0) {
    CNNLCreateTensorDesc(&q_desc_);
    CNNLCreateTensorDesc(&k_desc_);
    CNNLCreateTensorDesc(&v_desc_);
    CNNLCreateTensorDesc(&o_desc_);
    CNNLCreateTensorDesc(&seq_desc_);
    CNNLCreateTensorDesc(&lse_desc_);
    CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&attn_desc_));
  }

  ~CNNLFlashAttnGradImpl() {
    CNNLDestroyTensorDesc(q_desc_);
    CNNLDestroyTensorDesc(k_desc_);
    CNNLDestroyTensorDesc(v_desc_);
    CNNLDestroyTensorDesc(o_desc_);
    CNNLDestroyTensorDesc(seq_desc_);
    CNNLDestroyTensorDesc(lse_desc_);
    CNNL_CHECK(cnnlDestroyFlashAttentionDescriptor(attn_desc_));
  }

  template <typename T>
  void Setup(
      const int64_t batch_size,
      const int64_t seqlen_q,
      const int64_t seqlen_k,
      const int64_t num_heads_q,
      const int64_t num_heads_k,
      const int64_t head_dim,
      const int64_t causal,
      const float softmax_scale,
      const float dropout,
      Tensor& A_lse,
      Tensor& A_philox,
      Tensor& QK_seqlens,
      MLUContext* ctx) {
    const bool is_causal = causal > 0 && seqlen_q > 1;
    CNNLSetTensorDesc<T>(q_desc_, {seqlen_q, num_heads_q, head_dim});
    CNNLSetTensorDesc<T>(k_desc_, {seqlen_k, num_heads_k, head_dim});
    CNNLSetTensorDesc<T>(v_desc_, {seqlen_k, num_heads_k, head_dim});
    CNNLSetTensorDesc<T>(o_desc_, {seqlen_q, num_heads_q, head_dim});
    CNNLSetTensorDesc<int>(seq_desc_, {batch_size + 1});
    CNNLSetTensorDesc<float>(lse_desc_, {num_heads_q, seqlen_q});
    softmax_lse_ptr_ = A_lse.data<float, MLUContext>();
    rng_state_ptr_ = (size_t*)A_philox.data<int64_t, CPUContext>();
    q_seqlens_ptr_ = QK_seqlens.data<int, MLUContext>();
    k_seqlens_ptr_ = q_seqlens_ptr_ + batch_size + 1;
    CNNL_CHECK(cnnlSetFlashAttentionBackwardDescriptor(
        attn_desc_,
        CNNL_DTYPE_FLOAT,
        CNNL_ACTIVATION_HIGH_PRECISION,
        is_causal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE,
        true, // is_pack_mode
        false, // is_out_zero
        false, // return_softmax
        seqlen_q, // max_seqlen_q
        seqlen_k, // max_seqlen_k
        dropout,
        softmax_scale));
    CNNL_CHECK(cnnlGetFlashAttentionBackwardWorkspaceSize(
        ctx->cnnl_handle(),
        attn_desc_,
        q_desc_,
        k_desc_,
        v_desc_,
        &workspace_size_));
  }

  template <typename T>
  void Compute(
      const T* q,
      const T* k,
      const T* v,
      const T* o,
      const T* grad_o,
      T* grad_q,
      T* grad_k,
      T* grad_v,
      void* workspace,
      MLUContext* ctx) {
    CNNL_CHECK(cnnlFlashAttentionBackward(
        ctx->cnnl_handle(),
        attn_desc_,
        o_desc_,
        grad_o,
        q_desc_,
        q,
        k_desc_,
        k,
        v_desc_,
        v,
        o_desc_,
        o,
        lse_desc_,
        softmax_lse_ptr_,
        seq_desc_,
        q_seqlens_ptr_,
        seq_desc_,
        k_seqlens_ptr_,
        rng_state_ptr_,
        workspace_size_ > 0 ? workspace : nullptr,
        workspace_size_,
        q_desc_,
        grad_q,
        k_desc_,
        grad_k,
        v_desc_,
        grad_v,
        nullptr,
        nullptr,
        nullptr,
        nullptr));
  }

  size_t workspace_size() {
    return workspace_size_;
  }

 private:
  size_t workspace_size_;
  const float* softmax_lse_ptr_;
  const size_t* rng_state_ptr_;
  const int *q_seqlens_ptr_, *k_seqlens_ptr_;
  cnnlFlashAttentionDescriptor_t attn_desc_;
  cnnlTensorDescriptor_t q_desc_, k_desc_, v_desc_, o_desc_;
  cnnlTensorDescriptor_t seq_desc_, lse_desc_;
};

} // namespace dragon

#endif // USE_MLU

#endif // DRAGON_OPERATORS_FLASH_ATTN_OP_IMPL_CNNL_H_
