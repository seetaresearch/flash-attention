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

#ifndef DRAGON_OPERATORS_FLASH_ATTN_OP_H_
#define DRAGON_OPERATORS_FLASH_ATTN_OP_H_

#include <dragon/core/operator.h>
#include "../operators/flash_attn_op_impl_cnnl.h"
#include "../operators/flash_attn_op_impl_cuda.h"

namespace dragon {

#ifdef USE_CUDA
template <class Context>
class CUDAFlashAttnOp : public Operator<Context> {
 public:
  CUDAFlashAttnOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        causal_(OP_SINGLE_ARG(int64_t, "causal", 0)),
        window_size_left_(OP_SINGLE_ARG(int64_t, "window_size_left", -1)),
        window_size_right_(OP_SINGLE_ARG(int64_t, "window_size_right", -1)),
        softmax_scale_(OP_SINGLE_ARG(float, "softmax_scale", 1.f)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {
    memset(&params_, 0, sizeof(params_));
    philox_state_ = at::PhiloxCudaState(ctx()->random_seed(), 0);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, bfloat16>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t causal_;
  int64_t window_size_left_, window_size_right_;
  float softmax_scale_, dropout_;
  Flash_fwd_params params_;
  at::PhiloxCudaState philox_state_;
};

template <class Context>
class CUDAFlashAttnGradientOp : public Operator<Context> {
 public:
  CUDAFlashAttnGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        causal_(OP_SINGLE_ARG(int64_t, "causal", 0)),
        window_size_left_(OP_SINGLE_ARG(int64_t, "window_size_left", -1)),
        window_size_right_(OP_SINGLE_ARG(int64_t, "window_size_right", -1)),
        softmax_scale_(OP_SINGLE_ARG(float, "softmax_scale", 1.f)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {
    memset(&params_, 0, sizeof(params_));
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, bfloat16>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t causal_;
  int64_t window_size_left_, window_size_right_;
  float softmax_scale_, dropout_;
  Flash_bwd_params params_;
};
#endif // USE_CUDA

#ifdef USE_MLU
template <class Context>
class CNNLFlashAttnOp : public Operator<Context> {
 public:
  CNNLFlashAttnOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        causal_(OP_SINGLE_ARG(int64_t, "causal", 0)),
        softmax_scale_(OP_SINGLE_ARG(float, "softmax_scale", 1.f)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, bfloat16>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t causal_;
  float softmax_scale_, dropout_;
  CNNLFlashAttnImpl impl_;
};

template <class Context>
class CNNLFlashAttnGradientOp : public Operator<Context> {
 public:
  CNNLFlashAttnGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        causal_(OP_SINGLE_ARG(int64_t, "causal", 0)),
        softmax_scale_(OP_SINGLE_ARG(float, "softmax_scale", 1.f)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float16, bfloat16>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t causal_;
  float softmax_scale_, dropout_;
  CNNLFlashAttnGradImpl impl_;
};
#endif

} // namespace dragon

#endif // DRAGON_OPERATORS_FLASH_ATTN_OP_H_
