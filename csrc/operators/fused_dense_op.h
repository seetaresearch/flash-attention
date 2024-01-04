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

#ifndef DRAGON_OPERATORS_FUSED_DENSE_OP_H_
#define DRAGON_OPERATORS_FUSED_DENSE_OP_H_

#include <dragon/core/operator.h>
#include <dragon/operators/distributed/collective_op_impl.h>
#include <dragon/operators/math/gemm_op_impl_cuda.h>

namespace dragon {

#ifdef USE_CUDA
template <class Context>
class CUDAFusedDenseOp : public Operator<Context> {
 public:
  CUDAFusedDenseOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        epilogue_(OP_SINGLE_ARG(string, "epilogue", "")),
        col_parallel_(OP_SINGLE_ARG(int64_t, "col_parallel", 0)),
        row_parallel_(OP_SINGLE_ARG(int64_t, "row_parallel", 0)),
        seq_parallel_(OP_SINGLE_ARG(int64_t, "seq_parallel", 0)) {
    coll_impl_.SetComm(
        OP_SINGLE_ARG(int64_t, "comm", 0),
        OP_SINGLE_ARG(int64_t, "group", 0),
        OP_REPEATED_ARG(int64_t, "ranks"));
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string epilogue_;
  int64_t col_parallel_, row_parallel_, seq_parallel_;
  CUDAGemmOpImpl<cublasLtMatmulAlgo_t> Y_impl_;
  NCCLCollectiveOpImpl coll_impl_;
};

template <class Context>
class CUDAFusedDenseGradientOp : public Operator<Context> {
 public:
  CUDAFusedDenseGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        epilogue_(OP_SINGLE_ARG(string, "epilogue", "")),
        col_parallel_(OP_SINGLE_ARG(int64_t, "col_parallel", 0)),
        row_parallel_(OP_SINGLE_ARG(int64_t, "row_parallel", 0)),
        seq_parallel_(OP_SINGLE_ARG(int64_t, "seq_parallel", 0)) {
    coll_impl_.SetComm(
        OP_SINGLE_ARG(int64_t, "comm", 0),
        OP_SINGLE_ARG(int64_t, "group", 0),
        OP_REPEATED_ARG(int64_t, "ranks"));
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string epilogue_;
  int64_t col_parallel_, row_parallel_, seq_parallel_;
  CUDAGemmOpImpl<cublasLtMatmulAlgo_t> dX_impl_, dW_impl_;
  NCCLCollectiveOpImpl coll_impl_;
};
#endif // USE_CUDA

} // namespace dragon

#endif // DRAGON_OPERATORS_FUSED_DENSE_OP_H_
