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

#ifndef DRAGON_OPERATORS_ROTARY_OP_H_
#define DRAGON_OPERATORS_ROTARY_OP_H_

#include <dragon/core/operator.h>
#include <dragon/operators/array/transpose_op_impl_cnnl.h>

namespace dragon {

template <class Context>
class RotaryOp : public Operator<Context> {
 public:
  RotaryOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
#ifdef USE_MLU
  CNNLTransposeOpImpl trans_impl_;
#endif
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ROTARY_OP_H_
