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

#ifndef DRAGON_OPERATORS_RMS_NORM_OP_H_
#define DRAGON_OPERATORS_RMS_NORM_OP_H_

#include <dragon/core/operator.h>

namespace dragon {

template <class Context>
class RMSNormOp final : public Operator<Context> {
 public:
  RMSNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-5)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  double epsilon_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_RMS_NORM_OP_H_
