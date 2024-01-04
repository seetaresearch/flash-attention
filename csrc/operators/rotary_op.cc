#include <dragon/core/workspace.h>

#include "../kernels/rotary/op_kernels.h"
#include "../operators/rotary_op.h"

namespace dragon {

template <class Context>
template <typename T>
void RotaryOp<Context>::DoRunWithType() {
  auto &XQ = Input(0), &XK = Input(1), &W_cos = Input(2), &W_sin = Input(3);
  auto *YQ = Output(0)->ReshapeLike(XQ), *YK = Output(1)->ReshapeLike(XK);

  T *xq = nullptr, *xk = nullptr, *yq = nullptr, *yk = nullptr;
  auto compute_format = data_format();

#ifdef USE_MLU
  if (data_format() == "NLHD" &&
      TypeMeta::Id<Context>() == TypeMeta::Id<MLUContext>()) {
    xq = ctx()->workspace()->template data<T, Context>(XQ.count() * 4);
    xk = xq + XQ.count(), yq = xq + XQ.count() * 2, yk = xq + XQ.count() * 3;
    compute_format = "NHLD";
    trans_impl_.Setup<T>(XQ.dims(), {0, 2, 1, 3}, ctx());
    auto* _ = ctx()->workspace()->template data<Context>(
        trans_impl_.scratch_size(), "BufferKernel");
    trans_impl_.Compute(XQ.template data<T, Context>(), xq, _, ctx());
    trans_impl_.Compute(XK.template data<T, Context>(), xk, _, ctx());
  }
#endif

  kernels::Rotary(
      XQ.dim(0), // N
      data_format() == "NLHD" ? XQ.dim(1) : XQ.dim(-2), // L
      data_format() == "NLHD" ? XQ.dim(-2) : XQ.dim(1), // H
      XQ.dim(-1), // D
      def().type() == "RotaryGradient", // conjugate
      compute_format,
      xq != nullptr ? xq : XQ.template data<T, Context>(),
      xk != nullptr ? xk : XK.template data<T, Context>(),
      W_cos.template data<T, Context>(),
      W_sin.template data<T, Context>(),
      yq != nullptr ? yq : YQ->template mutable_data<T, Context>(),
      yk != nullptr ? yk : YK->template mutable_data<T, Context>(),
      ctx());

#ifdef USE_MLU
  if (yq != nullptr) {
    auto Y_dims = XQ.dims();
    std::swap(Y_dims[1], Y_dims[2]);
    trans_impl_.Setup<T>(Y_dims, {0, 2, 1, 3}, ctx());
    auto* _ = ctx()->workspace()->template data<Context>(
        trans_impl_.scratch_size(), "BufferKernel");
    trans_impl_.Compute(yq, YQ->template mutable_data<T, Context>(), _, ctx());
    trans_impl_.Compute(yk, YK->template mutable_data<T, Context>(), _, ctx());
  }
#endif
}

#ifdef USE_CUDA
INSTANTIATE_OPERATOR(Rotary, CUDAContext);
REGISTER_CUDA_OPERATOR(Rotary, RotaryOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(RotaryGradient, RotaryOp<CUDAContext>);
#endif
#ifdef USE_MLU
INSTANTIATE_OPERATOR(Rotary, MLUContext);
REGISTER_MLU_OPERATOR(Rotary, RotaryOp<MLUContext>);
REGISTER_MLU_OPERATOR(RotaryGradient, RotaryOp<MLUContext>);
#endif

OPERATOR_SCHEMA(Rotary).NumInputs(4).NumOutputs(2).AllowInplace(
    {{0, 0}, {1, 1}});
OPERATOR_SCHEMA(RotaryGradient)
    .NumInputs(4)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}});

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({GO(0), GO(1), I(2), I(3)}),
        vector<string>({GI(0), GI(1)}));
  }
};

} // namespace

REGISTER_GRADIENT(Rotary, GradientMaker);

} // namespace dragon
