#include "../operators/flash_attn_op.h"

namespace dragon {

OPERATOR_SCHEMA(FlashAttn).NumInputs(3, 6).NumOutputs(1);
OPERATOR_SCHEMA(FlashAttnGradient).NumInputs(5).NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(2), O(0), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(FlashAttn, GradientMaker);

} // namespace dragon
