#include "../../../kernels/flash_attn/cuda/flash_bwd_launch_template.h"
#include "../../../kernels/flash_attn/cuda/flash_fwd_launch_template.h"
#include "../../../kernels/flash_attn/op_kernels.h"

namespace dragon {

namespace kernels {

void FlashAttn(Flash_fwd_params& params, CUDAContext* ctx) {
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.d, [&] {
      auto stream = ctx->cuda_stream();
      if (params.num_splits <= 1 && params.knew_ptr == nullptr) {
        run_mha_fwd_<elem_type, kHeadDim>(params, stream);
      } else {
        run_mha_fwd_splitkv_<elem_type, kHeadDim>(params, stream);
      }
    });
  });
}

void FlashAttnGrad(Flash_bwd_params& params, CUDAContext* ctx) {
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(params.d, [&] {
      run_mha_bwd_<elem_type, kHeadDim>(params, ctx->cuda_stream(), false);
    });
  });
}

} // namespace kernels

} // namespace dragon
