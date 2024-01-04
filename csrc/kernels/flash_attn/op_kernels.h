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

#ifndef DRAGON_KERNELS_FLASH_ATTN_OP_KERNELS_H_
#define DRAGON_KERNELS_FLASH_ATTN_OP_KERNELS_H_

#ifdef USE_CUDA
#include <dragon/core/context_cuda.h>
#include "../../kernels/flash_attn/cuda/flash.h"
#endif

namespace dragon {

namespace kernels {

#ifdef USE_CUDA
void FlashAttn(Flash_fwd_params& params, CUDAContext* ctx);
void FlashAttnGrad(Flash_bwd_params& params, CUDAContext* ctx);
#endif // USE_CUDA

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_FLASH_ATTN_OP_KERNELS_H_
