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

#ifndef DRAGON_KERNELS_RMS_NORM_OP_KERNELS_H_
#define DRAGON_KERNELS_RMS_NORM_OP_KERNELS_H_

#include <dragon/core/context_cuda.h>

namespace dragon {

namespace kernels {

template <typename T, class Context>
void RMSNorm(
    const int N,
    const int C,
    const float epsilon,
    const T* x,
    const T* gamma,
    T* y,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_RMS_NORM_OP_KERNELS_H_
