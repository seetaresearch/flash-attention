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

#ifndef DRAGON_KERNELS_SWIGLU_OP_KERNELS_H_
#define DRAGON_KERNELS_SWIGLU_OP_KERNELS_H_

#include <dragon/core/context_cuda.h>

namespace dragon {

namespace kernels {

template <typename T, class Context>
void SwiGLU(const int N, const int C, const T* x, T* y, Context* ctx);

template <typename T, class Context>
void SwiGLUGrad(
    const int N,
    const int C,
    const T* dy,
    const T* x,
    T* y,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_SWIGLU_OP_KERNELS_H_
