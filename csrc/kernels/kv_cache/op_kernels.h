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

#ifndef DRAGON_KERNELS_KV_CACHE_OP_KERNELS_H_
#define DRAGON_KERNELS_KV_CACHE_OP_KERNELS_H_

#include <dragon/core/context_cuda.h>

namespace dragon {

namespace kernels {

template <typename T, class Context>
void KVCache(
    const int N,
    const int L,
    const int H,
    const int D,
    const int seqlen,
    const int max_seqlen,
    const string& data_format,
    const T* xk,
    const T* xv,
    T* cache_k,
    T* cache_v,
    T* yk,
    T* yv,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_KV_CACHE_OP_KERNELS_H_
