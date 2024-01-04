#include <dragon/utils/math_functions.h>
#include "../../../kernels/flash_attn/cuda/static_switch.h"
#include "../../../kernels/kv_cache/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, bool SeqFirst>
__global__ void _KVCache(
    const int nthreads,
    const int L,
    const int H,
    const int D,
    const int seqlen,
    const int max_seqlen,
    const T* xk,
    const T* xv,
    T* cache_k,
    T* cache_v,
    T* yk,
    T* yv) {
  const int x_seqlen = L - seqlen;
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int d = yi % D;
    const int h = SeqFirst ? (yi / D) % H : (yi / D / L) % H;
    const int l = SeqFirst ? (yi / D / H) % L : (yi / D) % L;
    const int n = yi / D / H / L;
    const int xi = SeqFirst ? ((n * x_seqlen + l - seqlen) * H + h) * D + d
                            : ((n * H + h) * x_seqlen + l - seqlen) * D + d;
    const int zi = SeqFirst ? ((n * max_seqlen + l) * H + h) * D + d
                            : ((n * H + h) * max_seqlen + l) * D + d;
    yk[yi] = l >= seqlen ? (cache_k[zi] = xk[xi]) : cache_k[zi];
    yv[yi] = l >= seqlen ? (cache_v[zi] = xv[xi]) : cache_v[zi];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void KVCache<T, CUDAContext>(                                               \
      const int N,                                                            \
      const int L,                                                            \
      const int H,                                                            \
      const int D,                                                            \
      const int seqlen,                                                       \
      const int max_seqlen,                                                   \
      const string& data_format,                                              \
      const T* xk,                                                            \
      const T* xv,                                                            \
      T* cache_k,                                                             \
      T* cache_v,                                                             \
      T* yk,                                                                  \
      T* yv,                                                                  \
      CUDAContext* ctx) {                                                     \
    const auto nthreads = N * L * H * D;                                      \
    BOOL_SWITCH((data_format == "NLHD" ? true : false), SeqFirst, [&] {       \
      auto kernel = &_KVCache<math::Traits<T>::scalar_type, SeqFirst>;        \
      kernel<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          nthreads,                                                           \
          L,                                                                  \
          H,                                                                  \
          D,                                                                  \
          seqlen,                                                             \
          max_seqlen,                                                         \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(xk),          \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(xv),          \
          reinterpret_cast<math::Traits<T>::scalar_type*>(cache_k),           \
          reinterpret_cast<math::Traits<T>::scalar_type*>(cache_v),           \
          reinterpret_cast<math::Traits<T>::scalar_type*>(yk),                \
          reinterpret_cast<math::Traits<T>::scalar_type*>(yv));               \
    });                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
