#include <dragon/utils/math_functions.h>
#include "../../../kernels/flash_attn/cuda/static_switch.h"
#include "../../../kernels/rotary/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, bool SeqFirst>
__global__ void _Rotary(
    const int nthreads,
    const int L,
    const int H,
    const int D_halve,
    const bool conjugate,
    const T* xq,
    const T* xk,
    const T* cosw,
    const T* sinw,
    T* yq,
    T* yk) {
  const auto madd = math::FMAFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int d_halve = i % D_halve;
    const int h = SeqFirst ? (i / D_halve) % H : (i / D_halve / L) % H;
    const int l = SeqFirst ? (i / D_halve / H) % L : (i / D_halve) % L;
    const int n = i / D_halve / H / L;
    const int xi = SeqFirst ? (((n * L + l) * H + h) * D_halve + d_halve) * 2
                            : (((n * H + h) * L + l) * D_halve + d_halve) * 2;
    const int wi = l * D_halve + d_halve;
    const T q0 = xq[xi], q1 = xq[xi + 1];
    const T k0 = xk[xi], k1 = xk[xi + 1];
    const T cos = cosw[wi], sin = sinw[wi];
    yq[xi] = conjugate ? madd(q0, cos, q1 * sin) : madd(q0, cos, -q1 * sin);
    yk[xi] = conjugate ? madd(k0, cos, k1 * sin) : madd(k0, cos, -k1 * sin);
    yq[xi + 1] = conjugate ? madd(q1, cos, -q0 * sin) : madd(q1, cos, q0 * sin);
    yk[xi + 1] = conjugate ? madd(k1, cos, -k0 * sin) : madd(k1, cos, k0 * sin);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void Rotary<T, CUDAContext>(                                                \
      const int N,                                                            \
      const int L,                                                            \
      const int H,                                                            \
      const int D,                                                            \
      const bool conjugate,                                                   \
      const string& data_format,                                              \
      const T* xq,                                                            \
      const T* xk,                                                            \
      const T* cosw,                                                          \
      const T* sinw,                                                          \
      T* yq,                                                                  \
      T* yk,                                                                  \
      CUDAContext* ctx) {                                                     \
    const auto nthreads = N * L * H * D / 2;                                  \
    BOOL_SWITCH((data_format == "NLHD" ? true : false), SeqFirst, [&] {       \
      auto kernel = &_Rotary<math::Traits<T>::scalar_type, SeqFirst>;         \
      kernel<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          nthreads,                                                           \
          L,                                                                  \
          H,                                                                  \
          D / 2,                                                              \
          conjugate,                                                          \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(xq),          \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(xk),          \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(cosw),        \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(sinw),        \
          reinterpret_cast<math::Traits<T>::scalar_type*>(yq),                \
          reinterpret_cast<math::Traits<T>::scalar_type*>(yk));               \
    });                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
