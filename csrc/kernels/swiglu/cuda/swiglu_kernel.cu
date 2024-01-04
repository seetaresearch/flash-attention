#include <dragon/utils/math_functions.h>
#include "../../../kernels/swiglu/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_SwiGLU(const int nthreads, const int C_halve, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int xi = yi / C_halve * C_halve * 2 + yi % C_halve;
    const AccT gate = AccT(x[xi + C_halve]);
    y[yi] = gate * AccT(x[xi]) / (AccT(1) + exp(-gate));
  }
}

template <typename T, typename AccT>
__global__ void _SwiGLUGrad(
    const int nthreads,
    const int C_halve,
    const T* dy,
    const T* x,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int xi = yi / C_halve * C_halve * 2 + yi % C_halve;
    const AccT gate = AccT(x[xi + C_halve]);
    const AccT s = AccT(1) / (AccT(1) + expf(-gate));
    const AccT g = AccT(dy[yi]);
    dx[xi] = gate * s * g;
    dx[xi + C_halve] = s * (AccT(1) + gate * (AccT(1) - s)) * g * AccT(x[xi]);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void SwiGLU<T, CUDAContext>(                                               \
      const int N, const int C, const T* x, T* y, CUDAContext* ctx) {        \
    const auto nthreads = N * C / 2, C_halve = C / 2;                        \
    _SwiGLU<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type> \
        <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
            nthreads,                                                        \
            C_halve,                                                         \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
            reinterpret_cast<math::Traits<T>::scalar_type*>(y));             \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                    \
  template <>                                                             \
  void SwiGLUGrad<T, CUDAContext>(                                        \
      const int N,                                                        \
      const int C,                                                        \
      const T* dy,                                                        \
      const T* x,                                                         \
      T* dx,                                                              \
      CUDAContext* ctx) {                                                 \
    const auto nthreads = N * C / 2, C_halve = C / 2;                     \
    _SwiGLUGrad<                                                          \
        math::Traits<T>::scalar_type,                                     \
        math::Traits<T>::accumulator_type>                                \
        <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            nthreads,                                                     \
            C_halve,                                                      \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),    \
            reinterpret_cast<const math::Traits<T>::scalar_type*>(x),     \
            reinterpret_cast<math::Traits<T>::scalar_type*>(dx));         \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
