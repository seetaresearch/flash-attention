#include <dragon/utils/device/common_cub.h>
#include <dragon/utils/math_functions.h>
#include "../../../kernels/rms_norm/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT, int kBlockSize>
__global__ void _RMSNorm(
    const int N,
    const int C,
    const AccT epsilon,
    const T* x,
    const T* gamma,
    T* y) {
  using BlockReduce = cub::BlockReduce<AccT, kBlockSize>;
  __shared__ AccT block_rrms;
  __shared__ typename BlockReduce::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, N) { // clang-format off
    AccT sumsqr = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      sumsqr += math::utils::Sqr(math::utils::LDGC<AccT>(x + i * C + j));
    }
    sumsqr = BlockReduce(storage).Sum(sumsqr);
    if (threadIdx.x == 0) block_rrms = rsqrt(sumsqr / AccT(C) + epsilon);
    __syncthreads();
    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int index = i * C + j;
      y[index] = math::utils::LDGC<AccT>(x + index) * block_rrms *
                 math::utils::LDGC<AccT>(gamma + j);
    } // clang-format on
  }
}

} // namespace

#define DISPATCH_KERNEL(Func, T, AccT, kBlocks, kThreads, ...)                \
  if (kThreads == 256) {                                                      \
    Func<T, AccT, 256><<<kBlocks, 256, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else {                                                                    \
    Func<T, AccT, 512><<<kBlocks, 512, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  }

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                               \
  template <>                                                         \
  void RMSNorm<T, CUDAContext>(                                       \
      const int N,                                                    \
      const int C,                                                    \
      const float epsilon,                                            \
      const T* x,                                                     \
      const T* gamma,                                                 \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    const auto num_blocks = N, num_threads = C > 3072 ? 512 : 256;    \
    DISPATCH_KERNEL(                                                  \
        _RMSNorm,                                                     \
        math::Traits<T>::scalar_type,                                 \
        AccT,                                                         \
        num_blocks,                                                   \
        num_threads,                                                  \
        N,                                                            \
        C,                                                            \
        AccT(epsilon),                                                \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),     \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(gamma), \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));          \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_KERNEL

} // namespace kernels

} // namespace dragon
