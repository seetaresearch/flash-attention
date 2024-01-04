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

#ifndef DRAGON_UTILS_ATEN_COMMON_CUDA_H_
#define DRAGON_UTILS_ATEN_COMMON_CUDA_H_

#include <dragon/utils/device/common_cuda.h>

#define C10_CUDA_CHECK(condition)                                      \
  do {                                                                 \
    using namespace dragon;                                            \
    cudaError_t error = condition;                                     \
    CHECK_EQ(error, cudaSuccess) << "\n" << cudaGetErrorString(error); \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace at {

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(
      int64_t* seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

namespace cuda {

namespace {

inline cudaDeviceProp* getCurrentDeviceProperties() {
  int device = dragon::CUDAGetDevice();
  auto& prop = dragon::CUDAGetDeviceProp(device);
  return const_cast<cudaDeviceProp*>(&prop);
}

} // namespace

namespace philox {

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/detail/UnpackRaw.cuh
__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(
    at::PhiloxCudaState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to
    // "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire
    // kernel. For most threads' reads it will hit in cache, so it shouldn't
    // hurt performance.
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
  }
}

} // namespace philox

} // namespace cuda

} // namespace at

#endif // DRAGON_UTILS_ATEN_COMMON_CUDA_H_
