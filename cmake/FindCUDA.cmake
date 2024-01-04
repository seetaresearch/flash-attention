# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the CUDA libraries
#
# Following variables can be set and are optional:
#
#  CUDA_VERSION          - version of the CUDA
#  CUDA_VERSION_MAJOR    - the major version number of CUDA
#  CUDA_VERSION_MINOR    - the minor version number of CUDA
#  CUDA_TOOLKIT_ROOT_DIR - path to the CUDA toolkit
#  CUDA_INCLUDE_DIR      - path to the CUDA headers
#  CUDA_LIBRARIES_SHARED - path to the CUDA shared library
#  CUDA_LIBRARIES_STATIC - path to the CUDA static library
#

find_package(CUDA REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/SelectCudaArch.cmake)

# Set NVCC flags.
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH ${CUDA_ARCH})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w -O3 -std=c++17")

# Set include directory.
set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})

# Set libraries.
if (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
elseif (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
  list(APPEND THIRD_PARTY_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
endif()
set(CUDA_LIBRARIES_SHARED cudart cublas cublasLt)
set(CUDA_LIBRARIES_STATIC culibos cudart_static cublas_static cublasLt_static)
