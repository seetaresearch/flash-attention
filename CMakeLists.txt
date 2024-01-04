cmake_minimum_required(VERSION 3.0.2)
cmake_policy(SET CMP0054 NEW)
project(flashattn)

# ---[ Build Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# ---[ Library Options
option(USE_CUDA "Use CUDA" ON)
option(USE_MLU "Use MLU" OFF)
option(USE_PYTHON_LIBS "Use python libraries" ON)
option(USE_SHARED_LIBS "Use shared libraries" ON)
option(USE_SHARED_CUDA_LIBS "Use shared cuda libraries" ON)

# ---[ Project Variables
# Set the directory of third party.
if (NOT THIRD_PARTY_DIR)
  set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)
endif()

# Set the CUDA target architectures.
# If not, common architectures (>= 8.0) will be used.
if (USE_CUDA AND NOT CUDA_ARCH)
  set(CUDA_ARCH Auto)
endif()

# Set the MLU target architectures.
# If not, common architectures (3.0) will be used.
if (USE_MLU AND NOT MLU_ARCH)
  set(MLU_ARCH --bang-arch=compute_30)
endif()

# Set the python interpreter if necessary.
# If not, a searched interpreter will be used.
# set(PYTHON_EXECUTABLE <executable>)

# ---[ CMake Modules
include(${PROJECT_SOURCE_DIR}/cmake/MiscCheck.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/LinkLibrary.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/StripDebugInfo.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Dependencies.cmake)

# ---[ CMake Variables
set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "" FORCE)
set(CMAKE_CONFIGURATION_TYPES Release CACHE INTERNAL "" FORCE)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../targets/native
      CACHE INTERNAL "" FORCE)
endif()
if (NOT LIBRARY_INSTALL_PREFIX)
  set(LIBRARY_INSTALL_PREFIX "")
endif()

# ---[ Subdirectories
add_subdirectory(csrc/modules/extension)
