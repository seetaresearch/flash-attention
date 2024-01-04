# ---[ Sources
add_subdirectory(flash_attn)
add_subdirectory(kv_cache)
add_subdirectory(rms_norm)
add_subdirectory(rotary)
add_subdirectory(swiglu)

# ---[ CUDA Sources
if (USE_CUDA)
  set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} ${KERNEL_CUDA_SOURCES})
endif()

# ---[ MLU Sources
if (USE_MLU)
  set(MODULE_MLU_SOURCES ${MODULE_MLU_SOURCES} ${KERNEL_MLU_SOURCES})
endif()

# Submit to the parent scope.
set(MODULE_SOURCES ${MODULE_SOURCES} PARENT_SCOPE)
set(MODULE_CUDA_SOURCES ${MODULE_CUDA_SOURCES} PARENT_SCOPE)
set(MODULE_MLU_SOURCES ${MODULE_MLU_SOURCES} PARENT_SCOPE)
