# ---[ Messages
message(STATUS "Build module: ${CMAKE_CURRENT_LIST_DIR}")

# ---[ Sources
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc/kernels kernels)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc/operators operators)

# ---[ Targets
if (USE_CUDA)
  cuda_add_library(dragon_flashattn ${MODULE_SOURCES} ${MODULE_CUDA_SOURCES})
elseif (USE_MLU)
  bang_add_library(dragon_flashattn ${MODULE_SOURCES} ${MODULE_MLU_SOURCES})
else()
  add_library(dragon_flashattn ${MODULE_SOURCES})
endif()

# ---[ Target Defines
if (MSVC)
  target_compile_definitions(dragon_flashattn PRIVATE "DRAGON_API=__declspec(dllimport)")
  target_compile_definitions(dragon_flashattn PRIVATE "M_LOG2E=1.44269504088896340736")
else()
  target_compile_definitions(dragon_flashattn PRIVATE "DRAGON_API=")
endif()

# ---[ Target Properties
set(_install_rpath ${RPATH_PORTABLE_ORIGIN})
if (USE_SHARED_LIBS)
  set(_install_rpath ${_install_rpath} ${THIRD_PARTY_LIBRARY_DIRS})
endif()
set_target_properties(dragon_flashattn PROPERTIES INSTALL_RPATH "${_install_rpath}")

# ---[ Linker
target_link_libraries_v2(dragon_flashattn dragon)
if (USE_CUDA)
  if (USE_SHARED_CUDA_LIBS)
    target_link_libraries(dragon_flashattn ${CUDA_LIBRARIES_SHARED})
  else()
    target_link_libraries(dragon_flashattn ${CUDA_LIBRARIES_STATIC})
  endif()
endif()
if (USE_MLU)
  target_link_libraries(dragon_flashattn ${MLU_LIBRARIES})
endif()
if (WIN32)
  target_link_libraries_v2(dragon_flashattn protobuf)
endif()

# ---[ Command - Strip
strip_debug_symbol(dragon_flashattn)

# ---[ Install - Libraries
set(_install_dir ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS dragon_flashattn DESTINATION ${_install_dir}/${LIBRARY_INSTALL_PREFIX})
