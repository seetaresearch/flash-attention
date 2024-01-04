# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Licensed under the BSD 2-Clause License.

# - Find the Dragon libraries
#
# Following variables can be set and are optional:
#
#  DRAGON_FOUND              - was dragon found
#  DRAGON_VERSION            - the version of dragon found as a string
#  DRAGON_INCLUDE_DIR        - path to the dragon include files

unset(DRAGON_VERSION)
unset(DRAGON_INCLUDE_DIR)

if (USE_PYTHON_LIBS)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
      "import dragon; print(dragon.__version__); print(dragon.sysconfig.get_include());"
      RESULT_VARIABLE __result
      OUTPUT_VARIABLE __output
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (__result MATCHES 0)
    string(REGEX REPLACE ";" "\\\\;" __values ${__output})
    string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
    list(GET __values 0 DRAGON_VERSION)
    list(GET __values 1 DRAGON_INCLUDE_DIR)
    string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __ver_check "${DRAGON_VERSION}")
    if("${__ver_check}" STREQUAL "")
     unset(DRAGON_VERSION)
     unset(DRAGON_INCLUDE_DIR)
     message(STATUS "Requested Dragon version and include path, but got instead:\n${__output}\n")
  else()
     list(APPEND THIRD_PARTY_LIBRARY_DIRS ${DRAGON_INCLUDE_DIR}/../lib)
   endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dragon REQUIRED_VARS DRAGON_INCLUDE_DIR DRAGON_VERSION
                                         VERSION_VAR DRAGON_VERSION)

if(DRAGON_FOUND)
  message(STATUS "Dragon ver. ${DRAGON_VERSION} found (include: ${DRAGON_INCLUDE_DIR})")
endif()
