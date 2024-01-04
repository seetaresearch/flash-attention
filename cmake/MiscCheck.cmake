include(CheckCXXCompilerFlag)

# ---[ Check if CXX17 is supported
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---[ Use ``-fPIC`` for all compilers
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---[ Compiler flags
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_EXTENDED_ALIGNED_STORAGE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003 /wd4114 /wd4244 /wd4251 /wd4267 /wd4273 /wd4275 /wd4800 /wd4819 /wd4996")
  string(REPLACE "/W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
else()  # GNU, Clang, AppleClang
  set(CMAKE_ORIGIN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")
endif()
