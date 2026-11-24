cmake_minimum_required(VERSION 3.16)
project(mqmas_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_library(mqmas_core SHARED cpp/mqmas_core.cpp)

# Compiler-specific optimisation flags.  The original GNU-style flags are not
# valid MSVC options.
if(MSVC)
  target_compile_options(mqmas_core PRIVATE /O2 /fp:fast)
else()
  target_compile_options(mqmas_core PRIVATE -O3 -march=native -ffast-math)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(mqmas_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# A Windows DLL needs exported C entry points.  This asks CMake to generate the
# .def file automatically, avoiding compiler-specific __declspec annotations.
set_target_properties(mqmas_core PROPERTIES
  OUTPUT_NAME mqmas_core
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Visual Studio and other multi-config generators normally append Release/ or
# Debug/.  Keep the runtime library directly in the requested build directory
# so the Python loader has the same layout on every platform.
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set_target_properties(mqmas_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}"
  )
endforeach()
