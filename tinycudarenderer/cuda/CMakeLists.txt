# tinycudarenderer / cuda — CUDA port of the ssloy tinyrenderer.
#
# Builds two artifacts:
#   1. tinycudarenderer_cuda      (executable; analog of ../tinyrenderer)
#   2. tinycudarenderer_cuda_lib  (static library; reused by ../vulkan/)
#
# Splitting the renderer into a static library is the seam the Vulkan
# driver in ../vulkan/ needs: tinyvk_driver.cpp links against it and
# routes Vulkan-shaped commands into the CUDA kernels here.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Host-only sources (compiled by the C++ compiler, double-precision geometry.h).
set(_TC_CUDA_HOST_SOURCES
    tgaimage.cpp
    model.cpp
    host_loader.cpp
)

# CUDA sources (compiled by nvcc, float-precision geometry.cuh).
set(_TC_CUDA_DEVICE_SOURCES
    render.cu
    cuda_renderer.cu
)

# --- Static library: every translation unit minus the executable's main.cu ---
add_library(tinycudarenderer_cuda_lib STATIC
    ${_TC_CUDA_HOST_SOURCES}
    ${_TC_CUDA_DEVICE_SOURCES}
)
target_include_directories(tinycudarenderer_cuda_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
set_source_files_properties(${_TC_CUDA_HOST_SOURCES}
    PROPERTIES LANGUAGE CXX)
set_source_files_properties(${_TC_CUDA_DEVICE_SOURCES}
    PROPERTIES LANGUAGE CUDA)
set_target_properties(tinycudarenderer_cuda_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
    target_compile_options(tinycudarenderer_cuda_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif()

# --- Executable wrapping the library ---
add_executable(tinycudarenderer_cuda main.cu)
set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
target_link_libraries(tinycudarenderer_cuda PRIVATE tinycudarenderer_cuda_lib)
set_target_properties(tinycudarenderer_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tinycudarenderer/cuda"
)
