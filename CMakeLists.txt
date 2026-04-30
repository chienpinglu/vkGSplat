cmake_minimum_required(VERSION 3.24)

project(vkSplat
    VERSION 0.0.1
    DESCRIPTION "Hybrid CUDA + Vulkan renderer for 3D Gaussian Splatting"
    LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Dependencies ----------------------------------------------------------
find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- Library ---------------------------------------------------------------
add_library(vksplat STATIC
    # core
    src/core/scene.cpp
    src/core/camera.cpp
    # vulkan frontend
    src/vulkan/instance.cpp
    src/vulkan/device.cpp
    src/vulkan/swapchain.cpp
    # cuda backend (kernels added later)
    src/cuda/rasterizer.cu
    # interop
    src/interop/external_memory.cpp
    src/interop/timeline_semaphore.cpp
)

target_include_directories(vksplat
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vksplat
    PUBLIC
        Vulkan::Vulkan
        CUDA::cudart
        CUDA::cuda_driver
)

set_target_properties(vksplat PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89;90"
    POSITION_INDEPENDENT_CODE ON
)

# --- Apps ------------------------------------------------------------------
add_subdirectory(apps/viewer)

# --- Tests -----------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- Status ----------------------------------------------------------------
message(STATUS "vkSplat ${PROJECT_VERSION}")
message(STATUS "  Vulkan : ${Vulkan_VERSION} (${Vulkan_LIBRARY})")
message(STATUS "  CUDA   : ${CUDAToolkit_VERSION}")
message(STATUS "  Build  : ${CMAKE_BUILD_TYPE}")
