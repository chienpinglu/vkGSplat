cmake_minimum_required(VERSION 3.24)

project(vkGSplat
    VERSION 0.0.1
    DESCRIPTION "Hybrid compute + Vulkan renderer for synthetic data"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Options ---------------------------------------------------------------
option(VKGSPLAT_ENABLE_CUDA
    "Build the experimental CUDA rasterizer and CUDA/Vulkan interop bridge"
    OFF)
option(VKGSPLAT_ENABLE_VULKAN
    "Build the Vulkan frontend and viewer when the Vulkan SDK is available"
    ON)
option(VKGSPLAT_ENABLE_3DGS
    "Build the paused 3DGS renderer, viewer, and 3DGS-specific tests"
    OFF)
option(VKGSPLAT_ENABLE_TORCH
    "Build the optional PyTorch tensor extension for CUDA reconstruction kernels"
    OFF)
option(VKGSPLAT_ENABLE_WICKED_NVIDIA_TESTS
    "Register optional Wicked Engine smoke tests that require a real NVIDIA Vulkan GPU"
    OFF)
option(VKGSPLAT_ENABLE_VK_RT_CAPTURE
    "Build the vk_rt_capture Phase-4 substitute probe (requires Vulkan)"
    OFF)
set(VKGSPLAT_CUDA_ARCHITECTURES ""
    CACHE STRING "CUDA architectures for vkGSplat CUDA targets; empty selects a default list")
if(APPLE)
    set(VKGSPLAT_ENABLE_METAL_DEFAULT ON)
else()
    set(VKGSPLAT_ENABLE_METAL_DEFAULT OFF)
endif()
option(VKGSPLAT_ENABLE_METAL
    "Build the native Apple Metal compute backend"
    ${VKGSPLAT_ENABLE_METAL_DEFAULT})

# --- Dependencies ----------------------------------------------------------
if(VKGSPLAT_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND)
        message(STATUS "Vulkan SDK not found; building CPU-only targets")
    endif()
endif()

if(VKGSPLAT_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "VKGSPLAT_ENABLE_METAL requires Apple platforms")
    endif()
    enable_language(OBJCXX)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

# --- Library ---------------------------------------------------------------
add_library(vkgsplat STATIC
    # core
    src/core/scene.cpp
    src/core/camera.cpp
    src/core/denoise.cpp
    src/core/raytrace_seed.cpp
    src/core/reprojection.cpp
    src/core/sensor_model.cpp
    src/core/tile_raster.cpp
    src/spirv/module.cpp
    src/spirv/translator.cpp
    src/renderer.cpp
)

target_include_directories(vkgsplat
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(vkgsplat PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(VKGSPLAT_ENABLE_3DGS)
    target_sources(vkgsplat PRIVATE
        src/core/cpu_reference_renderer.cpp
    )
    target_compile_definitions(vkgsplat PUBLIC VKGSPLAT_ENABLE_3DGS=1)
endif()

if(VKGSPLAT_ENABLE_METAL)
    target_sources(vkgsplat PRIVATE
        src/metal/denoise.mm
    )
    target_link_libraries(vkgsplat PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    target_compile_definitions(vkgsplat PUBLIC VKGSPLAT_ENABLE_METAL=1)
    set_source_files_properties(src/metal/denoise.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc")
endif()

if(VKGSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
    target_sources(vkgsplat PRIVATE
        src/vulkan/instance.cpp
        src/vulkan/device.cpp
        src/vulkan/swapchain.cpp
        src/vulkan/mesh_shader_3dgs.cpp
    )
    target_link_libraries(vkgsplat PUBLIC Vulkan::Vulkan)
    target_compile_definitions(vkgsplat PUBLIC VKGSPLAT_ENABLE_VULKAN=1)
    if(WIN32)
        target_compile_definitions(vkgsplat PUBLIC
            VK_USE_PLATFORM_WIN32_KHR=1
            NOMINMAX=1)
    endif()
endif()

if(VKGSPLAT_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if(NOT VKGSPLAT_CUDA_ARCHITECTURES)
        set(VKGSPLAT_CUDA_ARCHITECTURES 75 80 86 89 90)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND VKGSPLAT_CUDA_ARCHITECTURES 120)
        endif()
    endif()

    set(VKGSPLAT_CUDA_ARCHITECTURES "" CACHE STRING
        "CUDA architectures to build; defaults include Blackwell when supported by the toolkit")
    if(VKGSPLAT_CUDA_ARCHITECTURES STREQUAL "")
        set(VKGSPLAT_CUDA_ARCHITECTURES 75 80 86 89 90)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND VKGSPLAT_CUDA_ARCHITECTURES 120)
        endif()
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${VKGSPLAT_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures to build" FORCE)

    if(VKGSPLAT_ENABLE_3DGS)
        target_sources(vkgsplat PRIVATE
            src/cuda/gaussian_reconstruction.cu
            src/cuda/rasterizer.cu
            src/cuda/tile_renderer.cu
        )
    endif()
    target_link_libraries(vkgsplat
        PUBLIC
            CUDA::cudart
            CUDA::cuda_driver
    )
    set_target_properties(vkgsplat PROPERTIES
        CUDA_ARCHITECTURES "${VKGSPLAT_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(vkgsplat PUBLIC VKGSPLAT_ENABLE_CUDA=1)

    if(VKGSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
        target_sources(vkgsplat PRIVATE
            src/interop/external_memory.cpp
            src/interop/timeline_semaphore.cpp
        )
        target_compile_definitions(vkgsplat PUBLIC VKGSPLAT_ENABLE_CUDA_VULKAN_INTEROP=1)
    endif()
endif()

if(VKGSPLAT_ENABLE_TORCH)
    if(NOT VKGSPLAT_ENABLE_CUDA OR NOT VKGSPLAT_ENABLE_3DGS)
        message(FATAL_ERROR "VKGSPLAT_ENABLE_TORCH requires VKGSPLAT_ENABLE_CUDA=ON and VKGSPLAT_ENABLE_3DGS=ON")
    endif()
    find_package(Torch REQUIRED)
    add_library(vkgsplat_torch MODULE
        src/torch/gaussian_reconstruction_torch.cpp
    )
    target_link_libraries(vkgsplat_torch PRIVATE
        vkgsplat
        ${TORCH_LIBRARIES}
        CUDA::cudart
    )
    target_include_directories(vkgsplat_torch PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_options(vkgsplat_torch PRIVATE ${TORCH_CXX_FLAGS})
    set_target_properties(vkgsplat_torch PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_vkgsplat_cuda")
endif()

# --- Apps ------------------------------------------------------------------
add_subdirectory(apps/viewer)
add_subdirectory(apps/vk_rt_capture)

# --- Tests -----------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- tinycudarenderer (Vulkan-on-CUDA proof of concept) --------------------
option(VKGSPLAT_BUILD_TINYCUDARENDERER
    "Build the tinycudarenderer sandbox (CPU + CUDA + minimal Vulkan-on-CUDA driver)"
    OFF)
if(VKGSPLAT_BUILD_TINYCUDARENDERER)
    add_subdirectory(tinycudarenderer)
endif()

# --- Status ----------------------------------------------------------------
message(STATUS "vkGSplat ${PROJECT_VERSION}")
message(STATUS "  3DGS   : ${VKGSPLAT_ENABLE_3DGS}")
message(STATUS "  Metal  : ${VKGSPLAT_ENABLE_METAL}")
if(VKGSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
    message(STATUS "  Vulkan : ${Vulkan_VERSION} (${Vulkan_LIBRARY})")
else()
    message(STATUS "  Vulkan : OFF")
endif()
message(STATUS "  CUDA   : ${VKGSPLAT_ENABLE_CUDA}")
if(VKGSPLAT_ENABLE_CUDA)
    message(STATUS "  CUDA arch: ${VKGSPLAT_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build  : ${CMAKE_BUILD_TYPE}")
