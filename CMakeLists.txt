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
endif()

if(VKGSPLAT_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    if(VKGSPLAT_ENABLE_3DGS)
        target_sources(vkgsplat PRIVATE
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
        CUDA_ARCHITECTURES "75;80;86;89;90"
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

# --- Apps ------------------------------------------------------------------
add_subdirectory(apps/viewer)

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
message(STATUS "  Build  : ${CMAKE_BUILD_TYPE}")
