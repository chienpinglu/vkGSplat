cmake_minimum_required(VERSION 3.24)

project(vkSplat
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
option(VKSPLAT_ENABLE_CUDA
    "Build the experimental CUDA rasterizer and CUDA/Vulkan interop bridge"
    OFF)
option(VKSPLAT_ENABLE_VULKAN
    "Build the Vulkan frontend and viewer when the Vulkan SDK is available"
    ON)
option(VKSPLAT_ENABLE_3DGS
    "Build the paused 3DGS renderer, viewer, and 3DGS-specific tests"
    OFF)
if(APPLE)
    set(VKSPLAT_ENABLE_METAL_DEFAULT ON)
else()
    set(VKSPLAT_ENABLE_METAL_DEFAULT OFF)
endif()
option(VKSPLAT_ENABLE_METAL
    "Build the native Apple Metal compute backend"
    ${VKSPLAT_ENABLE_METAL_DEFAULT})

# --- Dependencies ----------------------------------------------------------
if(VKSPLAT_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND)
        message(STATUS "Vulkan SDK not found; building CPU-only targets")
    endif()
endif()

if(VKSPLAT_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "VKSPLAT_ENABLE_METAL requires Apple platforms")
    endif()
    enable_language(OBJCXX)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

# --- Library ---------------------------------------------------------------
add_library(vksplat STATIC
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

target_include_directories(vksplat
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(vksplat PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(VKSPLAT_ENABLE_3DGS)
    target_sources(vksplat PRIVATE
        src/core/cpu_reference_renderer.cpp
    )
    target_compile_definitions(vksplat PUBLIC VKSPLAT_ENABLE_3DGS=1)
endif()

if(VKSPLAT_ENABLE_METAL)
    target_sources(vksplat PRIVATE
        src/metal/denoise.mm
    )
    target_link_libraries(vksplat PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    target_compile_definitions(vksplat PUBLIC VKSPLAT_ENABLE_METAL=1)
    set_source_files_properties(src/metal/denoise.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc")
endif()

if(VKSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
    target_sources(vksplat PRIVATE
        src/vulkan/instance.cpp
        src/vulkan/device.cpp
        src/vulkan/swapchain.cpp
    )
    if(VKSPLAT_ENABLE_3DGS)
        target_sources(vksplat PRIVATE
            src/vulkan/mesh_shader_3dgs.cpp
        )
    endif()
    target_link_libraries(vksplat PUBLIC Vulkan::Vulkan)
    target_compile_definitions(vksplat PUBLIC VKSPLAT_ENABLE_VULKAN=1)
endif()

if(VKSPLAT_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    if(VKSPLAT_ENABLE_3DGS)
        target_sources(vksplat PRIVATE
            src/cuda/rasterizer.cu
            src/cuda/tile_renderer.cu
        )
    endif()
    target_link_libraries(vksplat
        PUBLIC
            CUDA::cudart
            CUDA::cuda_driver
    )
    set_target_properties(vksplat PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;89;90"
    )
    target_compile_definitions(vksplat PUBLIC VKSPLAT_ENABLE_CUDA=1)

    if(VKSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
        target_sources(vksplat PRIVATE
            src/interop/external_memory.cpp
            src/interop/timeline_semaphore.cpp
        )
        target_compile_definitions(vksplat PUBLIC VKSPLAT_ENABLE_CUDA_VULKAN_INTEROP=1)
    endif()
endif()

# --- Apps ------------------------------------------------------------------
add_subdirectory(apps/viewer)

# --- Tests -----------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- tinycudarenderer (Vulkan-on-CUDA proof of concept) --------------------
option(VKSPLAT_BUILD_TINYCUDARENDERER
    "Build the tinycudarenderer sandbox (CPU + CUDA + minimal Vulkan-on-CUDA driver)"
    OFF)
if(VKSPLAT_BUILD_TINYCUDARENDERER)
    add_subdirectory(tinycudarenderer)
endif()

# --- Status ----------------------------------------------------------------
message(STATUS "vkSplat ${PROJECT_VERSION}")
message(STATUS "  3DGS   : ${VKSPLAT_ENABLE_3DGS}")
message(STATUS "  Metal  : ${VKSPLAT_ENABLE_METAL}")
if(VKSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
    message(STATUS "  Vulkan : ${Vulkan_VERSION} (${Vulkan_LIBRARY})")
else()
    message(STATUS "  Vulkan : OFF")
endif()
message(STATUS "  CUDA   : ${VKSPLAT_ENABLE_CUDA}")
message(STATUS "  Build  : ${CMAKE_BUILD_TYPE}")
