# tinycudarenderer — three views of the same render job:
#
#   tinycudarenderer            (root)   : CPU baseline, ssloy tinyrenderer.
#   tinycudarenderer_cuda       (cuda/)  : direct CUDA rasterizer.
#   tinycudarenderer_vulkan     (vulkan/): Vulkan-shaped API on a CUDA driver.
#
# All three render the same scene to framebuffer.tga so the outputs can
# be byte- (or pixel-) compared.

cmake_minimum_required(VERSION 3.18)

# When this directory is built standalone (not via the parent vkGSplat
# tree) we still want a project() call so language detection runs.
if(NOT PROJECT_NAME)
    project(tinycudarenderer LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
endif()

# ---- CPU baseline ----------------------------------------------------------
add_executable(tinycudarenderer
    main.cpp
    our_gl.cpp
    model.cpp
    tgaimage.cpp
)

set_target_properties(tinycudarenderer PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tinycudarenderer"
)

find_package(OpenMP COMPONENTS CXX QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tinycudarenderer PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
    target_compile_options(tinycudarenderer PRIVATE -Wall)
endif()

# ---- CUDA + Vulkan ---------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(cuda)
    add_subdirectory(vulkan)
else()
    message(STATUS
        "[tinycudarenderer] CUDA compiler not found; skipping cuda/ and vulkan/")
endif()
