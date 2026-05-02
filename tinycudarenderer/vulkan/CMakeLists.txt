# tinycudarenderer / vulkan — minimal Vulkan-shaped host code wired to
# the CUDA backend. The driver translation unit links against
# tinycudarenderer_cuda_lib (defined in ../cuda/CMakeLists.txt).

cmake_minimum_required(VERSION 3.18)

# The application loader uses the host-only OBJ + texture readers from
# ../cuda/. They are pure C++ and do not pull in any CUDA headers.
add_executable(tinycudarenderer_vulkan
    main.cpp
    tinyvk_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/host_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/tgaimage.cpp
)

target_include_directories(tinycudarenderer_vulkan
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
)

target_link_libraries(tinycudarenderer_vulkan
    PRIVATE
        tinycudarenderer_cuda_lib
)

set_target_properties(tinycudarenderer_vulkan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tinycudarenderer/vulkan"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
    target_compile_options(tinycudarenderer_vulkan PRIVATE -Wall)
endif()
