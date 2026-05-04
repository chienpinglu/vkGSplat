# vkGSplat unit tests.
#
# We avoid pulling in a heavyweight test framework in v0 — each test is
# a tiny standalone executable that returns nonzero on failure and is
# registered with CTest. Once the surface area justifies it we will
# switch to GoogleTest or doctest.

function(vkgsplat_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE vkgsplat)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

vkgsplat_add_test(test_camera)
vkgsplat_add_test(test_denoise)
vkgsplat_add_test(test_public_headers)
vkgsplat_add_test(test_raytrace_seed)
vkgsplat_add_test(test_reprojection)
vkgsplat_add_test(test_scene)
vkgsplat_add_test(test_spirv_translate)
vkgsplat_add_test(test_tile_raster)
vkgsplat_add_test(test_types)

if(VKGSPLAT_ENABLE_METAL)
    add_executable(test_metal_denoise test_metal_denoise.cpp)
    target_link_libraries(test_metal_denoise PRIVATE vkgsplat)
    add_test(NAME test_metal_denoise COMMAND test_metal_denoise)
    set_tests_properties(test_metal_denoise PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(VKGSPLAT_ENABLE_3DGS)
    vkgsplat_add_test(test_cpp_backend)
    vkgsplat_add_test(test_cpu_3dgs_render)
    vkgsplat_add_test(test_gpu_pipeline)
    vkgsplat_add_test(test_real_asset_path)
    vkgsplat_add_test(test_scene_io)
    vkgsplat_add_test(test_scene_ply)
endif()

if(VKGSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
    add_executable(test_vulkan_m7_offscreen test_vulkan_m7_offscreen.cpp)
    target_link_libraries(test_vulkan_m7_offscreen PRIVATE vkgsplat)
    add_test(NAME test_vulkan_m7_offscreen COMMAND test_vulkan_m7_offscreen)
    set_tests_properties(test_vulkan_m7_offscreen PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(VKGSPLAT_ENABLE_CUDA AND VKGSPLAT_ENABLE_3DGS)
    add_executable(test_cuda_tile_renderer test_cuda_tile_renderer.cu)
    target_link_libraries(test_cuda_tile_renderer PRIVATE vkgsplat CUDA::cudart)
    set_target_properties(test_cuda_tile_renderer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME test_cuda_tile_renderer COMMAND test_cuda_tile_renderer)
    set_tests_properties(test_cuda_tile_renderer PROPERTIES SKIP_RETURN_CODE 77)
endif()
