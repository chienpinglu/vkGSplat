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
vkgsplat_add_test(test_sensor_model)
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
    add_executable(test_cuda_gaussian_reconstruction test_cuda_gaussian_reconstruction.cu)
    target_link_libraries(test_cuda_gaussian_reconstruction PRIVATE vkgsplat CUDA::cudart)
    set_target_properties(test_cuda_gaussian_reconstruction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME test_cuda_gaussian_reconstruction COMMAND test_cuda_gaussian_reconstruction)
    set_tests_properties(test_cuda_gaussian_reconstruction PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_cuda_tile_renderer test_cuda_tile_renderer.cu)
    target_link_libraries(test_cuda_tile_renderer PRIVATE vkgsplat CUDA::cudart)
    set_target_properties(test_cuda_tile_renderer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME test_cuda_tile_renderer COMMAND test_cuda_tile_renderer)
    set_tests_properties(test_cuda_tile_renderer PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_cuda_rasterizer_smoke test_cuda_rasterizer_smoke.cpp)
    target_link_libraries(test_cuda_rasterizer_smoke PRIVATE vkgsplat CUDA::cudart)
    set_target_properties(test_cuda_rasterizer_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME test_cuda_rasterizer_smoke COMMAND test_cuda_rasterizer_smoke)
    set_tests_properties(test_cuda_rasterizer_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Honest Phase-6 interop gate: requires both CUDA and Vulkan, exercises the
# ExternalImage zero-copy seam end-to-end (CUDA kernel writes, Vulkan reads,
# pixel-exact diff).
if(VKGSPLAT_ENABLE_CUDA AND VKGSPLAT_ENABLE_VULKAN AND Vulkan_FOUND)
    add_executable(test_cuda_vulkan_interop test_cuda_vulkan_interop.cu)
    target_link_libraries(test_cuda_vulkan_interop PRIVATE
        vkgsplat
        CUDA::cudart
        Vulkan::Vulkan)
    set_target_properties(test_cuda_vulkan_interop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME test_cuda_vulkan_interop COMMAND test_cuda_vulkan_interop)
    set_tests_properties(test_cuda_vulkan_interop PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Cornell-box CPU vs CUDA reference-image gate: renders a non-trivial
# Gaussian scene through both backends and asserts PSNR. Catches silent
# regressions in either the CPU reference or the CUDA tile/scan pipeline
# that the existing 2-Gaussian smoke tests miss.
if(VKGSPLAT_ENABLE_CUDA AND VKGSPLAT_ENABLE_3DGS)
    add_executable(test_cornell_cpu_vs_cuda test_cornell_cpu_vs_cuda.cpp)
    target_link_libraries(test_cornell_cpu_vs_cuda PRIVATE vkgsplat CUDA::cudart)
    set_target_properties(test_cornell_cpu_vs_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME test_cornell_cpu_vs_cuda COMMAND test_cornell_cpu_vs_cuda)
    set_tests_properties(test_cornell_cpu_vs_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(VKGSPLAT_ENABLE_VK_RT_CAPTURE)
    if(WIN32)
        find_program(VKGSPLAT_POWERSHELL_VKRT NAMES pwsh powershell)
        if(NOT VKGSPLAT_POWERSHELL_VKRT)
            message(FATAL_ERROR "VKGSPLAT_ENABLE_VK_RT_CAPTURE=ON on Windows requires pwsh or powershell")
        endif()
        add_test(
            NAME test_vk_rt_smoke
            COMMAND ${VKGSPLAT_POWERSHELL_VKRT}
                -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_SOURCE_DIR}/scripts/run_vk_rt_smoke.ps1
                -AllowSkip
                -BuildDir ${CMAKE_BINARY_DIR}
                -Config $<CONFIG>
        )
        set_tests_properties(test_vk_rt_smoke PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
            DEPENDS vk_rt_capture
        )
    endif()
endif()

if(VKGSPLAT_ENABLE_WICKED_NVIDIA_TESTS)
    if(WIN32)
        find_program(VKGSPLAT_POWERSHELL NAMES pwsh powershell)
        if(NOT VKGSPLAT_POWERSHELL)
            message(FATAL_ERROR "VKGSPLAT_ENABLE_WICKED_NVIDIA_TESTS=ON on Windows requires pwsh or powershell")
        endif()
        set(VKGSPLAT_WICKED_NVIDIA_COMMAND
            ${VKGSPLAT_POWERSHELL}
            -NoProfile
            -ExecutionPolicy
            Bypass
            -File
            ${CMAKE_SOURCE_DIR}/scripts/run_wicked_nvidia_smoke.ps1
            -AllowSkip
        )
    else()
        set(VKGSPLAT_WICKED_NVIDIA_COMMAND
            ${CMAKE_SOURCE_DIR}/scripts/run_wicked_nvidia_smoke.sh
            --allow-skip
        )
    endif()
    add_test(
        NAME test_wicked_nvidia_vulkan_smoke
        COMMAND ${VKGSPLAT_WICKED_NVIDIA_COMMAND}
    )
    set_tests_properties(test_wicked_nvidia_vulkan_smoke PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 900
    )
endif()
