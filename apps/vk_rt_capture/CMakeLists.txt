# vk_rt_capture: Phase-4 alternative probe.
# Builds only when both VKGSPLAT_ENABLE_VULKAN and VKGSPLAT_ENABLE_VK_RT_CAPTURE are ON.

if(NOT VKGSPLAT_ENABLE_VULKAN)
    return()
endif()
if(NOT VKGSPLAT_ENABLE_VK_RT_CAPTURE)
    return()
endif()

add_executable(vk_rt_capture vk_rt_capture.cpp)
target_link_libraries(vk_rt_capture PRIVATE Vulkan::Vulkan)
target_compile_features(vk_rt_capture PRIVATE cxx_std_20)
set_target_properties(vk_rt_capture PROPERTIES OUTPUT_NAME vk_rt_capture)
