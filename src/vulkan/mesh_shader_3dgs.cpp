// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/vulkan/mesh_shader_3dgs.h"

#include <cstring>
#include <vector>

namespace vkgsplat::vk {

MeshShader3dgsRequirements query_mesh_shader_3dgs_requirements(VkPhysicalDevice physical) {
    std::uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, exts.data());

    MeshShader3dgsRequirements req{};
    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0) {
            req.has_mesh_shader_extension = true;
        }
        if (std::strcmp(ext.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
            req.has_buffer_device_address = true;
        }
        if (std::strcmp(ext.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
            req.has_synchronization2 = true;
        }
    }

    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{};
    mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &mesh;
    vkGetPhysicalDeviceFeatures2(physical, &features);

    req.has_mesh_shader_feature = mesh.meshShader == VK_TRUE;
    return req;
}

bool physical_device_supports_mesh_shader_3dgs(VkPhysicalDevice physical) {
    return query_mesh_shader_3dgs_requirements(physical).supported();
}

} // namespace vkgsplat::vk
