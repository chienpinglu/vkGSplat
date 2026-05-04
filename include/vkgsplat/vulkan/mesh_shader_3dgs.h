// SPDX-License-Identifier: Apache-2.0
//
// Minimal Vulkan mesh-shader 3DGS prototype hooks. The full renderer
// will grow here; for M3 this captures the required feature/extension
// contract and keeps mesh shaders as the only primitive-amplification
// path for splats.
#pragma once

#include <vulkan/vulkan.h>

namespace vkgsplat::vk {

struct MeshShader3dgsRequirements {
    bool has_mesh_shader_extension = false;
    bool has_mesh_shader_feature = false;
    bool has_buffer_device_address = false;
    bool has_synchronization2 = false;

    [[nodiscard]] bool supported() const noexcept {
        return has_mesh_shader_extension && has_mesh_shader_feature &&
               has_buffer_device_address && has_synchronization2;
    }
};

[[nodiscard]] MeshShader3dgsRequirements query_mesh_shader_3dgs_requirements(
    VkPhysicalDevice physical);

[[nodiscard]] bool physical_device_supports_mesh_shader_3dgs(VkPhysicalDevice physical);

} // namespace vkgsplat::vk
