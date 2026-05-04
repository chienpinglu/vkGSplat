// SPDX-License-Identifier: Apache-2.0
//
// VkInstance wrapper. The v1 implementation uses the system Vulkan
// loader via the standard ICD discovery mechanism — the position paper
// (Section 4) describes a future direction in which vkGSplat itself
// ships as the ICD; that path is staged behind a build-time flag and
// scaffolded in src/vulkan/icd_entry.cpp.
#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkgsplat::vk {

struct InstanceCreateInfo {
    std::string app_name = "vkgsplat-app";
    std::uint32_t app_version = 0;
    bool enable_validation = false;
    bool enable_debug_utils = false;
    std::vector<const char*> additional_extensions;
    std::vector<const char*> additional_layers;
};

class Instance {
public:
    Instance() = default;
    ~Instance();

    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) noexcept;
    Instance& operator=(Instance&&) noexcept;

    static Instance create(const InstanceCreateInfo& info);

    [[nodiscard]] VkInstance handle() const noexcept { return handle_; }
    [[nodiscard]] bool       valid()  const noexcept { return handle_ != VK_NULL_HANDLE; }

    [[nodiscard]] const std::vector<VkPhysicalDevice>& physical_devices() const noexcept {
        return physical_devices_;
    }

private:
    VkInstance                  handle_                = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT    debug_messenger_       = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> physical_devices_;
};

// Convenience: log all reported instance extensions/layers to stderr.
void dump_instance_capabilities();

} // namespace vkgsplat::vk
