// SPDX-License-Identifier: Apache-2.0
//
// VkDevice wrapper. Selects a physical device that exposes the set of
// extensions vkSplat depends on for CUDA interop:
//
//   VK_KHR_external_memory       (+ _fd / _win32 platform variant)
//   VK_KHR_external_semaphore    (+ _fd / _win32 platform variant)
//   VK_KHR_timeline_semaphore    (or core in 1.2+)
//   VK_KHR_swapchain             (only when a presentation surface is requested)
//
// On NVIDIA we additionally pick the queue family that exposes the
// LUID required by cudaImportExternalMemory.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace vksplat::vk {

class Instance;

struct DeviceCreateInfo {
    bool   require_swapchain = true;
    bool   require_external_memory = true;
    bool   require_timeline_semaphore = true;
    // Optional UUID match — useful when there are multiple GPUs and we
    // want the same device CUDA picked via cudaGetDevice().
    std::optional<std::array<std::uint8_t, VK_UUID_SIZE>> match_device_uuid;
};

struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphics;
    std::optional<std::uint32_t> compute;
    std::optional<std::uint32_t> transfer;
    std::optional<std::uint32_t> present;

    [[nodiscard]] bool has_required(bool need_present) const noexcept {
        return graphics.has_value() && compute.has_value() &&
               (!need_present || present.has_value());
    }
};

class Device {
public:
    Device() = default;
    ~Device();

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    static Device create(const Instance&         instance,
                         const DeviceCreateInfo& info,
                         VkSurfaceKHR            surface = VK_NULL_HANDLE);

    [[nodiscard]] VkDevice         handle()           const noexcept { return handle_; }
    [[nodiscard]] VkPhysicalDevice physical()         const noexcept { return physical_; }
    [[nodiscard]] const QueueFamilyIndices& queues()  const noexcept { return queues_; }
    [[nodiscard]] VkQueue graphics_queue()            const noexcept { return graphics_queue_; }
    [[nodiscard]] VkQueue compute_queue()             const noexcept { return compute_queue_; }
    [[nodiscard]] VkQueue present_queue()             const noexcept { return present_queue_; }

    // The physical device UUID — needed to bind to a specific CUDA device.
    [[nodiscard]] const std::array<std::uint8_t, VK_UUID_SIZE>& device_uuid() const noexcept {
        return device_uuid_;
    }

private:
    VkDevice         handle_   = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;

    QueueFamilyIndices queues_{};
    VkQueue            graphics_queue_ = VK_NULL_HANDLE;
    VkQueue            compute_queue_  = VK_NULL_HANDLE;
    VkQueue            present_queue_  = VK_NULL_HANDLE;

    std::array<std::uint8_t, VK_UUID_SIZE> device_uuid_{};
};

} // namespace vksplat::vk
