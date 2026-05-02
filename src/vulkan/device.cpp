// SPDX-License-Identifier: Apache-2.0
//
// VkDevice creation. The device picker prioritises:
//   1. A discrete GPU
//   2. With graphics + compute (and optionally present) queue families
//   3. That advertises every extension required for CUDA interop
//   4. Whose UUID matches the requested CUDA device, if pinned
//
// On platforms where CUDA is not available (macOS), the interop
// extensions are not required and the picker falls back to whatever
// discrete (or, failing that, integrated) device is present.

#include "vksplat/vulkan/device.h"

#include "vksplat/vulkan/instance.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vksplat::vk {

namespace {

void check(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"Vulkan error in "} + where +
                                 ": " + std::to_string(r));
    }
}

QueueFamilyIndices find_queue_families(VkPhysicalDevice pd, VkSurfaceKHR surface) {
    QueueFamilyIndices q;
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());

    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& p = props[i];
        if ((p.queueFlags & VK_QUEUE_GRAPHICS_BIT) && !q.graphics) q.graphics = i;
        if ((p.queueFlags & VK_QUEUE_COMPUTE_BIT)  && !q.compute)  q.compute  = i;
        if ((p.queueFlags & VK_QUEUE_TRANSFER_BIT) && !q.transfer) q.transfer = i;

        if (surface != VK_NULL_HANDLE) {
            VkBool32 ok = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &ok);
            if (ok && !q.present) q.present = i;
        }
    }
    return q;
}

bool device_supports_extensions(VkPhysicalDevice pd,
                                const std::vector<const char*>& required) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> avail(count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, avail.data());

    std::set<std::string> needed(required.begin(), required.end());
    for (const auto& e : avail) needed.erase(e.extensionName);
    return needed.empty();
}

std::vector<const char*> required_device_extensions(const DeviceCreateInfo& info) {
    std::vector<const char*> exts;
    if (info.require_swapchain)         exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (info.require_external_memory) {
        exts.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#if defined(_WIN32)
        exts.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
        exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
    }
    if (info.require_timeline_semaphore) {
        // Core in 1.2+, but we still request the extension for clarity
        // and because some vendor drivers report it that way.
        exts.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }
    return exts;
}

int score_device(VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   score += 1000;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;
    score += static_cast<int>(props.limits.maxImageDimension2D / 1024);
    return score;
}

} // namespace

Device Device::create(const Instance&         instance,
                      const DeviceCreateInfo& info,
                      VkSurfaceKHR            surface)
{
    const auto required_exts = required_device_extensions(info);

    VkPhysicalDevice picked = VK_NULL_HANDLE;
    int picked_score = -1;
    QueueFamilyIndices picked_queues{};

    for (VkPhysicalDevice pd : instance.physical_devices()) {
        if (!device_supports_extensions(pd, required_exts)) continue;
        const auto q = find_queue_families(pd, surface);
        if (!q.has_required(info.require_swapchain)) continue;

        if (info.match_device_uuid) {
            VkPhysicalDeviceIDProperties id_props{};
            id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &id_props;
            vkGetPhysicalDeviceProperties2(pd, &props2);
            if (std::memcmp(id_props.deviceUUID, info.match_device_uuid->data(),
                            VK_UUID_SIZE) != 0) continue;
        }

        const int s = score_device(pd);
        if (s > picked_score) {
            picked       = pd;
            picked_score = s;
            picked_queues = q;
        }
    }

    if (picked == VK_NULL_HANDLE) {
        throw std::runtime_error("Device::create: no suitable Vulkan device found");
    }

    // One queue per unique family; if graphics/compute/present collide
    // we collapse to a single VkQueue handle below.
    std::set<std::uint32_t> unique_families;
    unique_families.insert(*picked_queues.graphics);
    unique_families.insert(*picked_queues.compute);
    if (picked_queues.present) unique_families.insert(*picked_queues.present);

    const float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (auto fam : unique_families) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = fam;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;
        qcis.push_back(qci);
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
    timeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline_features.timelineSemaphore = info.require_timeline_semaphore ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &timeline_features;
    dci.queueCreateInfoCount    = static_cast<std::uint32_t>(qcis.size());
    dci.pQueueCreateInfos       = qcis.data();
    dci.enabledExtensionCount   = static_cast<std::uint32_t>(required_exts.size());
    dci.ppEnabledExtensionNames = required_exts.empty() ? nullptr : required_exts.data();

    Device out;
    out.physical_ = picked;
    out.queues_   = picked_queues;
    check(vkCreateDevice(picked, &dci, nullptr, &out.handle_), "vkCreateDevice");

    vkGetDeviceQueue(out.handle_, *picked_queues.graphics, 0, &out.graphics_queue_);
    vkGetDeviceQueue(out.handle_, *picked_queues.compute,  0, &out.compute_queue_);
    if (picked_queues.present) {
        vkGetDeviceQueue(out.handle_, *picked_queues.present, 0, &out.present_queue_);
    }

    VkPhysicalDeviceIDProperties id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(picked, &props2);
    std::memcpy(out.device_uuid_.data(), id_props.deviceUUID, VK_UUID_SIZE);

    std::fprintf(stderr, "[vksplat] using device: %s\n", props2.properties.deviceName);
    return out;
}

Device::~Device() {
    if (handle_) vkDestroyDevice(handle_, nullptr);
}

Device::Device(Device&& o) noexcept
    : handle_(o.handle_),
      physical_(o.physical_),
      queues_(o.queues_),
      graphics_queue_(o.graphics_queue_),
      compute_queue_(o.compute_queue_),
      present_queue_(o.present_queue_),
      device_uuid_(o.device_uuid_) {
    o.handle_   = VK_NULL_HANDLE;
    o.physical_ = VK_NULL_HANDLE;
}

Device& Device::operator=(Device&& o) noexcept {
    if (this != &o) {
        this->~Device();
        handle_         = o.handle_;
        physical_       = o.physical_;
        queues_         = o.queues_;
        graphics_queue_ = o.graphics_queue_;
        compute_queue_  = o.compute_queue_;
        present_queue_  = o.present_queue_;
        device_uuid_    = o.device_uuid_;
        o.handle_       = VK_NULL_HANDLE;
        o.physical_     = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vksplat::vk
