// SPDX-License-Identifier: Apache-2.0
//
// VkInstance creation. v1 uses the system Vulkan loader and discovers
// physical devices via vkEnumeratePhysicalDevices. The position paper
// (Section 4) outlines a path where vkGSplat itself ships as the ICD;
// that codepath is staged behind VKGSPLAT_BUILD_ICD and is not enabled
// in v1.

#include "vkgsplat/vulkan/instance.h"

#include "vkgsplat/version.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkgsplat::vk {

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*                                       /*user*/)
{
    const char* level = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)   level = "error";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) level = "warn";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) level = "verbose";

    std::fprintf(stderr, "[vkgsplat][vk-%s] %s\n", level, data->pMessage);
    return VK_FALSE;
}

void check(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"Vulkan error in "} + where +
                                 ": " + std::to_string(r));
    }
}

} // namespace

void dump_instance_capabilities() {
    std::uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());
    std::fprintf(stderr, "[vkgsplat] %u instance extensions:\n", ext_count);
    for (const auto& e : exts) {
        std::fprintf(stderr, "  %s (rev %u)\n", e.extensionName, e.specVersion);
    }

    std::uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    std::fprintf(stderr, "[vkgsplat] %u instance layers:\n", layer_count);
    for (const auto& l : layers) {
        std::fprintf(stderr, "  %s -- %s\n", l.layerName, l.description);
    }
}

Instance Instance::create(const InstanceCreateInfo& info) {
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = info.app_name.c_str();
    app.applicationVersion = info.app_version;
    app.pEngineName        = "vkGSplat";
    app.engineVersion      = VK_MAKE_VERSION(VKGSPLAT_VERSION_MAJOR,
                                             VKGSPLAT_VERSION_MINOR,
                                             VKGSPLAT_VERSION_PATCH);
    app.apiVersion         = VK_API_VERSION_1_3;

    std::vector<const char*> extensions = info.additional_extensions;
    if (info.enable_debug_utils) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char*> layers = info.additional_layers;
    if (info.enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    ci.enabledLayerCount       = static_cast<std::uint32_t>(layers.size());
    ci.ppEnabledLayerNames     = layers.empty() ? nullptr : layers.data();

    Instance out;
    check(vkCreateInstance(&ci, nullptr, &out.handle_), "vkCreateInstance");

    if (info.enable_debug_utils) {
        VkDebugUtilsMessengerCreateInfoEXT dbg{};
        dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        dbg.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg.pfnUserCallback = debug_messenger_callback;

        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(out.handle_, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) {
            check(fn(out.handle_, &dbg, nullptr, &out.debug_messenger_),
                  "vkCreateDebugUtilsMessengerEXT");
        }
    }

    std::uint32_t pd_count = 0;
    vkEnumeratePhysicalDevices(out.handle_, &pd_count, nullptr);
    out.physical_devices_.resize(pd_count);
    vkEnumeratePhysicalDevices(out.handle_, &pd_count, out.physical_devices_.data());

    return out;
}

Instance::~Instance() {
    if (debug_messenger_ && handle_) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(handle_, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(handle_, debug_messenger_, nullptr);
    }
    if (handle_) vkDestroyInstance(handle_, nullptr);
}

Instance::Instance(Instance&& o) noexcept
    : handle_(o.handle_),
      debug_messenger_(o.debug_messenger_),
      physical_devices_(std::move(o.physical_devices_)) {
    o.handle_          = VK_NULL_HANDLE;
    o.debug_messenger_ = VK_NULL_HANDLE;
}

Instance& Instance::operator=(Instance&& o) noexcept {
    if (this != &o) {
        this->~Instance();
        handle_           = o.handle_;
        debug_messenger_  = o.debug_messenger_;
        physical_devices_ = std::move(o.physical_devices_);
        o.handle_         = VK_NULL_HANDLE;
        o.debug_messenger_ = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vkgsplat::vk
