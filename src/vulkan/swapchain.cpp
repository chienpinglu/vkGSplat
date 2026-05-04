// SPDX-License-Identifier: Apache-2.0
//
// VkSwapchain creation for the optional debug viewer. Picks an SRGB
// 8-bit format for general compatibility; SDG production paths skip
// the swapchain entirely and write to interop images instead.

#include "vkgsplat/vulkan/swapchain.h"

#include "vkgsplat/vulkan/device.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vkgsplat::vk {

namespace {

void check(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"Vulkan error in "} + where +
                                 ": " + std::to_string(r));
    }
}

VkSurfaceFormatKHR pick_format(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.front();
}

VkPresentModeKHR pick_present_mode(const std::vector<VkPresentModeKHR>& modes, bool vsync) {
    if (vsync) return VK_PRESENT_MODE_FIFO_KHR; // always available
    for (auto m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

} // namespace

Swapchain Swapchain::create(const Device& device, const SwapchainCreateInfo& info) {
    Swapchain s;
    s.recreate(device, info);
    return s;
}

void Swapchain::recreate(const Device& device, const SwapchainCreateInfo& info) {
    device_ = device.handle();

    if (handle_) {
        for (auto v : image_views_) vkDestroyImageView(device_, v, nullptr);
        image_views_.clear();
        images_.clear();
        vkDestroySwapchainKHR(device_, handle_, nullptr);
        handle_ = VK_NULL_HANDLE;
    }

    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical(), info.surface, &caps);

    std::uint32_t fc = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical(), info.surface, &fc, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fc);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical(), info.surface, &fc, formats.data());

    std::uint32_t pmc = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device.physical(), info.surface, &pmc, nullptr);
    std::vector<VkPresentModeKHR> modes(pmc);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device.physical(), info.surface, &pmc, modes.data());

    const auto fmt = pick_format(formats);
    const auto pm  = pick_present_mode(modes, info.vsync);

    extent_.width  = std::clamp(info.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    extent_.height = std::clamp(info.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    format_ = fmt.format;

    std::uint32_t image_count = std::max(info.min_image_count, caps.minImageCount);
    if (caps.maxImageCount > 0) image_count = std::min(image_count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = info.surface;
    ci.minImageCount    = image_count;
    ci.imageFormat      = fmt.format;
    ci.imageColorSpace  = fmt.colorSpace;
    ci.imageExtent      = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = pm;
    ci.clipped          = VK_TRUE;

    check(vkCreateSwapchainKHR(device_, &ci, nullptr, &handle_), "vkCreateSwapchainKHR");

    std::uint32_t real_count = 0;
    vkGetSwapchainImagesKHR(device_, handle_, &real_count, nullptr);
    images_.resize(real_count);
    vkGetSwapchainImagesKHR(device_, handle_, &real_count, images_.data());

    image_views_.resize(real_count);
    for (std::uint32_t i = 0; i < real_count; ++i) {
        VkImageViewCreateInfo vi{};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = format_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        check(vkCreateImageView(device_, &vi, nullptr, &image_views_[i]), "vkCreateImageView");
    }
}

std::uint32_t Swapchain::acquire_next_image(VkSemaphore signal_semaphore, VkFence fence) {
    std::uint32_t index = 0;
    check(vkAcquireNextImageKHR(device_, handle_, UINT64_MAX,
                                signal_semaphore, fence, &index),
          "vkAcquireNextImageKHR");
    return index;
}

VkResult Swapchain::present(VkQueue queue, std::uint32_t image_index, VkSemaphore wait_semaphore) {
    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = wait_semaphore ? 1u : 0u;
    pi.pWaitSemaphores    = wait_semaphore ? &wait_semaphore : nullptr;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &handle_;
    pi.pImageIndices      = &image_index;
    return vkQueuePresentKHR(queue, &pi);
}

Swapchain::~Swapchain() {
    if (device_ == VK_NULL_HANDLE) return;
    for (auto v : image_views_) vkDestroyImageView(device_, v, nullptr);
    if (handle_) vkDestroySwapchainKHR(device_, handle_, nullptr);
}

Swapchain::Swapchain(Swapchain&& o) noexcept
    : device_(o.device_),
      handle_(o.handle_),
      format_(o.format_),
      extent_(o.extent_),
      images_(std::move(o.images_)),
      image_views_(std::move(o.image_views_)) {
    o.device_ = VK_NULL_HANDLE;
    o.handle_ = VK_NULL_HANDLE;
}

Swapchain& Swapchain::operator=(Swapchain&& o) noexcept {
    if (this != &o) {
        this->~Swapchain();
        device_      = o.device_;
        handle_      = o.handle_;
        format_      = o.format_;
        extent_      = o.extent_;
        images_      = std::move(o.images_);
        image_views_ = std::move(o.image_views_);
        o.device_    = VK_NULL_HANDLE;
        o.handle_    = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vkgsplat::vk
