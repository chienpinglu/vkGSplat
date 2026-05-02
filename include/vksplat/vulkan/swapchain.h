// SPDX-License-Identifier: Apache-2.0
//
// Swapchain wrapper. Optional in vkSplat — the production SDG path is
// headless. The viewer (apps/viewer) uses this for an interactive
// debug surface.
#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace vksplat::vk {

class Device;

struct SwapchainCreateInfo {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::uint32_t width  = 0;
    std::uint32_t height = 0;
    bool vsync = true;
    std::uint32_t min_image_count = 2;
};

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) noexcept;
    Swapchain& operator=(Swapchain&&) noexcept;

    static Swapchain create(const Device& device, const SwapchainCreateInfo& info);

    void recreate(const Device& device, const SwapchainCreateInfo& info);

    [[nodiscard]] VkSwapchainKHR handle()       const noexcept { return handle_; }
    [[nodiscard]] VkFormat       format()       const noexcept { return format_; }
    [[nodiscard]] VkExtent2D     extent()       const noexcept { return extent_; }
    [[nodiscard]] std::uint32_t  image_count()  const noexcept {
        return static_cast<std::uint32_t>(images_.size());
    }
    [[nodiscard]] const std::vector<VkImage>&     images()      const noexcept { return images_; }
    [[nodiscard]] const std::vector<VkImageView>& image_views() const noexcept { return image_views_; }

    // Acquire the next presentable image. Returns the image index.
    // signal_semaphore is signalled when the image is ready to be
    // rendered into.
    std::uint32_t acquire_next_image(VkSemaphore signal_semaphore,
                                     VkFence     fence = VK_NULL_HANDLE);

    // Present a previously-acquired image after wait_semaphore signals.
    VkResult present(VkQueue queue, std::uint32_t image_index, VkSemaphore wait_semaphore);

private:
    VkDevice                  device_ = VK_NULL_HANDLE;
    VkSwapchainKHR            handle_ = VK_NULL_HANDLE;
    VkFormat                  format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                extent_{};
    std::vector<VkImage>      images_;
    std::vector<VkImageView>  image_views_;
};

} // namespace vksplat::vk
