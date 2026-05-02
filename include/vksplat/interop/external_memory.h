// SPDX-License-Identifier: Apache-2.0
//
// External-memory bridge: allocate a VkImage with VK_KHR_external_memory,
// export its OS handle (file descriptor on Linux, NT HANDLE on Windows),
// and import that handle into CUDA as a cudaExternalMemory_t / cudaArray.
//
// This is the zero-copy seam described in docs/architecture.md: CUDA
// rasterizes into the same physical pixels the Vulkan presentation
// engine subsequently samples for the swapchain blit.
#pragma once

#include "../types.h"

#include <vulkan/vulkan.h>

namespace vksplat::vk { class Device; }

namespace vksplat::interop {

// One side of an externally-shared image. The CUDA side owns a
// cudaExternalMemory_t + cudaMipmappedArray_t / cudaSurfaceObject_t;
// the Vulkan side owns the VkImage + VkDeviceMemory. Lifetimes are
// pinned together: ExternalImage::~ExternalImage() destroys both.
class ExternalImage {
public:
    ExternalImage() = default;
    ~ExternalImage();

    ExternalImage(const ExternalImage&)            = delete;
    ExternalImage& operator=(const ExternalImage&) = delete;
    ExternalImage(ExternalImage&&) noexcept;
    ExternalImage& operator=(ExternalImage&&) noexcept;

    // Allocate VkImage + export handle + import to CUDA in one shot.
    static ExternalImage create(const vksplat::vk::Device& device,
                                const ImageDesc&           desc);

    [[nodiscard]] VkImage          vk_image()        const noexcept { return image_; }
    [[nodiscard]] VkImageView      vk_image_view()   const noexcept { return view_; }
    [[nodiscard]] VkDeviceMemory   vk_memory()       const noexcept { return memory_; }
    [[nodiscard]] VkFormat         vk_format()       const noexcept { return format_; }
    [[nodiscard]] const ImageDesc& desc()            const noexcept { return desc_; }

    // Opaque handles to the CUDA-side resources. Defined as void* in the
    // public surface so this header does not pull in <cuda_runtime.h>.
    [[nodiscard]] void* cuda_external_memory() const noexcept { return cu_ext_memory_; }
    [[nodiscard]] void* cuda_surface()         const noexcept { return cu_surface_; }

private:
    VkDevice         device_ = VK_NULL_HANDLE;
    VkImage          image_  = VK_NULL_HANDLE;
    VkImageView      view_   = VK_NULL_HANDLE;
    VkDeviceMemory   memory_ = VK_NULL_HANDLE;
    VkFormat         format_ = VK_FORMAT_UNDEFINED;
    ImageDesc        desc_{};

    // Opaque to keep CUDA includes out of public headers.
    void* cu_ext_memory_ = nullptr;
    void* cu_surface_    = nullptr;
};

} // namespace vksplat::interop
