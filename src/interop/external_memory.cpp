// SPDX-License-Identifier: Apache-2.0
//
// VkImage <-> CUarray external-memory bridge.
//
// On Linux this exports a POSIX file descriptor via
// VK_KHR_external_memory_fd; on Windows an NT HANDLE via
// VK_KHR_external_memory_win32. Both are imported into CUDA with
// cudaImportExternalMemory + cudaExternalMemoryGetMappedMipmappedArray,
// then wrapped in a cudaSurfaceObject_t for kernel writes.
//
// CUDA <-> Vulkan interop sample patterns:
//   https://github.com/NVIDIA/cuda-samples (vulkanImageCUDA)
// Cited in the position paper as [nvidia2020vkcuda].

#include "vksplat/interop/external_memory.h"

#include "vksplat/vulkan/device.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
  #include <windows.h>
  #include <vulkan/vulkan_win32.h>
#else
  #include <unistd.h>
#endif

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace vksplat::interop {

namespace {

#define VKSPLAT_CUDA_CHECK(expr)                                       \
    do {                                                               \
        cudaError_t err__ = (expr);                                    \
        if (err__ != cudaSuccess) {                                    \
            throw std::runtime_error(std::string{"CUDA error in "} +   \
                #expr + ": " + cudaGetErrorString(err__));             \
        }                                                              \
    } while (0)

void vk_check(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"Vulkan error in "} + where +
                                 ": " + std::to_string(r));
    }
}

VkFormat to_vk(PixelFormat f) {
    switch (f) {
        case PixelFormat::R8G8B8A8_UNORM:        return VK_FORMAT_R8G8B8A8_UNORM;
        case PixelFormat::R8G8B8A8_SRGB:         return VK_FORMAT_R8G8B8A8_SRGB;
        case PixelFormat::R16G16B16A16_SFLOAT:   return VK_FORMAT_R16G16B16A16_SFLOAT;
        case PixelFormat::R32G32B32A32_SFLOAT:   return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

std::uint32_t find_memory_type(VkPhysicalDevice pd,
                               std::uint32_t type_filter,
                               VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mem_props);
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("ExternalImage: no suitable memory type found");
}

} // namespace

ExternalImage ExternalImage::create(const vksplat::vk::Device& device,
                                    const ImageDesc&           desc)
{
    ExternalImage out;
    out.device_ = device.handle();
    out.desc_   = desc;
    out.format_ = to_vk(desc.format);

#if defined(_WIN32)
    constexpr auto kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    constexpr auto kHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkExternalMemoryImageCreateInfo ext_image_info{};
    ext_image_info.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_image_info.handleTypes = kHandleType;

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.pNext         = &ext_image_info;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = out.format_;
    ici.extent        = { desc.width, desc.height, 1 };
    ici.mipLevels     = desc.mip_levels;
    ici.arrayLayers   = desc.array_layers;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_STORAGE_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vk_check(vkCreateImage(out.device_, &ici, nullptr, &out.image_), "vkCreateImage");

    VkMemoryRequirements mem_req{};
    vkGetImageMemoryRequirements(out.device_, out.image_, &mem_req);

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = kHandleType;

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext           = &export_info;
    mai.allocationSize  = mem_req.size;
    mai.memoryTypeIndex = find_memory_type(device.physical(),
                                           mem_req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vk_check(vkAllocateMemory(out.device_, &mai, nullptr, &out.memory_), "vkAllocateMemory");
    vk_check(vkBindImageMemory(out.device_, out.image_, out.memory_, 0), "vkBindImageMemory");

    VkImageViewCreateInfo vi{};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = out.image_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = out.format_;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = desc.mip_levels;
    vi.subresourceRange.layerCount = desc.array_layers;
    vk_check(vkCreateImageView(out.device_, &vi, nullptr, &out.view_), "vkCreateImageView");

    // Export the OS-level handle and import it into CUDA.
    cudaExternalMemoryHandleDesc cu_desc{};
    cu_desc.size = mem_req.size;

#if defined(_WIN32)
    HANDLE handle = nullptr;
    VkMemoryGetWin32HandleInfoKHR ghi{};
    ghi.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    ghi.memory     = out.memory_;
    ghi.handleType = kHandleType;
    auto pfn = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(out.device_, "vkGetMemoryWin32HandleKHR"));
    if (!pfn) throw std::runtime_error("vkGetMemoryWin32HandleKHR not available");
    vk_check(pfn(out.device_, &ghi, &handle), "vkGetMemoryWin32HandleKHR");
    cu_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    cu_desc.handle.win32.handle = handle;
#else
    int fd = -1;
    VkMemoryGetFdInfoKHR gfdi{};
    gfdi.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gfdi.memory     = out.memory_;
    gfdi.handleType = kHandleType;
    auto pfn = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(out.device_, "vkGetMemoryFdKHR"));
    if (!pfn) throw std::runtime_error("vkGetMemoryFdKHR not available");
    vk_check(pfn(out.device_, &gfdi, &fd), "vkGetMemoryFdKHR");
    cu_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    cu_desc.handle.fd = fd;
#endif

    cudaExternalMemory_t cu_ext = nullptr;
    VKSPLAT_CUDA_CHECK(cudaImportExternalMemory(&cu_ext, &cu_desc));
    out.cu_ext_memory_ = cu_ext;

    // Map the imported memory as a 2D mipmapped array, then wrap level 0
    // in a surface object so kernels can do surf2Dwrite into it.
    cudaExternalMemoryMipmappedArrayDesc map_desc{};
    map_desc.offset    = 0;
    map_desc.numLevels = desc.mip_levels;
    map_desc.formatDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    map_desc.extent    = make_cudaExtent(desc.width, desc.height, 0);
    map_desc.flags     = cudaArraySurfaceLoadStore;

    cudaMipmappedArray_t mip = nullptr;
    VKSPLAT_CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mip, cu_ext, &map_desc));

    cudaArray_t level0 = nullptr;
    VKSPLAT_CUDA_CHECK(cudaGetMipmappedArrayLevel(&level0, mip, 0));

    cudaResourceDesc rd{};
    rd.resType         = cudaResourceTypeArray;
    rd.res.array.array = level0;

    cudaSurfaceObject_t surf = 0;
    VKSPLAT_CUDA_CHECK(cudaCreateSurfaceObject(&surf, &rd));
    out.cu_surface_ = reinterpret_cast<void*>(static_cast<std::uintptr_t>(surf));

    return out;
}

ExternalImage::~ExternalImage() {
    if (cu_surface_) {
        cudaDestroySurfaceObject(static_cast<cudaSurfaceObject_t>(
            reinterpret_cast<std::uintptr_t>(cu_surface_)));
    }
    if (cu_ext_memory_) {
        cudaDestroyExternalMemory(static_cast<cudaExternalMemory_t>(cu_ext_memory_));
    }
    if (device_) {
        if (view_)   vkDestroyImageView(device_, view_, nullptr);
        if (image_)  vkDestroyImage(device_, image_, nullptr);
        if (memory_) vkFreeMemory(device_, memory_, nullptr);
    }
}

ExternalImage::ExternalImage(ExternalImage&& o) noexcept
    : device_(o.device_),
      image_(o.image_),
      view_(o.view_),
      memory_(o.memory_),
      format_(o.format_),
      desc_(o.desc_),
      cu_ext_memory_(o.cu_ext_memory_),
      cu_surface_(o.cu_surface_) {
    o.device_ = VK_NULL_HANDLE;
    o.image_  = VK_NULL_HANDLE;
    o.view_   = VK_NULL_HANDLE;
    o.memory_ = VK_NULL_HANDLE;
    o.cu_ext_memory_ = nullptr;
    o.cu_surface_    = nullptr;
}

ExternalImage& ExternalImage::operator=(ExternalImage&& o) noexcept {
    if (this != &o) {
        this->~ExternalImage();
        device_       = o.device_;
        image_        = o.image_;
        view_         = o.view_;
        memory_       = o.memory_;
        format_       = o.format_;
        desc_         = o.desc_;
        cu_ext_memory_ = o.cu_ext_memory_;
        cu_surface_    = o.cu_surface_;
        o.device_ = VK_NULL_HANDLE;
        o.image_  = VK_NULL_HANDLE;
        o.view_   = VK_NULL_HANDLE;
        o.memory_ = VK_NULL_HANDLE;
        o.cu_ext_memory_ = nullptr;
        o.cu_surface_    = nullptr;
    }
    return *this;
}

} // namespace vksplat::interop
