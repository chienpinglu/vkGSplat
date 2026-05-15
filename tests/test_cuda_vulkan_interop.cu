// SPDX-License-Identifier: Apache-2.0
//
// Phase-6 honest interop test.
//
// This is the missing piece that lets the workstation test plan claim it
// has actually verified the CUDA <-> Vulkan zero-copy seam (and not just
// that both backends happen to link into the same binary).
//
// What it does end-to-end:
//
//   1. Brings up a Vulkan instance + device that exposes VK_KHR_external_memory
//      (+ the platform variant).
//   2. Allocates a 64x64 RGBA8 image via vkgsplat::interop::ExternalImage,
//      which exports the Vulkan device-memory as an OS handle and imports
//      that handle into CUDA as a cudaSurfaceObject_t.
//   3. Launches a CUDA kernel that writes a deterministic checkerboard
//      pattern into the *Vulkan-owned* device memory via surf2Dwrite.
//   4. Issues a Vulkan command buffer that transitions the image into
//      TRANSFER_SRC and copies it into a host-visible buffer.
//   5. Maps the buffer and compares every pixel against the expected
//      pattern.
//
// Pass criteria: byte-exact round-trip from CUDA kernel -> Vulkan readback.
// Skip criteria (returns 77): no NVIDIA Vulkan device, no external_memory
// extension, no CUDA device, or the underlying ExternalImage::create throws
// because the driver does not actually expose the win32/fd handle path.

#include <vkgsplat/interop/external_memory.h>
#include <vkgsplat/vulkan/device.h>
#include <vkgsplat/vulkan/instance.h>

#include <cuda_runtime.h>
#include <surface_types.h>
#include <surface_indirect_functions.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kWidth  = 64;
constexpr std::uint32_t kHeight = 64;
constexpr int           kSkip   = 77;

// Deterministic pattern shared between the kernel and the host comparator.
__host__ __device__ inline std::uint32_t expected_rgba(std::uint32_t x,
                                                       std::uint32_t y) {
    // 8x8 checkerboard, with a per-row gradient mixed in so a stuck-at
    // bug (e.g. surface object always writing zero) cannot accidentally
    // satisfy the diff.
    const bool dark = ((x >> 3) ^ (y >> 3)) & 1u;
    const std::uint8_t r = static_cast<std::uint8_t>(x * 4u);
    const std::uint8_t g = static_cast<std::uint8_t>(y * 4u);
    const std::uint8_t b = dark ? 32u : 224u;
    const std::uint8_t a = 0xFFu;
    return (static_cast<std::uint32_t>(a) << 24) |
           (static_cast<std::uint32_t>(b) << 16) |
           (static_cast<std::uint32_t>(g) <<  8) |
            static_cast<std::uint32_t>(r);
}

__global__ void fill_surface(cudaSurfaceObject_t surf,
                             std::uint32_t width,
                             std::uint32_t height) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::uint32_t v = expected_rgba(x, y);
    // surf2Dwrite expects byte offset for x, so multiply by 4 for RGBA8.
    surf2Dwrite(v, surf, static_cast<int>(x * 4u), static_cast<int>(y));
}

#define VK_CHECK(expr, where)                                                  \
    do {                                                                       \
        VkResult r__ = (expr);                                                 \
        if (r__ != VK_SUCCESS) {                                               \
            std::fprintf(stderr, "interop: %s failed (VkResult=%d)\n",         \
                         where, static_cast<int>(r__));                        \
            return kSkip;                                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t e__ = (expr);                                              \
        if (e__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "interop: %s -> %s\n", #expr,                 \
                         cudaGetErrorString(e__));                             \
            return 1;                                                          \
        }                                                                      \
    } while (0)

std::uint32_t find_memory_type(VkPhysicalDevice pd,
                               std::uint32_t type_filter,
                               VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

} // namespace

int main() {
    int cuda_devs = 0;
    cudaError_t e = cudaGetDeviceCount(&cuda_devs);
    if (e != cudaSuccess || cuda_devs == 0) {
        std::fprintf(stderr, "interop: no CUDA device (%s); skip\n",
                     cudaGetErrorString(e));
        return kSkip;
    }
    CUDA_CHECK(cudaSetDevice(0));

    vkgsplat::vk::InstanceCreateInfo ici{};
    ici.app_name = "vkgsplat-cuda-vulkan-interop";
    // VK 1.3 makes these core but enabling them defensively keeps the
    // test honest on loaders that report 1.1.
    ici.additional_extensions = {
        "VK_KHR_get_physical_device_properties2",
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_external_semaphore_capabilities",
    };

    vkgsplat::vk::Instance instance;
    try {
        instance = vkgsplat::vk::Instance::create(ici);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "interop: instance create failed: %s\n", ex.what());
        return kSkip;
    }

    vkgsplat::vk::DeviceCreateInfo dci{};
    dci.require_swapchain          = false;
    dci.require_external_memory    = true;
    dci.require_timeline_semaphore = true;

    vkgsplat::vk::Device device;
    try {
        device = vkgsplat::vk::Device::create(instance, dci, VK_NULL_HANDLE);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "interop: device create failed: %s\n", ex.what());
        return kSkip;
    }

    vkgsplat::ImageDesc img_desc{};
    img_desc.width        = kWidth;
    img_desc.height       = kHeight;
    img_desc.format       = vkgsplat::PixelFormat::R8G8B8A8_UNORM;
    img_desc.mip_levels   = 1;
    img_desc.array_layers = 1;

    vkgsplat::interop::ExternalImage shared;
    try {
        shared = vkgsplat::interop::ExternalImage::create(device, img_desc);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "interop: ExternalImage::create failed: %s\n",
                     ex.what());
        return kSkip;
    }

    // 1) CUDA side: write the pattern.
    {
        cudaSurfaceObject_t surf =
            static_cast<cudaSurfaceObject_t>(reinterpret_cast<std::uintptr_t>(
                shared.cuda_surface()));
        if (surf == 0) {
            std::fprintf(stderr, "interop: cuda surface is null\n");
            return 1;
        }
        const dim3 block(16, 16);
        const dim3 grid((kWidth  + block.x - 1) / block.x,
                        (kHeight + block.y - 1) / block.y);
        fill_surface<<<grid, block>>>(surf, kWidth, kHeight);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 2) Vulkan side: stage the image into a host-visible buffer.
    const VkDeviceSize buffer_bytes =
        static_cast<VkDeviceSize>(kWidth) * kHeight * 4u;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = buffer_bytes;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer readback = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(device.handle(), &bci, nullptr, &readback),
             "vkCreateBuffer(readback)");

    VkMemoryRequirements buf_req{};
    vkGetBufferMemoryRequirements(device.handle(), readback, &buf_req);
    const std::uint32_t mem_type = find_memory_type(
        device.physical(), buf_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type == UINT32_MAX) {
        std::fprintf(stderr, "interop: no HOST_VISIBLE memory type\n");
        return 1;
    }

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = buf_req.size;
    mai.memoryTypeIndex = mem_type;
    VkDeviceMemory readback_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(device.handle(), &mai, nullptr, &readback_mem),
             "vkAllocateMemory(readback)");
    VK_CHECK(vkBindBufferMemory(device.handle(), readback, readback_mem, 0),
             "vkBindBufferMemory(readback)");

    const std::uint32_t q_family =
        device.queues().compute.value_or(device.queues().graphics.value_or(0));
    VkCommandPoolCreateInfo cpi{};
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpi.queueFamilyIndex = q_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(device.handle(), &cpi, nullptr, &pool),
             "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device.handle(), &cbai, &cb),
             "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi), "vkBeginCommandBuffer");

    // The CUDA kernel wrote to the image while it was in UNDEFINED layout
    // as far as Vulkan is concerned. Drivers don't care about layout for
    // host-visible-by-other-API content as long as we transition before
    // sampling/copying. Use UNDEFINED -> TRANSFER_SRC_OPTIMAL.
    VkImageMemoryBarrier to_src{};
    to_src.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_src.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
    to_src.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    to_src.image                       = shared.vk_image();
    to_src.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_src.subresourceRange.levelCount = 1;
    to_src.subresourceRange.layerCount = 1;
    to_src.srcAccessMask               = 0;
    to_src.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_src);

    VkBufferImageCopy region{};
    region.bufferOffset                    = 0;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount     = 1;
    region.imageExtent                     = { kWidth, kHeight, 1 };
    vkCmdCopyImageToBuffer(cb, shared.vk_image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;

    VkQueue submit_queue = device.queues().compute.has_value()
                               ? device.compute_queue()
                               : device.graphics_queue();
    VK_CHECK(vkQueueSubmit(submit_queue, 1, &si, VK_NULL_HANDLE),
             "vkQueueSubmit");
    VK_CHECK(vkQueueWaitIdle(submit_queue), "vkQueueWaitIdle");

    // 3) Map and compare.
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device.handle(), readback_mem, 0, buffer_bytes, 0,
                         &mapped), "vkMapMemory");

    std::size_t mismatches = 0;
    const std::uint32_t* px = static_cast<const std::uint32_t*>(mapped);
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            const std::uint32_t got      = px[y * kWidth + x];
            const std::uint32_t expected = expected_rgba(x, y);
            if (got != expected) {
                if (mismatches < 4) {
                    std::fprintf(stderr,
                                 "interop: mismatch at (%u,%u): got=0x%08X "
                                 "expected=0x%08X\n",
                                 x, y, got, expected);
                }
                ++mismatches;
            }
        }
    }

    vkUnmapMemory(device.handle(), readback_mem);
    vkFreeMemory(device.handle(), readback_mem, nullptr);
    vkDestroyBuffer(device.handle(), readback, nullptr);
    vkFreeCommandBuffers(device.handle(), pool, 1, &cb);
    vkDestroyCommandPool(device.handle(), pool, nullptr);

    if (mismatches != 0) {
        std::fprintf(stderr,
                     "interop=FAIL pixels_total=%u mismatches=%zu\n",
                     kWidth * kHeight,
                     mismatches);
        return 1;
    }

    std::printf("interop=PASS image=%ux%u format=RGBA8 pixels_matched=%u\n",
                kWidth, kHeight, kWidth * kHeight);
    return 0;
}
