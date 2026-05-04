// SPDX-License-Identifier: Apache-2.0
//
// Timeline-semaphore bridge: a single semaphore object created in
// Vulkan, exported via the OS handle path, then imported into CUDA.
// One semaphore, two API surfaces, monotonically advancing counter.

#include "vkgsplat/interop/timeline_semaphore.h"

#include "vkgsplat/vulkan/device.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
  #include <windows.h>
  #include <vulkan/vulkan_win32.h>
#endif

#include <stdexcept>
#include <utility>

namespace vkgsplat::interop {

namespace {

#define VKGSPLAT_CUDA_CHECK(expr)                                       \
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

} // namespace

TimelineSemaphore TimelineSemaphore::create(const vkgsplat::vk::Device& device,
                                            std::uint64_t              initial_value)
{
    TimelineSemaphore out;
    out.device_ = device.handle();
    out.value_  = initial_value;

#if defined(_WIN32)
    constexpr auto kHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    constexpr auto kHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue  = initial_value;

    VkExportSemaphoreCreateInfo export_info{};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.pNext       = &type_info;
    export_info.handleTypes = kHandleType;

    VkSemaphoreCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ci.pNext = &export_info;

    vk_check(vkCreateSemaphore(out.device_, &ci, nullptr, &out.semaphore_),
             "vkCreateSemaphore (timeline+export)");

    cudaExternalSemaphoreHandleDesc cu_desc{};

#if defined(_WIN32)
    HANDLE handle = nullptr;
    VkSemaphoreGetWin32HandleInfoKHR gwi{};
    gwi.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    gwi.semaphore  = out.semaphore_;
    gwi.handleType = kHandleType;
    auto pfn = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(out.device_, "vkGetSemaphoreWin32HandleKHR"));
    if (!pfn) throw std::runtime_error("vkGetSemaphoreWin32HandleKHR not available");
    vk_check(pfn(out.device_, &gwi, &handle), "vkGetSemaphoreWin32HandleKHR");
    cu_desc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    cu_desc.handle.win32.handle = handle;
#else
    int fd = -1;
    VkSemaphoreGetFdInfoKHR gfd{};
    gfd.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    gfd.semaphore  = out.semaphore_;
    gfd.handleType = kHandleType;
    auto pfn = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(out.device_, "vkGetSemaphoreFdKHR"));
    if (!pfn) throw std::runtime_error("vkGetSemaphoreFdKHR not available");
    vk_check(pfn(out.device_, &gfd, &fd), "vkGetSemaphoreFdKHR");
    cu_desc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    cu_desc.handle.fd = fd;
#endif

    cudaExternalSemaphore_t cu_sem = nullptr;
    VKGSPLAT_CUDA_CHECK(cudaImportExternalSemaphore(&cu_sem, &cu_desc));
    out.cu_semaphore_ = cu_sem;

    return out;
}

VkResult TimelineSemaphore::wait(std::uint64_t value, std::uint64_t timeout_ns) const {
    VkSemaphoreWaitInfo wi{};
    wi.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &semaphore_;
    wi.pValues        = &value;
    return vkWaitSemaphores(device_, &wi, timeout_ns);
}

TimelineSemaphore::~TimelineSemaphore() {
    if (cu_semaphore_) {
        cudaDestroyExternalSemaphore(static_cast<cudaExternalSemaphore_t>(cu_semaphore_));
    }
    if (device_ && semaphore_) {
        vkDestroySemaphore(device_, semaphore_, nullptr);
    }
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore&& o) noexcept
    : device_(o.device_),
      semaphore_(o.semaphore_),
      cu_semaphore_(o.cu_semaphore_),
      value_(o.value_) {
    o.device_       = VK_NULL_HANDLE;
    o.semaphore_    = VK_NULL_HANDLE;
    o.cu_semaphore_ = nullptr;
}

TimelineSemaphore& TimelineSemaphore::operator=(TimelineSemaphore&& o) noexcept {
    if (this != &o) {
        this->~TimelineSemaphore();
        device_       = o.device_;
        semaphore_    = o.semaphore_;
        cu_semaphore_ = o.cu_semaphore_;
        value_        = o.value_;
        o.device_       = VK_NULL_HANDLE;
        o.semaphore_    = VK_NULL_HANDLE;
        o.cu_semaphore_ = nullptr;
    }
    return *this;
}

} // namespace vkgsplat::interop
