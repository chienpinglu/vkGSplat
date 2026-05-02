// SPDX-License-Identifier: Apache-2.0
//
// Timeline-semaphore bridge: a single semaphore object visible to both
// Vulkan and CUDA, used to order the per-frame handoff:
//
//   t1  Vulkan acquires swapchain image, signals S=N
//   t2  CUDA waits S=N, rasterizes
//   t3  CUDA signals S=N+1
//   t4  Vulkan waits S=N+1, postprocess + present
//
// The same semaphore is reused across frames; only the timeline value
// advances. See docs/architecture.md.
#pragma once

#include "../types.h"

#include <cstdint>

#include <vulkan/vulkan.h>

namespace vksplat::vk { class Device; }

namespace vksplat::interop {

class TimelineSemaphore {
public:
    TimelineSemaphore() = default;
    ~TimelineSemaphore();

    TimelineSemaphore(const TimelineSemaphore&)            = delete;
    TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;
    TimelineSemaphore(TimelineSemaphore&&) noexcept;
    TimelineSemaphore& operator=(TimelineSemaphore&&) noexcept;

    static TimelineSemaphore create(const vksplat::vk::Device& device,
                                    std::uint64_t              initial_value = 0);

    [[nodiscard]] VkSemaphore   vk_handle() const noexcept { return semaphore_; }
    [[nodiscard]] void*         cuda_handle() const noexcept { return cu_semaphore_; }
    [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

    // Advance the host-side counter and return the new value. Both the
    // Vulkan submit and the CUDA stream signal/wait use this value.
    std::uint64_t next() noexcept { return ++value_; }

    // Block on the host until the underlying timeline reaches `value`.
    // Implemented via vkWaitSemaphoresKHR — the CUDA side does not
    // expose a host wait API.
    VkResult wait(std::uint64_t value, std::uint64_t timeout_ns) const;

private:
    VkDevice      device_       = VK_NULL_HANDLE;
    VkSemaphore   semaphore_    = VK_NULL_HANDLE;
    void*         cu_semaphore_ = nullptr;   // cudaExternalSemaphore_t (opaque)
    std::uint64_t value_        = 0;
};

} // namespace vksplat::interop
