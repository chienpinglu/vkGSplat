// SPDX-License-Identifier: Apache-2.0
//
// Renderer: the abstract backend interface. The legacy C++/CUDA 3DGS
// rasterizers are behind VKGSPLAT_ENABLE_3DGS while the default path moves
// toward Vulkan ray-tracing seed frames, temporal reconstruction, denoising,
// and CUDA lowering.
#pragma once

#include "camera.h"
#include "scene.h"
#include "types.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

namespace vkgsplat {

// Per-render parameters that are not part of the scene or camera but
// matter for SDG reproducibility (deterministic ordering, seeded jitter).
struct RenderParams {
    std::uint64_t seed = 0;
    bool deterministic = true;
    int  spp = 1;             // samples per pixel for low-sample seed paths
    bool clear_to_background = true;
    float3 background = { 0.0f, 0.0f, 0.0f };
};

// Where the rendered image ends up. INTEROP_IMAGE means the renderer
// writes into a VkImage that has been imported into the compute
// backend via VK_KHR_external_memory; HOST_BUFFER copies to a host
// staging buffer for headless capture.
enum class RenderTargetKind {
    INTEROP_IMAGE,
    HOST_BUFFER,
};

struct RenderTarget {
    RenderTargetKind kind = RenderTargetKind::HOST_BUFFER;
    ImageDesc        desc{};
    void*            user_handle = nullptr;  // VkImage or host pointer
};

class Renderer {
public:
    virtual ~Renderer() = default;

    // Identify the backend at runtime (e.g. "cpp", "cuda", "triton").
    [[nodiscard]] virtual std::string_view backend_name() const = 0;

    // Upload a scene; backends may keep a pinned device copy.
    virtual void upload(const Scene& scene) = 0;

    // Render a single image. Returns the timeline value that, once
    // signalled on the renderer's semaphore, indicates completion.
    virtual FrameId render(const Camera& camera,
                           const RenderParams& params,
                           const RenderTarget& target) = 0;

    // Block until the given frame has finished writing its target.
    virtual void wait(FrameId frame) = 0;
};

// Backend factory. Returns nullptr if the requested backend is not
// compiled in. The "cpp"/"cpu"/"reference" 3DGS backend is available only
// when VKGSPLAT_ENABLE_3DGS is set.
std::unique_ptr<Renderer> make_renderer(std::string_view backend_name);

} // namespace vkgsplat
