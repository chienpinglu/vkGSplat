// SPDX-License-Identifier: Apache-2.0
//
// CUDA backend: 3D Gaussian Splatting forward rasterizer.
//
// The kernel layout follows Kerbl et al. 2023 (the reference 3DGS
// implementation): per-Gaussian preprocess, screen-space tile binning,
// per-tile radix sort, and front-to-back alpha blending. v1 is the
// forward pass only — backward (training) is on the roadmap.
//
// SDG-specific extensions vs the reference:
//   * deterministic ordering (stable sort, fixed seed for jitter)
//   * deeper sample budgets per pixel (configurable spp)
//   * optional denoise pass after blend (future)
//
// This header is C++ only — actual CUDA types live in the .cu unit
// behind an opaque pointer to keep .cu out of the public surface.
#pragma once

#include "../renderer.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vkgsplat::cuda {

class RasterizerImpl; // forward decl, defined in src/cuda/rasterizer.cu

struct RasterizerFrameStats {
    std::uint32_t projected_splats = 0;
    std::uint32_t nonempty_tiles = 0;
    std::uint32_t tile_splat_entries = 0;
    std::uint32_t tile_splat_overflow = 0;
    std::uint32_t max_tile_splats = 0;
};

// CUDA-resident 3DGS rasterizer implementing the Renderer interface.
class Rasterizer final : public Renderer {
public:
    Rasterizer();
    ~Rasterizer() override;

    Rasterizer(const Rasterizer&)            = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;

    [[nodiscard]] std::string_view backend_name() const override { return "cuda"; }

    void   upload(const Scene& scene) override;
    FrameId render(const Camera& camera,
                   const RenderParams& params,
                   const RenderTarget& target) override;
    void   wait(FrameId frame) override;

    // Bind a CUDA device by UUID — used to ensure the renderer runs on
    // the same physical GPU as the Vulkan device, which is required
    // for VK_KHR_external_memory imports to succeed.
    void bind_to_device_uuid(const std::array<unsigned char, 16>& uuid);

    [[nodiscard]] RasterizerFrameStats last_stats() const;

private:
    std::unique_ptr<RasterizerImpl> impl_;
};

// Tunables that match parameters in the reference 3DGS rasterizer.
// Exposed so the Vulkan front-end can pass through application-side
// settings (e.g. via VK_VKGSPLAT_gaussian_splatting).
struct RasterizerTunables {
    int   tile_size      = 16;     // pixels per tile edge
    float density_clamp  = 100.0f; // numerical safety for tiny scales
    int   max_radii_pixels = 256;  // skip pathologically large gaussians
    int   max_splats_per_tile = 1024; // M0 fixed-capacity deterministic tile lists
    bool  use_anti_aliased_filter = true; // Mip-Splatting style filter
};

} // namespace vkgsplat::cuda
