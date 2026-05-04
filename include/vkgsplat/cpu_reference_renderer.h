// SPDX-License-Identifier: Apache-2.0
//
// Deterministic CPU reference implementation of the vkGSplat 3DGS fast
// path. It is deliberately simple, but follows the production shape:
// project Gaussians, bin into tiles, sort per tile, and blend pixels.
#pragma once

#include "camera.h"
#include "renderer.h"
#include "scene.h"
#include "tile_raster.h"

#include <cstdint>
#include <span>
#include <vector>

namespace vkgsplat {

struct ProjectedSplat {
    std::uint32_t index = 0;
    float2 center{};
    float depth = 0.0f;
    float2 basis_u{};
    float2 basis_v{};
    float conic_a = 1.0f;
    float conic_b = 0.0f;
    float conic_c = 1.0f;
    float3 color{};
    float opacity = 0.0f;
    ScreenSplatBounds bounds{};
};

struct CpuReferenceRenderResult {
    ImageDesc desc{};
    std::vector<float4> pixels;
    std::vector<ProjectedSplat> projected;
    std::vector<ScreenSplatBounds> bounds;
    std::vector<TileBin> bins;
};

struct CpuReferenceRenderOptions {
    std::uint32_t tile_size = 16;
    float splat_extent_sigma = 3.0f;
};

[[nodiscard]] CpuReferenceRenderResult render_3dgs_cpu_reference(
    const Scene& scene,
    const Camera& camera,
    const RenderParams& params,
    const ImageDesc& target_desc,
    const CpuReferenceRenderOptions& options = {});

[[nodiscard]] std::vector<ProjectedSplat> project_3dgs_cpu_reference(
    const Scene& scene,
    const Camera& camera,
    const ImageDesc& target_desc,
    const CpuReferenceRenderOptions& options = {});

} // namespace vkgsplat
