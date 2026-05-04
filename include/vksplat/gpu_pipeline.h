// SPDX-License-Identifier: Apache-2.0
//
// Shared GPU-driven 3DGS pipeline ABI. These POD structs are the bridge
// between the CPU reference, Vulkan mesh-shader path, and future CUDA
// tile backend.
#pragma once

#include "types.h"

#include <cstdint>

namespace vksplat {

struct GpuGaussian {
    float3 position;
    float3 scale_log;
    float4 rotation;
    float opacity_logit;
    float3 sh[Gaussian::sh_coeffs];
};

struct GpuProjectedSplat {
    float2 center_px;
    float depth;
    float2 basis_u_px;
    float2 basis_v_px;
    float3 conic;
    float opacity;
    float3 color;
    std::uint32_t splat_index;
};

struct GpuTileRange {
    std::uint32_t offset;
    std::uint32_t count;
};

struct GpuIndirectParams {
    std::uint32_t splat_count;
    std::uint32_t projected_count;
    std::uint32_t tile_count;
    std::uint32_t mesh_group_count_x;
};

static_assert(sizeof(GpuProjectedSplat) % 4 == 0);
static_assert(sizeof(GpuTileRange) == sizeof(std::uint32_t) * 2);

} // namespace vksplat
