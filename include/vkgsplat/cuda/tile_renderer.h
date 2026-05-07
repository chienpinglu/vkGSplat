// SPDX-License-Identifier: Apache-2.0
//
// CUDA tile renderer entry point. This header intentionally avoids CUDA
// runtime types so the public CPU-only build can include it when the
// CUDA backend is disabled.
#pragma once

#include "../gpu_pipeline.h"
#include "../types.h"

#include <cstdint>

namespace vkgsplat::cuda {

enum TileRendererFlags : std::uint32_t {
    TILE_RENDERER_CLEAR_OUTPUT = 1u << 0,
};

struct TileRendererLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tile_size = 16;
    std::uint32_t tiles_x = 0;
    std::uint32_t tiles_y = 0;
    float4 background = { 0.0f, 0.0f, 0.0f, 0.0f };
    float transmittance_cutoff = 1.0e-3f;
    std::uint32_t flags = TILE_RENDERER_CLEAR_OUTPUT;
};

// Device pointers are represented as void* to keep CUDA headers private
// to the .cu translation unit.
void launch_tile_renderer(const TileRendererLaunch& launch,
                          const GpuProjectedSplat* projected,
                          const std::uint32_t* sorted_projected_indices,
                          const GpuTileRange* tile_ranges,
                          float4* output_rgba,
                          void* cuda_stream);

} // namespace vkgsplat::cuda
