// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/cuda/tile_renderer.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace vkgsplat::cuda {
namespace {

__device__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ float splat_alpha(const GpuProjectedSplat& s, float px, float py) {
    const float dx = px - s.center_px.x;
    const float dy = py - s.center_px.y;
    const float q = s.conic.x * dx * dx + 2.0f * s.conic.y * dx * dy + s.conic.z * dy * dy;
    if (!isfinite(q) || q > 9.0f) return 0.0f;
    return clamp01(s.opacity) * expf(-0.5f * q);
}

__global__ void tile_kernel(TileRendererLaunch launch,
                            const GpuProjectedSplat* projected,
                            const std::uint32_t* sorted_projected_indices,
                            const GpuTileRange* tile_ranges,
                            float4* output_rgba) {
    const std::uint32_t tile_x = blockIdx.x;
    const std::uint32_t tile_y = blockIdx.y;
    const std::uint32_t local = threadIdx.x;
    const std::uint32_t pixels_per_tile = launch.tile_size * launch.tile_size;
    if (local >= pixels_per_tile) return;

    const std::uint32_t px_in_tile = local % launch.tile_size;
    const std::uint32_t py_in_tile = local / launch.tile_size;
    const std::uint32_t x = tile_x * launch.tile_size + px_in_tile;
    const std::uint32_t y = tile_y * launch.tile_size + py_in_tile;
    if (x >= launch.width || y >= launch.height) return;

    const std::size_t out_index = static_cast<std::size_t>(y) * launch.width + x;
    float4 color = (launch.flags & TILE_RENDERER_CLEAR_OUTPUT) ?
        launch.background : output_rgba[out_index];
    color.w = clamp01(color.w);
    const GpuTileRange range = tile_ranges[tile_y * launch.tiles_x + tile_x];
    const float sx = static_cast<float>(x) + 0.5f;
    const float sy = static_cast<float>(y) + 0.5f;
    const float cutoff = fmaxf(launch.transmittance_cutoff, 0.0f);

    for (std::uint32_t i = 0; i < range.count; ++i) {
        if (color.w >= 1.0f - cutoff) {
            break;
        }
        const std::uint32_t sorted = sorted_projected_indices[range.offset + i];
        const GpuProjectedSplat s = projected[sorted];
        const float a = splat_alpha(s, sx, sy);
        if (a <= 0.0f) {
            continue;
        }
        color.x = s.color.x * a + color.x * (1.0f - a);
        color.y = s.color.y * a + color.y * (1.0f - a);
        color.z = s.color.z * a + color.z * (1.0f - a);
        color.w = a + color.w * (1.0f - a);
    }
    output_rgba[out_index] = color;
}

} // namespace

void launch_tile_renderer(const TileRendererLaunch& launch,
                          const GpuProjectedSplat* projected,
                          const std::uint32_t* sorted_projected_indices,
                          const GpuTileRange* tile_ranges,
                          float4* output_rgba,
                          void* cuda_stream) {
    const dim3 grid(launch.tiles_x, launch.tiles_y, 1);
    const dim3 block(launch.tile_size * launch.tile_size, 1, 1);
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    tile_kernel<<<grid, block, 0, stream>>>(
        launch, projected, sorted_projected_indices, tile_ranges, output_rgba);
}

} // namespace vkgsplat::cuda
