// SPDX-License-Identifier: Apache-2.0
//
// CPU reference helpers for vkGSplat's tile-based software renderer.
// The production backend will run this shape on CUDA/SYCL-style
// workgroups; these helpers keep the binning contract testable without
// requiring a GPU toolchain.
#pragma once

#include "types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace vkgsplat {

struct TileGrid {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tile_size = 16;
    std::uint32_t tiles_x = 0;
    std::uint32_t tiles_y = 0;
};

struct ScreenSplatBounds {
    std::uint32_t splat_index = 0;
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
};

struct TileBin {
    std::uint32_t tile_x = 0;
    std::uint32_t tile_y = 0;
    std::vector<std::uint32_t> splat_indices;
};

[[nodiscard]] TileGrid make_tile_grid(const ImageDesc& desc, std::uint32_t tile_size);

// Builds one bin per tile in row-major order. Bounds are pixel-space,
// half-open rectangles [min,max). Splat order is preserved within each
// tile so later stages can sort deterministically.
[[nodiscard]] std::vector<TileBin> build_tile_bins(const TileGrid& grid,
                                                   std::span<const ScreenSplatBounds> splats);

} // namespace vkgsplat
