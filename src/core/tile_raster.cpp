// SPDX-License-Identifier: Apache-2.0

#include "vksplat/tile_raster.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vksplat {

TileGrid make_tile_grid(const ImageDesc& desc, std::uint32_t tile_size) {
    if (desc.width == 0 || desc.height == 0) {
        throw std::runtime_error("make_tile_grid: image dimensions must be nonzero");
    }
    if (tile_size == 0) {
        throw std::runtime_error("make_tile_grid: tile_size must be nonzero");
    }

    TileGrid grid;
    grid.width = desc.width;
    grid.height = desc.height;
    grid.tile_size = tile_size;
    grid.tiles_x = (desc.width + tile_size - 1) / tile_size;
    grid.tiles_y = (desc.height + tile_size - 1) / tile_size;
    return grid;
}

std::vector<TileBin> build_tile_bins(const TileGrid& grid,
                                     std::span<const ScreenSplatBounds> splats) {
    if (grid.width == 0 || grid.height == 0 || grid.tile_size == 0 ||
        grid.tiles_x == 0 || grid.tiles_y == 0) {
        throw std::runtime_error("build_tile_bins: invalid tile grid");
    }

    std::vector<TileBin> bins;
    bins.reserve(static_cast<std::size_t>(grid.tiles_x) * grid.tiles_y);
    for (std::uint32_t y = 0; y < grid.tiles_y; ++y) {
        for (std::uint32_t x = 0; x < grid.tiles_x; ++x) {
            bins.push_back(TileBin{ x, y, {} });
        }
    }

    const auto tile_index = [&](std::uint32_t x, std::uint32_t y) {
        return static_cast<std::size_t>(y) * grid.tiles_x + x;
    };

    for (const auto& s : splats) {
        const float min_x = std::max(0.0f, s.min_x);
        const float min_y = std::max(0.0f, s.min_y);
        const float max_x = std::min(static_cast<float>(grid.width), s.max_x);
        const float max_y = std::min(static_cast<float>(grid.height), s.max_y);
        if (!(min_x < max_x && min_y < max_y)) continue;

        const auto first_x = static_cast<std::uint32_t>(std::floor(min_x / grid.tile_size));
        const auto first_y = static_cast<std::uint32_t>(std::floor(min_y / grid.tile_size));
        const auto last_x = static_cast<std::uint32_t>(
            std::floor((std::nextafter(max_x, min_x)) / grid.tile_size));
        const auto last_y = static_cast<std::uint32_t>(
            std::floor((std::nextafter(max_y, min_y)) / grid.tile_size));

        for (std::uint32_t y = first_y; y <= std::min(last_y, grid.tiles_y - 1); ++y) {
            for (std::uint32_t x = first_x; x <= std::min(last_x, grid.tiles_x - 1); ++x) {
                bins[tile_index(x, y)].splat_indices.push_back(s.splat_index);
            }
        }
    }

    return bins;
}

} // namespace vksplat
