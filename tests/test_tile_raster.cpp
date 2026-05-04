// SPDX-License-Identifier: Apache-2.0
//
// Contract test for tile-based splat binning. This is the CPU reference
// for the CUDA-friendly renderer shape: fixed tiles, half-open bounds,
// deterministic per-tile splat order.

#include <vkgsplat/tile_raster.h>

#include <cstdio>
#include <vector>

namespace {

bool expect_indices(const vkgsplat::TileBin& bin, std::initializer_list<std::uint32_t> want) {
    const std::vector<std::uint32_t> expected(want);
    if (bin.splat_indices == expected) return true;

    std::fprintf(stderr, "tile (%u,%u) mismatch: got", bin.tile_x, bin.tile_y);
    for (auto v : bin.splat_indices) std::fprintf(stderr, " %u", v);
    std::fprintf(stderr, " expected");
    for (auto v : expected) std::fprintf(stderr, " %u", v);
    std::fprintf(stderr, "\n");
    return false;
}

} // namespace

int main() {
    using namespace vkgsplat;

    const TileGrid grid = make_tile_grid({ 32, 32, PixelFormat::R8G8B8A8_UNORM, 1, 1 }, 16);
    if (grid.tiles_x != 2 || grid.tiles_y != 2) {
        std::fprintf(stderr, "unexpected grid: %ux%u\n", grid.tiles_x, grid.tiles_y);
        return 1;
    }

    const ScreenSplatBounds splats[] = {
        { 7,  2.0f,  2.0f,  7.0f,  7.0f },  // top-left only
        { 8, 15.0f, 15.0f, 17.0f, 17.0f },  // crosses all four tiles
        { 9, 16.0f,  0.0f, 31.9f, 16.0f },  // top-right only, half-open y max
        { 10, -8.0f, 20.0f,  4.0f, 28.0f }, // clipped into bottom-left
        { 11, 40.0f, 40.0f, 50.0f, 50.0f }, // fully outside
    };

    const auto bins = build_tile_bins(grid, splats);
    if (bins.size() != 4) {
        std::fprintf(stderr, "bin count mismatch: %zu\n", bins.size());
        return 1;
    }

    bool ok = true;
    ok &= expect_indices(bins[0], { 7, 8 });
    ok &= expect_indices(bins[1], { 8, 9 });
    ok &= expect_indices(bins[2], { 8, 10 });
    ok &= expect_indices(bins[3], { 8 });
    return ok ? 0 : 1;
}
