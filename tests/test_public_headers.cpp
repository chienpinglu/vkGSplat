// SPDX-License-Identifier: Apache-2.0
//
// Verifies that the umbrella header is usable in the default CPU-only
// build where Vulkan and CUDA SDK headers may not be installed.

#include <vksplat/vksplat.h>

int main() {
    vksplat::Scene scene;
    vksplat::Camera camera;
    auto grid = vksplat::make_tile_grid({ 8, 8, vksplat::PixelFormat::R8G8B8A8_UNORM, 1, 1 }, 4);
    return scene.empty() && camera.width() == 0 && grid.tiles_x == 2 ? 0 : 1;
}
