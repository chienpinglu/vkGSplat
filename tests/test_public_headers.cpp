// SPDX-License-Identifier: Apache-2.0
//
// Verifies that the umbrella header is usable in the default CPU-only
// build where Vulkan and CUDA SDK headers may not be installed.

#include <vkgsplat/vkgsplat.h>

int main() {
    vkgsplat::Scene scene;
    vkgsplat::Camera camera;
    auto grid = vkgsplat::make_tile_grid({ 8, 8, vkgsplat::PixelFormat::R8G8B8A8_UNORM, 1, 1 }, 4);
    return scene.empty() && camera.width() == 0 && grid.tiles_x == 2 ? 0 : 1;
}
