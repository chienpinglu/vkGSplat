// SPDX-License-Identifier: Apache-2.0
//
// End-to-end CPU reference 3DGS test: synthetic Gaussians are projected,
// tiled, depth-sorted, and blended into a tiny image.

#include <vkgsplat/cpu_reference_renderer.h>

#include <cmath>
#include <cstdio>

namespace {

bool near(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) <= eps;
}

vkgsplat::Gaussian make_gaussian(vkgsplat::float3 position,
                                float scale,
                                vkgsplat::float3 color,
                                float opacity_logit) {
    constexpr float sh_c0 = 0.28209479177387814f;
    vkgsplat::Gaussian g{};
    g.position = position;
    g.scale_log = { std::log(scale), std::log(scale), std::log(scale) };
    g.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    g.opacity_logit = opacity_logit;
    g.sh[0] = {
        (color.x - 0.5f) / sh_c0,
        (color.y - 0.5f) / sh_c0,
        (color.z - 0.5f) / sh_c0,
    };
    return g;
}

} // namespace

int main() {
    using namespace vkgsplat;

    Scene scene;
    scene.resize(2);
    scene.gaussians()[0] = make_gaussian({ 0.0f, 0.0f, 0.25f }, 0.05f, { 1.0f, 0.0f, 0.0f }, 8.0f);
    scene.gaussians()[1] = make_gaussian({ 0.0f, 0.0f, 0.0f }, 0.05f, { 0.0f, 0.0f, 1.0f }, 8.0f);

    Camera camera;
    camera.set_resolution(16, 16);
    camera.set_perspective(1.57079632679f, 1.0f, 0.1f, 10.0f);
    camera.look_at({ 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    RenderParams params;
    params.background = { 0.0f, 0.0f, 0.0f };

    CpuReferenceRenderOptions options;
    options.tile_size = 8;
    options.splat_extent_sigma = 3.0f;

    const auto result = render_3dgs_cpu_reference(
        scene, camera, params, { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 }, options);

    if (result.pixels.size() != 256 || result.bounds.size() != 2 || result.bins.size() != 4) {
        std::fprintf(stderr, "unexpected result sizes: pixels=%zu bounds=%zu bins=%zu\n",
                     result.pixels.size(), result.bounds.size(), result.bins.size());
        return 1;
    }

    const auto& center = result.pixels[8 * 16 + 8];
    if (!(center.x > 0.40f && center.z > 0.10f && center.x > center.z)) {
        std::fprintf(stderr, "center pixel did not blend near red over far blue: rgb=(%.4f %.4f %.4f)\n",
                     center.x, center.y, center.z);
        return 1;
    }

    const auto& corner = result.pixels[0];
    if (!(near(corner.x, 0.0f) && near(corner.y, 0.0f) && near(corner.z, 0.0f))) {
        std::fprintf(stderr, "corner should stay background: rgb=(%.4f %.4f %.4f)\n",
                     corner.x, corner.y, corner.z);
        return 1;
    }

    std::size_t touched_tiles = 0;
    for (const auto& bin : result.bins) {
        if (!bin.splat_indices.empty()) ++touched_tiles;
    }
    if (touched_tiles != 4) {
        std::fprintf(stderr, "expected center splats to touch all 4 tiles, got %zu\n", touched_tiles);
        return 1;
    }

    Scene anisotropic;
    anisotropic.resize(1);
    anisotropic.gaussians()[0] = make_gaussian({ 0.0f, 0.0f, 0.0f }, 0.03f, { 0.8f, 0.8f, 0.8f }, 8.0f);
    anisotropic.gaussians()[0].scale_log = { std::log(0.12f), std::log(0.02f), std::log(0.02f) };
    const auto projected = project_3dgs_cpu_reference(
        anisotropic, camera, { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 }, options);
    if (projected.size() != 1) {
        std::fprintf(stderr, "expected one projected anisotropic splat, got %zu\n", projected.size());
        return 1;
    }
    const float width = projected[0].bounds.max_x - projected[0].bounds.min_x;
    const float height = projected[0].bounds.max_y - projected[0].bounds.min_y;
    if (!(width > height * 2.0f)) {
        std::fprintf(stderr, "anisotropic covariance did not widen x enough: %.4f x %.4f\n", width, height);
        return 1;
    }

    return 0;
}
