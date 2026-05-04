// SPDX-License-Identifier: Apache-2.0
//
// Generates a small visual artifact for docs/test_report.md from the same
// synthetic scene used by tests/test_cpu_3dgs_render.cpp.

#include <vkgsplat/cpu_reference_renderer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {

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

std::uint8_t to_byte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(value * 255.0f));
}

void write_ppm(const char* path,
               const std::vector<vkgsplat::float4>& pixels,
               std::uint32_t width,
               std::uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open output image");

    out << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& p : pixels) {
        const unsigned char rgb[] = { to_byte(p.x), to_byte(p.y), to_byte(p.z) };
        out.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* output = argc > 1 ? argv[1] : "cpu_3dgs_render.ppm";

    vkgsplat::Scene scene;
    scene.resize(2);
    scene.gaussians()[0] =
        make_gaussian({ 0.0f, 0.0f, 0.25f }, 0.05f, { 1.0f, 0.0f, 0.0f }, 8.0f);
    scene.gaussians()[1] =
        make_gaussian({ 0.0f, 0.0f, 0.0f }, 0.05f, { 0.0f, 0.0f, 1.0f }, 8.0f);

    vkgsplat::Camera camera;
    camera.set_resolution(16, 16);
    camera.set_perspective(1.57079632679f, 1.0f, 0.1f, 10.0f);
    camera.look_at({ 0.0f, 0.0f, 1.0f },
                   { 0.0f, 0.0f, 0.0f },
                   { 0.0f, 1.0f, 0.0f });

    vkgsplat::RenderParams params;
    params.background = { 0.0f, 0.0f, 0.0f };

    vkgsplat::CpuReferenceRenderOptions options;
    options.tile_size = 8;
    options.splat_extent_sigma = 3.0f;

    constexpr std::uint32_t width = 16;
    constexpr std::uint32_t height = 16;
    const auto result = render_3dgs_cpu_reference(
        scene,
        camera,
        params,
        { width, height, vkgsplat::PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        options);

    write_ppm(output, result.pixels, width, height);
    return 0;
}
