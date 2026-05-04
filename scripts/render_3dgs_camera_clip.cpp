// SPDX-License-Identifier: Apache-2.0
//
// Generates a visual clip from the CPU 3DGS renderer with an obvious
// camera orbit. This complements the M6 ray-tracing seed clip, which is
// not a 3DGS render.

#include <vkgsplat/cpu_reference_renderer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr float pi = 3.14159265358979323846f;

vkgsplat::Gaussian make_gaussian(vkgsplat::float3 position,
                                vkgsplat::float3 scale,
                                vkgsplat::float3 color,
                                float opacity_logit) {
    constexpr float sh_c0 = 0.28209479177387814f;
    vkgsplat::Gaussian g{};
    g.position = position;
    g.scale_log = { std::log(scale.x), std::log(scale.y), std::log(scale.z) };
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

void write_ppm(const std::filesystem::path& path,
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

std::filesystem::path frame_path(const std::filesystem::path& prefix, int frame) {
    std::ostringstream name;
    name << prefix.string() << "_" << std::setw(3) << std::setfill('0') << frame << ".ppm";
    return name.str();
}

void add_dot(std::vector<vkgsplat::Gaussian>& gaussians,
             vkgsplat::float3 position,
             vkgsplat::float3 color,
             float scale = 0.008f,
             float opacity_logit = 9.0f) {
    gaussians.push_back(make_gaussian(position, { scale, scale, scale }, color, opacity_logit));
}

void add_line(std::vector<vkgsplat::Gaussian>& gaussians,
              vkgsplat::float3 a,
              vkgsplat::float3 b,
              int steps,
              vkgsplat::float3 color,
              float scale = 0.007f) {
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        add_dot(gaussians,
                { a.x + (b.x - a.x) * t,
                  a.y + (b.y - a.y) * t,
                  a.z + (b.z - a.z) * t },
                color,
                scale,
                9.5f);
    }
}

void add_glyph(std::vector<vkgsplat::Gaussian>& gaussians,
               const char* const rows[7],
               float x0,
               float y0,
               float z,
               vkgsplat::float3 color) {
    constexpr float pitch = 0.040f;
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 5; ++x) {
            if (rows[y][x] == '#') {
                add_dot(gaussians,
                        { x0 + static_cast<float>(x) * pitch,
                          y0 - static_cast<float>(y) * pitch,
                          z },
                        color,
                        0.0065f,
                        10.0f);
            }
        }
    }
}

vkgsplat::Scene make_scene() {
    std::vector<vkgsplat::Gaussian> gaussians;

    static const char* const glyph_3[7] = {
        "####.",
        "....#",
        "....#",
        ".###.",
        "....#",
        "....#",
        "####.",
    };
    static const char* const glyph_d[7] = {
        "####.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "####.",
    };
    static const char* const glyph_g[7] = {
        ".####",
        "#....",
        "#....",
        "#.###",
        "#...#",
        "#...#",
        ".###.",
    };
    static const char* const glyph_s[7] = {
        ".####",
        "#....",
        "#....",
        ".###.",
        "....#",
        "....#",
        "####.",
    };

    add_glyph(gaussians, glyph_3, -0.62f, 0.42f,  0.18f, { 0.95f, 0.15f, 0.10f });
    add_glyph(gaussians, glyph_d, -0.34f, 0.42f,  0.06f, { 0.95f, 0.78f, 0.10f });
    add_glyph(gaussians, glyph_g, -0.06f, 0.42f, -0.06f, { 0.10f, 0.58f, 1.00f });
    add_glyph(gaussians, glyph_s,  0.22f, 0.42f, -0.18f, { 0.16f, 0.95f, 0.34f });

    // Crisp reference axes and a depth-staggered base grid make the camera
    // orbit obvious without relying on large blurry splats.
    add_line(gaussians, { -0.74f, -0.02f,  0.22f }, { 0.72f, -0.02f, -0.22f }, 42, { 0.72f, 0.72f, 0.76f }, 0.0055f);
    add_line(gaussians, { -0.78f, -0.50f,  0.30f }, { 0.78f, -0.50f,  0.30f }, 36, { 0.95f, 0.20f, 0.14f }, 0.0060f);
    add_line(gaussians, { -0.78f, -0.64f,  0.00f }, { 0.78f, -0.64f,  0.00f }, 36, { 0.18f, 0.62f, 1.00f }, 0.0060f);
    add_line(gaussians, { -0.78f, -0.78f, -0.30f }, { 0.78f, -0.78f, -0.30f }, 36, { 0.20f, 0.95f, 0.34f }, 0.0060f);

    for (int z = 0; z < 5; ++z) {
        const float wz = -0.42f + static_cast<float>(z) * 0.21f;
        const float shade = 0.22f + static_cast<float>(z) * 0.08f;
        add_line(gaussians,
                 { -0.78f, -0.92f, wz },
                 {  0.78f, -0.92f, wz },
                 28,
                 { shade, shade, shade + 0.05f },
                 0.0048f);
    }
    for (int x = 0; x < 9; ++x) {
        const float wx = -0.78f + static_cast<float>(x) * 0.195f;
        add_line(gaussians,
                 { wx, -0.92f, -0.42f },
                 { wx, -0.92f,  0.42f },
                 16,
                 { 0.28f, 0.28f, 0.34f },
                 0.0045f);
    }

    vkgsplat::Scene scene;
    scene.resize(gaussians.size());
    std::copy(gaussians.begin(), gaussians.end(), scene.gaussians().begin());
    return scene;
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path prefix =
        argc > 1 ? std::filesystem::path(argv[1])
                 : std::filesystem::path("docs/images/cpu_3dgs_camera_clip");
    const std::filesystem::path parent = prefix.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    const vkgsplat::Scene scene = make_scene();

    vkgsplat::RenderParams params;
    params.background = { 0.015f, 0.015f, 0.02f };

    vkgsplat::CpuReferenceRenderOptions options;
    options.tile_size = 16;
    options.splat_extent_sigma = 1.35f;

    constexpr std::uint32_t width = 640;
    constexpr std::uint32_t height = 480;
    constexpr int frame_count = 36;
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        const float t = static_cast<float>(frame_index) / static_cast<float>(frame_count);
        const float angle = -0.95f + 1.90f * t;
        const float radius = 1.75f;

        vkgsplat::Camera camera;
        camera.set_resolution(width, height);
        camera.set_perspective(0.84f, static_cast<float>(width) / static_cast<float>(height), 0.1f, 10.0f);
        camera.look_at({ std::sin(angle) * radius, 0.10f, std::cos(angle) * radius },
                       { 0.0f, -0.04f, 0.05f },
                       { 0.0f, 1.0f, 0.0f });

        const auto rendered = vkgsplat::render_3dgs_cpu_reference(
            scene,
            camera,
            params,
            { width, height, vkgsplat::PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
            options);

        write_ppm(frame_path(prefix, frame_index), rendered.pixels, width, height);
    }

    return 0;
}
