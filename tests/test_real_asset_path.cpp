// SPDX-License-Identifier: Apache-2.0
//
// M6-style integration test: generate a small Kerbl/Lyra-compatible
// 3DGS PLY, load it through Scene::load, and render it through the CPU
// reference pipeline.

#include <vksplat/cpu_reference_renderer.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr float kShC0 = 0.28209479177387814f;

struct AssetGaussian {
    float x, y, z;
    float r, g, b;
    float opacity;
    float sx, sy, sz;
};

void write_row(std::ofstream& out, const AssetGaussian& g) {
    const float row[] = {
        g.x, g.y, g.z,
        (g.r - 0.5f) / kShC0,
        (g.g - 0.5f) / kShC0,
        (g.b - 0.5f) / kShC0,
        g.opacity,
        std::log(g.sx), std::log(g.sy), std::log(g.sz),
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    out.write(reinterpret_cast<const char*>(row), sizeof(row));
}

} // namespace

int main() {
    using namespace vksplat;

    const auto path = std::filesystem::current_path() / "test_real_asset_path.ply";
    const AssetGaussian gaussians[] = {
        { -0.12f,  0.00f, 0.10f, 1.0f, 0.1f, 0.1f, 5.0f, 0.055f, 0.030f, 0.030f },
        {  0.12f,  0.00f, 0.05f, 0.1f, 1.0f, 0.1f, 5.0f, 0.055f, 0.030f, 0.030f },
        {  0.00f, -0.12f, 0.00f, 0.1f, 0.1f, 1.0f, 5.0f, 0.030f, 0.055f, 0.030f },
    };

    {
        std::ofstream out(path, std::ios::binary);
        out << "ply\n";
        out << "format binary_little_endian 1.0\n";
        out << "element vertex 3\n";
        const char* props[] = {
            "x", "y", "z",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        };
        for (const char* p : props) out << "property float " << p << "\n";
        out << "end_header\n";
        for (const auto& g : gaussians) write_row(out, g);
    }

    const Scene scene = Scene::load(path);
    std::filesystem::remove(path);
    if (scene.size() != 3) {
        std::fprintf(stderr, "expected 3 loaded gaussians, got %zu\n", scene.size());
        return 1;
    }

    Camera camera;
    camera.set_resolution(32, 32);
    camera.set_perspective(1.57079632679f, 1.0f, 0.1f, 10.0f);
    camera.look_at({ 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    RenderParams params;
    params.background = { 0.0f, 0.0f, 0.0f };
    CpuReferenceRenderOptions options;
    options.tile_size = 8;

    const auto rendered = render_3dgs_cpu_reference(
        scene, camera, params, { 32, 32, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 }, options);

    if (rendered.projected.size() != 3) {
        std::fprintf(stderr, "expected 3 projected gaussians, got %zu\n", rendered.projected.size());
        return 1;
    }

    float energy = 0.0f;
    std::size_t touched_tiles = 0;
    for (const auto& p : rendered.pixels) energy += p.x + p.y + p.z;
    for (const auto& b : rendered.bins) {
        if (!b.splat_indices.empty()) ++touched_tiles;
    }

    if (!(energy > 5.0f && touched_tiles >= 4)) {
        std::fprintf(stderr, "unexpected real-asset render signal: energy=%.4f touched_tiles=%zu\n",
                     energy, touched_tiles);
        return 1;
    }

    return 0;
}
