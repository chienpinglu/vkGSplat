// SPDX-License-Identifier: Apache-2.0
//
// Verifies binary little-endian 3DGS PLY decoding for the Kerbl/Lyra
// property layout.

#include <vksplat/scene.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool near(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) <= eps;
}

} // namespace

int main() {
    using namespace vksplat;

    const auto path = std::filesystem::current_path() / "test_scene_ply_fixture.ply";
    const std::vector<std::string> properties = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_rest_0", "f_rest_15", "f_rest_30",
    };

    {
        std::ofstream out(path, std::ios::binary);
        out << "ply\n";
        out << "format binary_little_endian 1.0\n";
        out << "element vertex 1\n";
        for (const auto& p : properties) out << "property float " << p << "\n";
        out << "end_header\n";

        const float row[] = {
            1.0f, 2.0f, 3.0f,
            0.1f, 0.2f, 0.3f,
            4.0f,
            -1.0f, -2.0f, -3.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.4f, 0.5f, 0.6f,
        };
        out.write(reinterpret_cast<const char*>(row), sizeof(row));
    }

    const Scene scene = Scene::load(path);
    std::filesystem::remove(path);

    if (scene.size() != 1) {
        std::fprintf(stderr, "scene size mismatch: %zu\n", scene.size());
        return 1;
    }

    const auto& g = scene.gaussians()[0];
    bool ok = true;
    ok &= near(g.position.x, 1.0f) && near(g.position.y, 2.0f) && near(g.position.z, 3.0f);
    ok &= near(g.sh[0].x, 0.1f) && near(g.sh[0].y, 0.2f) && near(g.sh[0].z, 0.3f);
    ok &= near(g.opacity_logit, 4.0f);
    ok &= near(g.scale_log.x, -1.0f) && near(g.scale_log.y, -2.0f) && near(g.scale_log.z, -3.0f);
    ok &= near(g.rotation.w, 1.0f);
    ok &= near(g.sh[1].x, 0.4f) && near(g.sh[1].y, 0.5f) && near(g.sh[1].z, 0.6f);

    if (!ok) {
        std::fprintf(stderr, "PLY decode mismatch\n");
        return 1;
    }
    return 0;
}
