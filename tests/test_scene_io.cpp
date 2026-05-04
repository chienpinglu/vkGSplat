// SPDX-License-Identifier: Apache-2.0
//
// Exercises real .splat decoding instead of only the Scene container.

#include <vksplat/scene.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {

struct SplatRecord {
    float position[3];
    float scale[3];
    std::uint8_t color[4];
    std::uint8_t rotation[4];
};
static_assert(sizeof(SplatRecord) == 32);

bool near(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

} // namespace

int main() {
    using namespace vksplat;

    const auto path = std::filesystem::current_path() / "test_scene_io_fixture.splat";
    const SplatRecord records[] = {
        {
            { 1.0f, 2.0f, 3.0f },
            { 1.0f, 2.0f, 4.0f },
            { 255, 128, 0, 128 },
            { 128, 128, 128, 255 },
        },
        {
            { -1.0f, -2.0f, -3.0f },
            { 0.5f, 1.0f, 8.0f },
            { 0, 64, 255, 64 },
            { 255, 128, 128, 128 },
        },
    };

    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(records), sizeof(records));
    }

    const Scene scene = Scene::load(path);
    std::filesystem::remove(path);

    if (scene.size() != 2) {
        std::fprintf(stderr, "scene size mismatch: %zu\n", scene.size());
        return 1;
    }

    const auto& a = scene.gaussians()[0];
    bool ok = true;
    ok &= near(a.position.x, 1.0f) && near(a.position.y, 2.0f) && near(a.position.z, 3.0f);
    ok &= near(a.scale_log.x, 0.0f) && near(a.scale_log.y, std::log(2.0f)) && near(a.scale_log.z, std::log(4.0f));
    ok &= near(a.rotation.x, 0.0f) && near(a.rotation.y, 0.0f) && near(a.rotation.z, 0.0f);
    ok &= near(a.rotation.w, 127.0f / 128.0f);

    const float alpha = 128.0f / 255.0f;
    ok &= near(a.opacity_logit, std::log(alpha / (1.0f - alpha)));

    constexpr float sh_c0 = 0.28209479177387814f;
    ok &= near(a.sh[0].x, (1.0f - 0.5f) / sh_c0);
    ok &= near(a.sh[0].y, (128.0f / 255.0f - 0.5f) / sh_c0);
    ok &= near(a.sh[0].z, (0.0f - 0.5f) / sh_c0);

    if (!ok) {
        std::fprintf(stderr, ".splat decode mismatch\n");
        return 1;
    }
    return 0;
}
