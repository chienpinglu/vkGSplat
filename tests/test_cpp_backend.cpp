// SPDX-License-Identifier: Apache-2.0
//
// Renderer factory and host-buffer coverage for the optional C++ 3DGS backend.

#include <vksplat/vksplat.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

vksplat::Gaussian make_gaussian(vksplat::float3 position,
                                float scale,
                                vksplat::float3 color,
                                float opacity_logit) {
    constexpr float sh_c0 = 0.28209479177387814f;
    vksplat::Gaussian g{};
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
    auto renderer = vksplat::make_renderer("cpp");
    if (!renderer) {
        std::fprintf(stderr, "cpp backend factory returned null\n");
        return 1;
    }
    if (renderer->backend_name() != "cpp") {
        std::fprintf(stderr, "unexpected backend name: %.*s\n",
                     static_cast<int>(renderer->backend_name().size()),
                     renderer->backend_name().data());
        return 1;
    }

    vksplat::Scene scene;
    scene.resize(1);
    scene.gaussians()[0] = make_gaussian(
        { 0.0f, 0.0f, 0.0f }, 0.08f, { 1.0f, 0.1f, 0.0f }, 8.0f);
    renderer->upload(scene);

    vksplat::Camera camera;
    camera.set_resolution(16, 16);
    camera.set_perspective(1.57079632679f, 1.0f, 0.1f, 10.0f);
    camera.look_at({ 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    std::vector<std::uint8_t> rgba(16 * 16 * 4);
    vksplat::RenderTarget target;
    target.kind = vksplat::RenderTargetKind::HOST_BUFFER;
    target.desc = { 16, 16, vksplat::PixelFormat::R8G8B8A8_UNORM, 1, 1 };
    target.user_handle = rgba.data();

    vksplat::RenderParams params;
    params.background = { 0.0f, 0.0f, 0.0f };
    const auto frame = renderer->render(camera, params, target);
    renderer->wait(frame);

    const std::size_t center = (8 * 16 + 8) * 4;
    if (!(rgba[center + 0] > 100 && rgba[center + 0] > rgba[center + 1] &&
          rgba[center + 0] > rgba[center + 2] && rgba[center + 3] > 0)) {
        std::fprintf(stderr, "cpp backend center pixel mismatch: rgba=(%u %u %u %u)\n",
                     rgba[center + 0],
                     rgba[center + 1],
                     rgba[center + 2],
                     rgba[center + 3]);
        return 1;
    }

    return 0;
}
