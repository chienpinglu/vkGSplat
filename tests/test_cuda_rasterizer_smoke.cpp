// SPDX-License-Identifier: Apache-2.0
//
// M0 CUDA rasterizer smoke test: exercise the full public renderer path
// (upload -> preprocess/project -> fixed device tile lists -> tile blend)
// on a tiny synthetic two-splat scene.

#include <vkgsplat/vkgsplat.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error in %s: %s\n", #expr,              \
                         cudaGetErrorString(err__));                            \
            return 1;                                                           \
        }                                                                       \
    } while (0)

bool near(float a, float b, float eps = 1.0e-4f) {
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

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices available\n");
        return 77; // CTest skip
    }

    auto renderer = make_renderer("cuda");
    if (!renderer) {
        std::fprintf(stderr, "CUDA renderer factory returned null\n");
        return 1;
    }

    Scene scene;
    scene.resize(2);
    scene.gaussians()[0] = make_gaussian(
        { 0.0f, 0.0f, 0.25f }, 0.05f, { 1.0f, 0.0f, 0.0f }, 8.0f);
    scene.gaussians()[1] = make_gaussian(
        { 0.0f, 0.0f, 0.0f }, 0.05f, { 0.0f, 0.0f, 1.0f }, 8.0f);

    Camera camera;
    camera.set_resolution(16, 16);
    camera.set_perspective(1.57079632679f, 1.0f, 0.1f, 10.0f);
    camera.look_at({ 0.0f, 0.0f, 1.0f },
                   { 0.0f, 0.0f, 0.0f },
                   { 0.0f, 1.0f, 0.0f });

    RenderParams params;
    params.background = { 0.0f, 0.0f, 0.0f };

    std::vector<float4> pixels(16 * 16, { 1.0f, 1.0f, 1.0f, 1.0f });
    const RenderTarget target{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        pixels.data(),
    };

    renderer->upload(scene);
    const FrameId frame = renderer->render(camera, params, target);
    renderer->wait(frame);

    const float4 corner = pixels[0];
    if (!near(corner.x, 0.0f) || !near(corner.y, 0.0f) ||
        !near(corner.z, 0.0f) || !near(corner.w, 0.0f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not clear background: corner=(%.4f %.4f %.4f %.4f)\n",
                     corner.x, corner.y, corner.z, corner.w);
        return 1;
    }

    const float4 center = pixels[8 * 16 + 8];
    if (!(center.x > 0.35f && center.z > 0.05f && center.x > center.z)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not blend near red over far blue: center=(%.4f %.4f %.4f %.4f)\n",
                     center.x, center.y, center.z, center.w);
        return 1;
    }

    return 0;
}
