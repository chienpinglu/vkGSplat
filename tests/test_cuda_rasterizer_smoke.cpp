// SPDX-License-Identifier: Apache-2.0
//
// M0 CUDA rasterizer smoke test: exercise the full public renderer path
// (upload -> preprocess/project -> fixed device tile lists -> tile blend)
// on a tiny synthetic two-splat scene.

#include <vkgsplat/vkgsplat.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
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

bool near_u8(std::uint8_t a, std::uint8_t b, std::uint8_t eps = 1) {
    const int delta = static_cast<int>(a) - static_cast<int>(b);
    return std::abs(delta) <= static_cast<int>(eps);
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

    std::vector<vkgsplat::float4> pixels(16 * 16, { 1.0f, 1.0f, 1.0f, 1.0f });
    const RenderTarget target{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        pixels.data(),
    };

    renderer->upload(scene);
    const FrameId frame = renderer->render(camera, params, target);
    renderer->wait(frame);

    const vkgsplat::float4 corner = pixels[0];
    if (!near(corner.x, 0.0f) || !near(corner.y, 0.0f) ||
        !near(corner.z, 0.0f) || !near(corner.w, 0.0f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not clear background: corner=(%.4f %.4f %.4f %.4f)\n",
                     corner.x, corner.y, corner.z, corner.w);
        return 1;
    }

    const vkgsplat::float4 center = pixels[8 * 16 + 8];
    if (!(center.x > 0.35f && center.z > 0.05f && center.x > center.z)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not blend near red over far blue: center=(%.4f %.4f %.4f %.4f)\n",
                     center.x, center.y, center.z, center.w);
        return 1;
    }

    std::vector<std::uint8_t> pixels_u8(16 * 16 * 4, 255);
    const RenderTarget target_u8{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R8G8B8A8_UNORM, 1, 1 },
        pixels_u8.data(),
    };
    const FrameId frame_u8 = renderer->render(camera, params, target_u8);
    renderer->wait(frame_u8);

    if (pixels_u8[0] != 0 || pixels_u8[1] != 0 ||
        pixels_u8[2] != 0 || pixels_u8[3] != 0) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not pack RGBA8 background: corner=(%u %u %u %u)\n",
                     static_cast<unsigned>(pixels_u8[0]),
                     static_cast<unsigned>(pixels_u8[1]),
                     static_cast<unsigned>(pixels_u8[2]),
                     static_cast<unsigned>(pixels_u8[3]));
        return 1;
    }

    const std::size_t center_u8 = (8 * 16 + 8) * 4;
    if (!(pixels_u8[center_u8 + 0] > 90 &&
          pixels_u8[center_u8 + 2] > 10 &&
          pixels_u8[center_u8 + 0] > pixels_u8[center_u8 + 2])) {
        std::fprintf(stderr,
                     "CUDA rasterizer RGBA8 center did not preserve red-over-blue order: center=(%u %u %u %u)\n",
                     static_cast<unsigned>(pixels_u8[center_u8 + 0]),
                     static_cast<unsigned>(pixels_u8[center_u8 + 1]),
                     static_cast<unsigned>(pixels_u8[center_u8 + 2]),
                     static_cast<unsigned>(pixels_u8[center_u8 + 3]));
        return 1;
    }

    const auto expected_alpha = static_cast<std::uint8_t>(
        std::min(255.0f, std::max(0.0f, center.w) * 255.0f + 0.5f));
    if (!near_u8(pixels_u8[center_u8 + 3], expected_alpha, 2)) {
        std::fprintf(stderr,
                     "CUDA rasterizer RGBA8 alpha mismatch: packed=%u float=%.4f expected=%u\n",
                     static_cast<unsigned>(pixels_u8[center_u8 + 3]),
                     center.w,
                     static_cast<unsigned>(expected_alpha));
        return 1;
    }

    cudaArray_t surface_array = nullptr;
    cudaSurfaceObject_t surface = 0;
    const cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    CHECK_CUDA(cudaMallocArray(&surface_array,
                               &channel_desc,
                               16,
                               16,
                               cudaArraySurfaceLoadStore));

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = surface_array;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resource_desc));

    const RenderTarget target_surface{
        RenderTargetKind::INTEROP_IMAGE,
        { 16, 16, PixelFormat::R8G8B8A8_UNORM, 1, 1 },
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(surface)),
    };
    const FrameId frame_surface = renderer->render(camera, params, target_surface);
    renderer->wait(frame_surface);

    std::vector<uchar4> surface_pixels(16 * 16);
    CHECK_CUDA(cudaMemcpy2DFromArray(surface_pixels.data(),
                                     16 * sizeof(uchar4),
                                     surface_array,
                                     0,
                                     0,
                                     16 * sizeof(uchar4),
                                     16,
                                     cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaFreeArray(surface_array));

    const uchar4 surface_corner = surface_pixels[0];
    if (surface_corner.x != 0 || surface_corner.y != 0 ||
        surface_corner.z != 0 || surface_corner.w != 0) {
        std::fprintf(stderr,
                     "CUDA rasterizer INTEROP_IMAGE did not clear background: corner=(%u %u %u %u)\n",
                     static_cast<unsigned>(surface_corner.x),
                     static_cast<unsigned>(surface_corner.y),
                     static_cast<unsigned>(surface_corner.z),
                     static_cast<unsigned>(surface_corner.w));
        return 1;
    }

    const uchar4 surface_center = surface_pixels[8 * 16 + 8];
    if (!near_u8(surface_center.x, pixels_u8[center_u8 + 0], 2) ||
        !near_u8(surface_center.y, pixels_u8[center_u8 + 1], 2) ||
        !near_u8(surface_center.z, pixels_u8[center_u8 + 2], 2) ||
        !near_u8(surface_center.w, pixels_u8[center_u8 + 3], 2)) {
        std::fprintf(stderr,
                     "CUDA rasterizer INTEROP_IMAGE mismatch: surface=(%u %u %u %u) host=(%u %u %u %u)\n",
                     static_cast<unsigned>(surface_center.x),
                     static_cast<unsigned>(surface_center.y),
                     static_cast<unsigned>(surface_center.z),
                     static_cast<unsigned>(surface_center.w),
                     static_cast<unsigned>(pixels_u8[center_u8 + 0]),
                     static_cast<unsigned>(pixels_u8[center_u8 + 1]),
                     static_cast<unsigned>(pixels_u8[center_u8 + 2]),
                     static_cast<unsigned>(pixels_u8[center_u8 + 3]));
        return 1;
    }

    return 0;
}
