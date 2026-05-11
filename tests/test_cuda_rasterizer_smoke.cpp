// SPDX-License-Identifier: Apache-2.0
//
// M0 CUDA rasterizer smoke test: exercise the full public renderer path
// (upload -> preprocess/project -> fixed device tile lists -> tile blend)
// on a tiny synthetic two-splat scene.

#include <vkgsplat/vkgsplat.h>
#include <vkgsplat/cuda/rasterizer.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

std::uint8_t to_u8(float v) {
    const float c = std::min(1.0f, std::max(0.0f, v));
    return static_cast<std::uint8_t>(c * 255.0f + 0.5f);
}

bool write_ppm_rgb8(const std::string& path,
                    const std::uint8_t* rgb,
                    int width,
                    int height,
                    int upscale = 1) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const int W = width * upscale;
    const int H = height * upscale;
    out << "P6\n" << W << ' ' << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
        const int sy = y / upscale;
        for (int x = 0; x < W; ++x) {
            const int sx = x / upscale;
            const std::uint8_t* p = rgb + (sy * width + sx) * 3;
            out.write(reinterpret_cast<const char*>(p), 3);
        }
    }
    return out.good();
}

bool dump_float_rgba(const std::string& path,
                     const vkgsplat::float4* px,
                     int width,
                     int height,
                     int upscale = 16) {
    std::vector<std::uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        rgb[i * 3 + 0] = to_u8(px[i].x);
        rgb[i * 3 + 1] = to_u8(px[i].y);
        rgb[i * 3 + 2] = to_u8(px[i].z);
    }
    return write_ppm_rgb8(path, rgb.data(), width, height, upscale);
}

bool dump_rgba8(const std::string& path,
                const std::uint8_t* rgba,
                int width,
                int height,
                int upscale = 16) {
    std::vector<std::uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return write_ppm_rgb8(path, rgb.data(), width, height, upscale);
}


#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error in %s: %s\n", #expr,              \
                         cudaGetErrorString(err__));                            \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Renamed from `near` because <windows.h> (transitively pulled in by the
// Vulkan-on build via vulkan_win32.h) defines `near` as an empty macro from
// the 16-bit pointer era, which destroys this function signature.
bool nearly_equal(float a, float b, float eps = 1.0e-4f) {
    return std::abs(a - b) <= eps;
}

bool near_u8(std::uint8_t a, std::uint8_t b, std::uint8_t eps = 1) {
    const int delta = static_cast<int>(a) - static_cast<int>(b);
    return std::abs(delta) <= static_cast<int>(eps);
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16u;
    const std::uint32_t exp = (bits >> 10u) & 0x1fu;
    const std::uint32_t mant = bits & 0x03ffu;

    std::uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int shift = 0;
            std::uint32_t normalized = mant;
            while ((normalized & 0x0400u) == 0u) {
                normalized <<= 1u;
                ++shift;
            }
            normalized &= 0x03ffu;
            const std::uint32_t exp32 = static_cast<std::uint32_t>(127 - 14 - shift);
            out = sign | (exp32 << 23u) | (normalized << 13u);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13u);
    } else {
        const std::uint32_t exp32 = exp + (127u - 15u);
        out = sign | (exp32 << 23u) | (mant << 13u);
    }

    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

template <typename Fn>
bool expect_runtime_error(const char* label, Fn fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    std::fprintf(stderr, "%s did not reject invalid launch\n", label);
    return false;
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
    auto* cuda_renderer = dynamic_cast<vkgsplat::cuda::Rasterizer*>(renderer.get());
    if (!cuda_renderer) {
        std::fprintf(stderr, "CUDA renderer factory returned unexpected backend type\n");
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

    Scene empty_scene;
    renderer->upload(empty_scene);
    RenderParams empty_params;
    empty_params.background = { 0.125f, 0.25f, 0.5f };

    std::vector<vkgsplat::float4> empty_pixels(4 * 4, { -1.0f, -1.0f, -1.0f, -1.0f });
    const RenderTarget empty_target{
        RenderTargetKind::HOST_BUFFER,
        { 4, 4, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        empty_pixels.data(),
    };
    const FrameId empty_frame = renderer->render(camera, empty_params, empty_target);
    renderer->wait(empty_frame);
    if (!nearly_equal(empty_pixels[0].x, empty_params.background.x) ||
        !nearly_equal(empty_pixels[0].y, empty_params.background.y) ||
        !nearly_equal(empty_pixels[0].z, empty_params.background.z) ||
        !nearly_equal(empty_pixels[0].w, 0.0f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer empty FP32 clear mismatch: pixel=(%.4f %.4f %.4f %.4f)\n",
                     empty_pixels[0].x,
                     empty_pixels[0].y,
                     empty_pixels[0].z,
                     empty_pixels[0].w);
        return 1;
    }
    const auto empty_stats = cuda_renderer->last_stats();
    if (empty_stats.projected_splats != 0u || empty_stats.nonempty_tiles != 0u ||
        empty_stats.tile_splat_entries != 0u || empty_stats.tile_splat_overflow != 0u ||
        empty_stats.max_tile_splats != 0u) {
        std::fprintf(stderr,
                     "CUDA rasterizer empty stats mismatch: projected=%u nonempty=%u entries=%u overflow=%u max=%u\n",
                     empty_stats.projected_splats,
                     empty_stats.nonempty_tiles,
                     empty_stats.tile_splat_entries,
                     empty_stats.tile_splat_overflow,
                     empty_stats.max_tile_splats);
        return 1;
    }

    std::vector<std::uint8_t> empty_pixels_u8(4 * 4 * 4, 255);
    const RenderTarget empty_target_u8{
        RenderTargetKind::HOST_BUFFER,
        { 4, 4, PixelFormat::R8G8B8A8_UNORM, 1, 1 },
        empty_pixels_u8.data(),
    };
    const FrameId empty_frame_u8 = renderer->render(camera, empty_params, empty_target_u8);
    renderer->wait(empty_frame_u8);
    if (!near_u8(empty_pixels_u8[0], 32u, 1) ||
        !near_u8(empty_pixels_u8[1], 64u, 1) ||
        !near_u8(empty_pixels_u8[2], 128u, 1) ||
        !near_u8(empty_pixels_u8[3], 0u, 1)) {
        std::fprintf(stderr,
                     "CUDA rasterizer empty RGBA8 clear mismatch: pixel=(%u %u %u %u)\n",
                     static_cast<unsigned>(empty_pixels_u8[0]),
                     static_cast<unsigned>(empty_pixels_u8[1]),
                     static_cast<unsigned>(empty_pixels_u8[2]),
                     static_cast<unsigned>(empty_pixels_u8[3]));
        return 1;
    }

    std::vector<std::uint16_t> empty_pixels_f16(4 * 4 * 4, 0xffffu);
    const RenderTarget empty_target_f16{
        RenderTargetKind::HOST_BUFFER,
        { 4, 4, PixelFormat::R16G16B16A16_SFLOAT, 1, 1 },
        empty_pixels_f16.data(),
    };
    const FrameId empty_frame_f16 = renderer->render(camera, empty_params, empty_target_f16);
    renderer->wait(empty_frame_f16);
    if (!nearly_equal(half_to_float(empty_pixels_f16[0]), empty_params.background.x, 1.0e-3f) ||
        !nearly_equal(half_to_float(empty_pixels_f16[1]), empty_params.background.y, 1.0e-3f) ||
        !nearly_equal(half_to_float(empty_pixels_f16[2]), empty_params.background.z, 1.0e-3f) ||
        !nearly_equal(half_to_float(empty_pixels_f16[3]), 0.0f, 1.0e-3f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer empty FP16 clear mismatch: pixel=(%.4f %.4f %.4f %.4f)\n",
                     half_to_float(empty_pixels_f16[0]),
                     half_to_float(empty_pixels_f16[1]),
                     half_to_float(empty_pixels_f16[2]),
                     half_to_float(empty_pixels_f16[3]));
        return 1;
    }

    std::vector<vkgsplat::float4> pixels(16 * 16, { 1.0f, 1.0f, 1.0f, 1.0f });
    const RenderTarget target{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        pixels.data(),
    };

    renderer->upload(scene);
    const auto default_tunables = cuda_renderer->tunables();

    auto invalid_tile_tunables = default_tunables;
    invalid_tile_tunables.tile_size = 33;
    cuda_renderer->set_tunables(invalid_tile_tunables);
    if (!expect_runtime_error("CUDA rasterizer oversized tile", [&] {
            renderer->render(camera, params, target);
        })) {
        return 1;
    }

    auto invalid_capacity_tunables = default_tunables;
    invalid_capacity_tunables.max_splats_per_tile = 0;
    cuda_renderer->set_tunables(invalid_capacity_tunables);
    if (!expect_runtime_error("CUDA rasterizer zero tile capacity", [&] {
            renderer->render(camera, params, target);
        })) {
        return 1;
    }

    cuda_renderer->set_tunables(default_tunables);
    const FrameId frame = renderer->render(camera, params, target);
    renderer->wait(frame);

    const auto out_dir = std::filesystem::absolute("cuda_smoke_output");
    std::filesystem::create_directories(out_dir);
    const auto path_float = (out_dir / "01_float_rgba.ppm").string();
    if (dump_float_rgba(path_float, pixels.data(), 16, 16)) {
        std::fprintf(stdout, "[dump] float buffer  -> %s\n", path_float.c_str());
    }

    const vkgsplat::float4 corner = pixels[0];
    if (!nearly_equal(corner.x, 0.0f) || !nearly_equal(corner.y, 0.0f) ||
        !nearly_equal(corner.z, 0.0f) || !nearly_equal(corner.w, 0.0f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not clear background: corner=(%.4f %.4f %.4f %.4f)\n",
                     corner.x, corner.y, corner.z, corner.w);
        return 1;
    }

    const auto stats = cuda_renderer->last_stats();
    if (stats.projected_splats != 2u || stats.nonempty_tiles != 1u ||
        stats.tile_splat_entries != 2u || stats.tile_splat_overflow != 0u ||
        stats.max_tile_splats != 2u) {
        std::fprintf(stderr,
                     "CUDA rasterizer stats mismatch: projected=%u nonempty=%u entries=%u overflow=%u max=%u\n",
                     stats.projected_splats,
                     stats.nonempty_tiles,
                     stats.tile_splat_entries,
                     stats.tile_splat_overflow,
                     stats.max_tile_splats);
        return 1;
    }

    const vkgsplat::float4 center = pixels[8 * 16 + 8];
    if (!(center.x > 0.35f && center.z > 0.05f && center.x > center.z)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not blend near red over far blue: center=(%.4f %.4f %.4f %.4f)\n",
                     center.x, center.y, center.z, center.w);
        return 1;
    }

    auto compact_tunables = default_tunables;
    compact_tunables.use_compact_tile_lists = true;
    compact_tunables.max_splats_per_tile = 0; // compact path is not capped by fixed tile buckets
    cuda_renderer->set_tunables(compact_tunables);
    std::vector<vkgsplat::float4> compact_pixels(16 * 16, { 1.0f, 1.0f, 1.0f, 1.0f });
    const RenderTarget compact_target{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        compact_pixels.data(),
    };
    const FrameId compact_frame = renderer->render(camera, params, compact_target);
    renderer->wait(compact_frame);
    const auto compact_stats = cuda_renderer->last_stats();
    if (compact_stats.projected_splats != 2u || compact_stats.tile_splat_entries != 2u ||
        compact_stats.tile_splat_overflow != 0u || compact_stats.max_tile_splats != 2u) {
        std::fprintf(stderr,
                     "CUDA rasterizer compact tile stats mismatch: projected=%u entries=%u overflow=%u max=%u\n",
                     compact_stats.projected_splats,
                     compact_stats.tile_splat_entries,
                     compact_stats.tile_splat_overflow,
                     compact_stats.max_tile_splats);
        return 1;
    }
    const vkgsplat::float4 compact_center = compact_pixels[8 * 16 + 8];
    if (!nearly_equal(compact_center.x, center.x, 1.0e-4f) ||
        !nearly_equal(compact_center.y, center.y, 1.0e-4f) ||
        !nearly_equal(compact_center.z, center.z, 1.0e-4f) ||
        !nearly_equal(compact_center.w, center.w, 1.0e-4f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer compact tile output mismatch: compact=(%.4f %.4f %.4f %.4f) fixed=(%.4f %.4f %.4f %.4f)\n",
                     compact_center.x, compact_center.y, compact_center.z, compact_center.w,
                     center.x, center.y, center.z, center.w);
        return 1;
    }
    cuda_renderer->set_tunables(default_tunables);

    RenderParams preserve_params = params;
    preserve_params.clear_to_background = false;
    std::vector<vkgsplat::float4> preserve_pixels(16 * 16, { 0.125f, 0.25f, 0.5f, 0.5f });
    const RenderTarget preserve_target{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        preserve_pixels.data(),
    };
    const FrameId preserve_frame = renderer->render(camera, preserve_params, preserve_target);
    renderer->wait(preserve_frame);
    const vkgsplat::float4 preserve_corner = preserve_pixels[0];
    if (!nearly_equal(preserve_corner.x, 0.125f) || !nearly_equal(preserve_corner.y, 0.25f) ||
        !nearly_equal(preserve_corner.z, 0.5f) || !nearly_equal(preserve_corner.w, 0.5f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer preserve-mode corner mismatch: corner=(%.4f %.4f %.4f %.4f)\n",
                     preserve_corner.x, preserve_corner.y, preserve_corner.z, preserve_corner.w);
        return 1;
    }
    const auto preserve_stats = cuda_renderer->last_stats();
    if (preserve_stats.projected_splats != 2u || preserve_stats.tile_splat_entries != 2u) {
        std::fprintf(stderr,
                     "CUDA rasterizer preserve stats mismatch: projected=%u entries=%u\n",
                     preserve_stats.projected_splats, preserve_stats.tile_splat_entries);
        return 1;
    }

    std::vector<std::uint8_t> preserve_pixels_u8(16 * 16 * 4);
    for (std::size_t i = 0; i < preserve_pixels_u8.size(); i += 4u) {
        preserve_pixels_u8[i + 0u] = 32u;
        preserve_pixels_u8[i + 1u] = 64u;
        preserve_pixels_u8[i + 2u] = 128u;
        preserve_pixels_u8[i + 3u] = 127u;
    }
    const RenderTarget preserve_target_u8{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R8G8B8A8_UNORM, 1, 1 },
        preserve_pixels_u8.data(),
    };
    const FrameId preserve_frame_u8 =
        renderer->render(camera, preserve_params, preserve_target_u8);
    renderer->wait(preserve_frame_u8);
    if (!near_u8(preserve_pixels_u8[0], 32u) || !near_u8(preserve_pixels_u8[1], 64u) ||
        !near_u8(preserve_pixels_u8[2], 128u) || !near_u8(preserve_pixels_u8[3], 127u)) {
        std::fprintf(stderr,
                     "CUDA rasterizer RGBA8 preserve-mode corner mismatch: corner=(%u %u %u %u)\n",
                     static_cast<unsigned>(preserve_pixels_u8[0]),
                     static_cast<unsigned>(preserve_pixels_u8[1]),
                     static_cast<unsigned>(preserve_pixels_u8[2]),
                     static_cast<unsigned>(preserve_pixels_u8[3]));
        return 1;
    }

    std::vector<std::uint16_t> preserve_pixels_f16(16 * 16 * 4);
    for (std::size_t i = 0; i < preserve_pixels_f16.size(); i += 4u) {
        preserve_pixels_f16[i + 0u] = 0x3000u; // 0.125
        preserve_pixels_f16[i + 1u] = 0x3400u; // 0.25
        preserve_pixels_f16[i + 2u] = 0x3800u; // 0.5
        preserve_pixels_f16[i + 3u] = 0x3a00u; // 0.75
    }
    const RenderTarget preserve_target_f16{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R16G16B16A16_SFLOAT, 1, 1 },
        preserve_pixels_f16.data(),
    };
    const FrameId preserve_frame_f16 =
        renderer->render(camera, preserve_params, preserve_target_f16);
    renderer->wait(preserve_frame_f16);
    if (!nearly_equal(half_to_float(preserve_pixels_f16[0]), 0.125f, 1.0e-3f) ||
        !nearly_equal(half_to_float(preserve_pixels_f16[1]), 0.25f, 1.0e-3f) ||
        !nearly_equal(half_to_float(preserve_pixels_f16[2]), 0.5f, 1.0e-3f) ||
        !nearly_equal(half_to_float(preserve_pixels_f16[3]), 0.75f, 1.0e-3f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer FP16 preserve-mode corner mismatch: corner=(%.4f %.4f %.4f %.4f)\n",
                     half_to_float(preserve_pixels_f16[0]),
                     half_to_float(preserve_pixels_f16[1]),
                     half_to_float(preserve_pixels_f16[2]),
                     half_to_float(preserve_pixels_f16[3]));
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

    const auto path_u8 = (out_dir / "02_host_rgba8.ppm").string();
    if (dump_rgba8(path_u8, pixels_u8.data(), 16, 16)) {
        std::fprintf(stdout, "[dump] host RGBA8    -> %s\n", path_u8.c_str());
    }

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

    std::vector<std::uint16_t> pixels_f16(16 * 16 * 4, 0);
    const RenderTarget target_f16{
        RenderTargetKind::HOST_BUFFER,
        { 16, 16, PixelFormat::R16G16B16A16_SFLOAT, 1, 1 },
        pixels_f16.data(),
    };
    const FrameId frame_f16 = renderer->render(camera, params, target_f16);
    renderer->wait(frame_f16);

    if (!nearly_equal(half_to_float(pixels_f16[0]), 0.0f, 1.0e-3f) ||
        !nearly_equal(half_to_float(pixels_f16[1]), 0.0f, 1.0e-3f) ||
        !nearly_equal(half_to_float(pixels_f16[2]), 0.0f, 1.0e-3f) ||
        !nearly_equal(half_to_float(pixels_f16[3]), 0.0f, 1.0e-3f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer did not pack FP16 background: corner=(%.4f %.4f %.4f %.4f)\n",
                     half_to_float(pixels_f16[0]),
                     half_to_float(pixels_f16[1]),
                     half_to_float(pixels_f16[2]),
                     half_to_float(pixels_f16[3]));
        return 1;
    }

    const std::size_t center_f16 = (8 * 16 + 8) * 4;
    const vkgsplat::float4 center_half{
        half_to_float(pixels_f16[center_f16 + 0]),
        half_to_float(pixels_f16[center_f16 + 1]),
        half_to_float(pixels_f16[center_f16 + 2]),
        half_to_float(pixels_f16[center_f16 + 3]),
    };
    if (!nearly_equal(center_half.x, center.x, 2.0e-3f) ||
        !nearly_equal(center_half.y, center.y, 2.0e-3f) ||
        !nearly_equal(center_half.z, center.z, 2.0e-3f) ||
        !nearly_equal(center_half.w, center.w, 2.0e-3f)) {
        std::fprintf(stderr,
                     "CUDA rasterizer FP16 output mismatch: half=(%.4f %.4f %.4f %.4f) float=(%.4f %.4f %.4f %.4f)\n",
                     center_half.x, center_half.y, center_half.z, center_half.w,
                     center.x, center.y, center.z, center.w);
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
    renderer->upload(empty_scene);
    const FrameId empty_frame_surface = renderer->render(camera, empty_params, target_surface);
    renderer->wait(empty_frame_surface);

    std::vector<uchar4> empty_surface_pixels(16 * 16);
    CHECK_CUDA(cudaMemcpy2DFromArray(empty_surface_pixels.data(),
                                     16 * sizeof(uchar4),
                                     surface_array,
                                     0,
                                     0,
                                     16 * sizeof(uchar4),
                                     16,
                                     cudaMemcpyDeviceToHost));
    const uchar4 empty_surface_corner = empty_surface_pixels[0];
    if (!near_u8(empty_surface_corner.x, 32u, 1) ||
        !near_u8(empty_surface_corner.y, 64u, 1) ||
        !near_u8(empty_surface_corner.z, 128u, 1) ||
        !near_u8(empty_surface_corner.w, 0u, 1)) {
        std::fprintf(stderr,
                     "CUDA rasterizer empty INTEROP_IMAGE clear mismatch: corner=(%u %u %u %u)\n",
                     static_cast<unsigned>(empty_surface_corner.x),
                     static_cast<unsigned>(empty_surface_corner.y),
                     static_cast<unsigned>(empty_surface_corner.z),
                     static_cast<unsigned>(empty_surface_corner.w));
        return 1;
    }
    const auto empty_surface_stats = cuda_renderer->last_stats();
    if (empty_surface_stats.projected_splats != 0u ||
        empty_surface_stats.nonempty_tiles != 0u ||
        empty_surface_stats.tile_splat_entries != 0u ||
        empty_surface_stats.tile_splat_overflow != 0u ||
        empty_surface_stats.max_tile_splats != 0u) {
        std::fprintf(stderr,
                     "CUDA rasterizer empty surface stats mismatch: projected=%u nonempty=%u entries=%u overflow=%u max=%u\n",
                     empty_surface_stats.projected_splats,
                     empty_surface_stats.nonempty_tiles,
                     empty_surface_stats.tile_splat_entries,
                     empty_surface_stats.tile_splat_overflow,
                     empty_surface_stats.max_tile_splats);
        return 1;
    }

    renderer->upload(scene);

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

    const auto path_surf = (out_dir / "03_cuda_surface.ppm").string();
    if (dump_rgba8(path_surf,
                   reinterpret_cast<const std::uint8_t*>(surface_pixels.data()),
                   16, 16)) {
        std::fprintf(stdout, "[dump] CUDA surface  -> %s\n", path_surf.c_str());
    }

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

    std::vector<uchar4> preserve_surface_seed(16 * 16);
    for (auto& p : preserve_surface_seed) {
        p = make_uchar4(32u, 64u, 128u, 127u);
    }
    CHECK_CUDA(cudaMemcpy2DToArray(surface_array,
                                   0,
                                   0,
                                   preserve_surface_seed.data(),
                                   16 * sizeof(uchar4),
                                   16 * sizeof(uchar4),
                                   16,
                                   cudaMemcpyHostToDevice));

    const FrameId preserve_frame_surface =
        renderer->render(camera, preserve_params, target_surface);
    renderer->wait(preserve_frame_surface);

    std::vector<uchar4> preserve_surface_pixels(16 * 16);
    CHECK_CUDA(cudaMemcpy2DFromArray(preserve_surface_pixels.data(),
                                     16 * sizeof(uchar4),
                                     surface_array,
                                     0,
                                     0,
                                     16 * sizeof(uchar4),
                                     16,
                                     cudaMemcpyDeviceToHost));

    const uchar4 preserve_surface_corner = preserve_surface_pixels[0];
    if (!near_u8(preserve_surface_corner.x, 32u) ||
        !near_u8(preserve_surface_corner.y, 64u) ||
        !near_u8(preserve_surface_corner.z, 128u) ||
        !near_u8(preserve_surface_corner.w, 127u)) {
        std::fprintf(stderr,
                     "CUDA rasterizer INTEROP_IMAGE preserve-mode corner mismatch: corner=(%u %u %u %u)\n",
                     static_cast<unsigned>(preserve_surface_corner.x),
                     static_cast<unsigned>(preserve_surface_corner.y),
                     static_cast<unsigned>(preserve_surface_corner.z),
                     static_cast<unsigned>(preserve_surface_corner.w));
        return 1;
    }

    const uchar4 preserve_surface_center = preserve_surface_pixels[8 * 16 + 8];
    if (!near_u8(preserve_surface_center.x, preserve_pixels_u8[center_u8 + 0], 2) ||
        !near_u8(preserve_surface_center.y, preserve_pixels_u8[center_u8 + 1], 2) ||
        !near_u8(preserve_surface_center.z, preserve_pixels_u8[center_u8 + 2], 2) ||
        !near_u8(preserve_surface_center.w, preserve_pixels_u8[center_u8 + 3], 2)) {
        std::fprintf(stderr,
                     "CUDA rasterizer INTEROP_IMAGE preserve-mode center mismatch: surface=(%u %u %u %u) host=(%u %u %u %u)\n",
                     static_cast<unsigned>(preserve_surface_center.x),
                     static_cast<unsigned>(preserve_surface_center.y),
                     static_cast<unsigned>(preserve_surface_center.z),
                     static_cast<unsigned>(preserve_surface_center.w),
                     static_cast<unsigned>(preserve_pixels_u8[center_u8 + 0]),
                     static_cast<unsigned>(preserve_pixels_u8[center_u8 + 1]),
                     static_cast<unsigned>(preserve_pixels_u8[center_u8 + 2]),
                     static_cast<unsigned>(preserve_pixels_u8[center_u8 + 3]));
        return 1;
    }

    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaFreeArray(surface_array));

    return 0;
}
