// SPDX-License-Identifier: Apache-2.0
//
// Generates visual frames for the M6 ray-tracing seed fixture. Each output
// image is a side-by-side panel: noisy 1-spp seed on the left, denoised
// temporal reconstruction on the right.

#include <vkgsplat/raytrace_seed.h>

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

vkgsplat::float4 color(float r, float g, float b) {
    return { r, g, b, 1.0f };
}

std::uint8_t to_byte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(value * 255.0f));
}

void add_quad(vkgsplat::RayTracingScene& scene,
              vkgsplat::float3 a,
              vkgsplat::float3 b,
              vkgsplat::float3 c,
              vkgsplat::float3 d,
              std::uint32_t material,
              std::uint32_t primitive) {
    scene.triangles.push_back({ a, b, c, material, primitive });
    scene.triangles.push_back({ a, c, d, material, primitive });
}

vkgsplat::RayTracingScene make_scene() {
    vkgsplat::RayTracingScene scene;
    scene.materials.push_back({ color(0.78f, 0.78f, 0.72f) });
    scene.materials.push_back({ color(0.92f, 0.22f, 0.12f) });
    scene.materials.push_back({ color(0.12f, 0.38f, 0.95f) });
    scene.materials.push_back({ color(0.18f, 0.78f, 0.36f) });

    add_quad(scene,
             { -2.0f, -1.15f, 4.2f }, {  2.0f, -1.15f, 4.2f },
             {  2.0f,  1.15f, 4.2f }, { -2.0f,  1.15f, 4.2f },
             0, 10);
    add_quad(scene,
             { -1.55f, -0.85f, 2.7f }, { -0.20f, -0.85f, 2.7f },
             { -0.20f,  0.80f, 2.7f }, { -1.55f,  0.80f, 2.7f },
             1, 20);
    add_quad(scene,
             { 0.10f, -0.75f, 3.15f }, { 1.45f, -0.75f, 3.15f },
             { 1.45f,  0.90f, 3.15f }, { 0.10f,  0.90f, 3.15f },
             2, 30);
    add_quad(scene,
             { -1.20f, -1.05f, 2.05f }, { 0.95f, -1.05f, 2.05f },
             { 0.95f, -0.45f, 2.55f }, { -1.20f, -0.45f, 2.55f },
             3, 40);

    return scene;
}

std::vector<vkgsplat::float4> compose_panel(const vkgsplat::RayTracingSeedFrame& seed,
                                           const vkgsplat::SvgfDenoiseResult& denoised) {
    constexpr std::uint32_t divider = 3;
    const std::uint32_t panel_width = seed.width * 2 + divider;
    const std::uint32_t panel_height = seed.height;
    std::vector<vkgsplat::float4> panel(
        static_cast<std::size_t>(panel_width) * panel_height,
        color(0.02f, 0.02f, 0.025f));

    for (std::uint32_t y = 0; y < seed.height; ++y) {
        for (std::uint32_t x = 0; x < seed.width; ++x) {
            const std::size_t src = static_cast<std::size_t>(y) * seed.width + x;
            const std::size_t noisy_dst = static_cast<std::size_t>(y) * panel_width + x;
            const std::size_t denoise_dst =
                static_cast<std::size_t>(y) * panel_width + seed.width + divider + x;
            panel[noisy_dst] = seed.color[src];
            panel[denoise_dst] = denoised.color[src];
        }
        for (std::uint32_t x = seed.width; x < seed.width + divider; ++x) {
            panel[static_cast<std::size_t>(y) * panel_width + x] = color(0.9f, 0.9f, 0.9f);
        }
    }

    return panel;
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

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path prefix =
        argc > 1 ? std::filesystem::path(argv[1])
                 : std::filesystem::path("docs/images/raytrace_seed_clip");
    const std::filesystem::path parent = prefix.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    const vkgsplat::RayTracingScene scene = make_scene();
    vkgsplat::RayTracingSeedFrame previous_seed;
    bool has_previous = false;

    constexpr int frame_count = 24;
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        const float t = static_cast<float>(frame_index) / static_cast<float>(frame_count - 1);
        const float camera_x = -0.16f + 0.32f * t;

        vkgsplat::RayTracingCamera camera;
        camera.eye = { camera_x, 0.0f, 0.0f };
        camera.target = { camera_x * 0.25f, 0.0f, 3.2f };
        camera.up = { 0.0f, 1.0f, 0.0f };
        camera.fov_y_radians = 0.82f;
        camera.z_near = 0.1f;
        camera.z_far = 16.0f;

        vkgsplat::RayTracingDispatch dispatch;
        dispatch.width = 128;
        dispatch.height = 72;
        dispatch.samples_per_pixel = 1;
        dispatch.seed = 9001u + static_cast<std::uint32_t>(frame_index * 97);
        dispatch.radiance_noise = 0.55f;

        const vkgsplat::RayTracingSeedFrame seed =
            vkgsplat::trace_raytracing_seed(scene, camera, dispatch);

        vkgsplat::ReprojectionResult reprojected;
        if (has_previous) {
            const vkgsplat::CameraMotionMap motion = vkgsplat::compute_camera_motion_map(
                seed.width,
                seed.height,
                seed.ndc_depth,
                seed.inverse_view_projection,
                previous_seed.view_projection);

            vkgsplat::ReprojectionOptions reprojection_options;
            reprojection_options.history_weight = 0.85f;
            reprojection_options.depth_threshold = 0.18f;
            reprojected = vkgsplat::reproject_history(
                vkgsplat::as_reprojection_frame(previous_seed),
                vkgsplat::as_reprojection_frame(seed),
                motion.current_to_previous_px,
                reprojection_options);
        } else {
            const std::size_t count = static_cast<std::size_t>(seed.width) * seed.height;
            reprojected.width = seed.width;
            reprojected.height = seed.height;
            reprojected.color = seed.color;
            reprojected.valid_history.assign(count, 0);
        }

        vkgsplat::SvgfDenoiseOptions denoise_options;
        denoise_options.history_weight = 0.55f;
        denoise_options.depth_threshold = 0.18f;
        denoise_options.spatial_radius = 1;

        const vkgsplat::SvgfDenoiseResult denoised =
            vkgsplat::denoise_svgf_baseline(vkgsplat::as_denoise_frame(seed),
                                            reprojected,
                                            denoise_options);

        const auto panel = compose_panel(seed, denoised);
        write_ppm(frame_path(prefix, frame_index),
                  panel,
                  seed.width * 2 + 3,
                  seed.height);

        previous_seed = seed;
        has_previous = true;
    }

    return 0;
}
