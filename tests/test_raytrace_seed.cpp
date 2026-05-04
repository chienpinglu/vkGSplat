// SPDX-License-Identifier: Apache-2.0
//
// M6 fixture: generate a low-sample Vulkan-ray-tracing-shaped seed frame,
// then feed it through M4 reprojection and M5 denoising.

#include <vkgsplat/raytrace_seed.h>

#include <cmath>
#include <cstdio>

namespace {

bool near(float a, float b, float eps = 1.0e-4f) {
    return std::abs(a - b) <= eps;
}

float luminance(vkgsplat::float4 c) {
    return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
}

} // namespace

int main() {
    using namespace vkgsplat;

    RayTracingScene scene;
    scene.materials.push_back({ { 0.9f, 0.25f, 0.12f, 1.0f } });
    scene.triangles.push_back({
        { -1.0f, -1.0f, 3.0f },
        {  1.0f, -1.0f, 3.0f },
        {  1.0f,  1.0f, 3.0f },
        0,
        42,
    });
    scene.triangles.push_back({
        { -1.0f, -1.0f, 3.0f },
        {  1.0f,  1.0f, 3.0f },
        { -1.0f,  1.0f, 3.0f },
        0,
        42,
    });

    RayTracingCamera camera;
    camera.eye = { 0.0f, 0.0f, 0.0f };
    camera.target = { 0.0f, 0.0f, 1.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.z_near = 0.1f;
    camera.z_far = 16.0f;

    RayTracingDispatch dispatch;
    dispatch.width = 8;
    dispatch.height = 8;
    dispatch.samples_per_pixel = 1;
    dispatch.radiance_noise = 0.35f;

    dispatch.seed = 123u;
    const RayTracingSeedFrame previous_seed = trace_raytracing_seed(scene, camera, dispatch);
    dispatch.seed = 456u;
    const RayTracingSeedFrame current_seed = trace_raytracing_seed(scene, camera, dispatch);

    const std::size_t count =
        static_cast<std::size_t>(dispatch.width) * static_cast<std::size_t>(dispatch.height);
    if (current_seed.color.size() != count || current_seed.depth.size() != count ||
        current_seed.ndc_depth.size() != count || current_seed.primitive_id.size() != count) {
        std::fprintf(stderr, "ray tracing seed sizes mismatch\n");
        return 1;
    }
    if (!current_seed.api_shape.top_level_acceleration_structure ||
        !current_seed.api_shape.ray_generation_shader ||
        !current_seed.api_shape.miss_shader ||
        !current_seed.api_shape.closest_hit_shader) {
        std::fprintf(stderr, "ray tracing API shape was not preserved\n");
        return 1;
    }

    std::size_t stable_hit = count;
    std::size_t miss = count;
    for (std::size_t i = 0; i < count; ++i) {
        const bool previous_hit = previous_seed.primitive_id[i] != invalid_raytrace_primitive_id;
        const bool current_hit = current_seed.primitive_id[i] != invalid_raytrace_primitive_id;
        if (stable_hit == count && previous_hit && current_hit &&
            previous_seed.primitive_id[i] == current_seed.primitive_id[i] &&
            std::isfinite(previous_seed.depth[i]) &&
            std::isfinite(current_seed.depth[i]) &&
            near(previous_seed.depth[i], current_seed.depth[i], 0.01f) &&
            std::abs(luminance(previous_seed.color[i]) - luminance(current_seed.color[i])) > 1.0e-4f) {
            stable_hit = i;
        }
        if (miss == count && !current_hit) {
            miss = i;
        }
    }
    if (stable_hit == count || miss == count) {
        std::fprintf(stderr, "ray tracing fixture did not produce expected hit/miss pixels\n");
        return 1;
    }

    const CameraMotionMap motion_map = compute_camera_motion_map(
        current_seed.width,
        current_seed.height,
        current_seed.ndc_depth,
        current_seed.inverse_view_projection,
        previous_seed.view_projection);
    if (motion_map.valid[stable_hit] != 1 ||
        !near(motion_map.current_to_previous_px[stable_hit].x, 0.0f, 1.0e-3f) ||
        !near(motion_map.current_to_previous_px[stable_hit].y, 0.0f, 1.0e-3f) ||
        motion_map.valid[miss] != 0) {
        std::fprintf(stderr, "camera-derived ray tracing motion map mismatch\n");
        return 1;
    }

    ReprojectionOptions reprojection_options;
    reprojection_options.history_weight = 1.0f;
    reprojection_options.depth_threshold = 0.05f;

    const ReprojectionResult reprojected = reproject_history(
        as_reprojection_frame(previous_seed),
        as_reprojection_frame(current_seed),
        motion_map.current_to_previous_px,
        reprojection_options);

    if (reprojected.valid_history[stable_hit] != 1 ||
        !near(reprojected.color[stable_hit].x, previous_seed.color[stable_hit].x, 1.0e-4f)) {
        std::fprintf(stderr, "stable ray traced hit did not reproject previous history\n");
        return 1;
    }
    if (reprojected.valid_history[miss] != 0) {
        std::fprintf(stderr, "ray tracing miss incorrectly reused history\n");
        return 1;
    }

    SvgfDenoiseOptions denoise_options;
    denoise_options.history_weight = 0.5f;
    denoise_options.depth_threshold = 0.05f;
    denoise_options.spatial_radius = 1;

    const SvgfDenoiseResult denoised =
        denoise_svgf_baseline(as_denoise_frame(current_seed), reprojected, denoise_options);

    if (denoised.history_used[stable_hit] != 1 || !(denoised.variance[stable_hit] > 0.0f)) {
        std::fprintf(stderr, "ray tracing seed did not feed denoise history/variance\n");
        return 1;
    }
    if (!std::isfinite(denoised.color[stable_hit].x) ||
        !std::isfinite(denoised.color[stable_hit].y) ||
        !std::isfinite(denoised.color[stable_hit].z)) {
        std::fprintf(stderr, "denoised ray tracing seed produced non-finite color\n");
        return 1;
    }

    return 0;
}
