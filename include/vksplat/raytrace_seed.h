// SPDX-License-Identifier: Apache-2.0
//
// CPU reference seed path for Vulkan-ray-tracing-shaped tests. This is
// not a production tracer; it preserves the data contract that a
// VK_KHR_ray_tracing_pipeline capture must provide before the CUDA
// backend exists locally.
#pragma once

#include "denoise.h"
#include "reprojection.h"
#include "types.h"

#include <cstdint>
#include <vector>

namespace vksplat {

constexpr std::uint32_t invalid_raytrace_primitive_id = 0xffffffffu;

struct RayTracingApiShape {
    bool top_level_acceleration_structure = true;
    bool ray_generation_shader = true;
    bool miss_shader = true;
    bool closest_hit_shader = true;
};

struct RayTracingMaterial {
    float4 base_color{ 1.0f, 1.0f, 1.0f, 1.0f };
};

struct RayTracingTriangle {
    float3 v0{};
    float3 v1{};
    float3 v2{};
    std::uint32_t material_index = 0;
    std::uint32_t primitive_id = 0;
};

struct RayTracingScene {
    std::vector<RayTracingMaterial> materials;
    std::vector<RayTracingTriangle> triangles;
};

struct RayTracingCamera {
    float3 eye{ 0.0f, 0.0f, 0.0f };
    float3 target{ 0.0f, 0.0f, 1.0f };
    float3 up{ 0.0f, 1.0f, 0.0f };
    float fov_y_radians = 1.0471975512f;
    float z_near = 0.01f;
    float z_far = 1000.0f;
};

struct RayTracingDispatch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t samples_per_pixel = 1;
    std::uint32_t seed = 1;
    float radiance_noise = 0.0f;
};

struct RayTracingSeedFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float4> color;
    std::vector<float> depth;
    std::vector<float> ndc_depth;
    std::vector<std::uint32_t> primitive_id;
    mat4 view_projection{};
    mat4 inverse_view_projection{};
    RayTracingApiShape api_shape{};
};

RayTracingSeedFrame trace_raytracing_seed(
    const RayTracingScene& scene,
    const RayTracingCamera& camera,
    const RayTracingDispatch& dispatch);

ReprojectionFrame as_reprojection_frame(const RayTracingSeedFrame& frame);
DenoiseFrame as_denoise_frame(const RayTracingSeedFrame& frame);

} // namespace vksplat
