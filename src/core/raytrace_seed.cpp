// SPDX-License-Identifier: Apache-2.0

#include "vksplat/raytrace_seed.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vksplat {
namespace {

float3 add(float3 a, float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
float3 sub(float3 a, float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
float3 mul(float3 v, float s) { return { v.x * s, v.y * s, v.z * s }; }

float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 cross(float3 a, float3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float3 normalize(float3 v) {
    const float len2 = dot(v, v);
    if (len2 <= 0.0f) {
        return { 0.0f, 0.0f, 0.0f };
    }
    const float inv_len = 1.0f / std::sqrt(len2);
    return mul(v, inv_len);
}

mat4 multiply(const mat4& a, const mat4& b) {
    mat4 result{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[static_cast<std::size_t>(k) * 4 + row] *
                       b.m[static_cast<std::size_t>(col) * 4 + k];
            }
            result.m[static_cast<std::size_t>(col) * 4 + row] = sum;
        }
    }
    return result;
}

float4 multiply(const mat4& m, float4 v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w,
    };
}

mat4 inverse(const mat4& m) {
    float a[4][8]{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            a[row][col] = m.m[static_cast<std::size_t>(col) * 4 + row];
        }
        a[row][4 + row] = 1.0f;
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        float pivot_abs = std::abs(a[col][col]);
        for (int row = col + 1; row < 4; ++row) {
            const float candidate = std::abs(a[row][col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= 1.0e-8f) {
            throw std::runtime_error("ray tracing camera matrix is singular");
        }
        if (pivot != col) {
            for (int k = 0; k < 8; ++k) {
                std::swap(a[col][k], a[pivot][k]);
            }
        }

        const float inv_pivot = 1.0f / a[col][col];
        for (int k = 0; k < 8; ++k) {
            a[col][k] *= inv_pivot;
        }
        for (int row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const float f = a[row][col];
            for (int k = 0; k < 8; ++k) {
                a[row][k] -= f * a[col][k];
            }
        }
    }

    mat4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result.m[static_cast<std::size_t>(col) * 4 + row] = a[row][4 + col];
        }
    }
    return result;
}

void set_col(mat4& m, int c, float x, float y, float z, float w) {
    m.m[static_cast<std::size_t>(c) * 4 + 0] = x;
    m.m[static_cast<std::size_t>(c) * 4 + 1] = y;
    m.m[static_cast<std::size_t>(c) * 4 + 2] = z;
    m.m[static_cast<std::size_t>(c) * 4 + 3] = w;
}

mat4 make_view(const RayTracingCamera& camera) {
    const float3 f = normalize(sub(camera.target, camera.eye));
    const float3 s = normalize(cross(f, camera.up));
    const float3 u = cross(s, f);

    mat4 m{};
    set_col(m, 0,  s.x,  u.x, -f.x, 0.0f);
    set_col(m, 1,  s.y,  u.y, -f.y, 0.0f);
    set_col(m, 2,  s.z,  u.z, -f.z, 0.0f);
    set_col(m, 3, -dot(s, camera.eye), -dot(u, camera.eye), dot(f, camera.eye), 1.0f);
    return m;
}

mat4 make_projection(const RayTracingCamera& camera, float aspect) {
    const float f = 1.0f / std::tan(camera.fov_y_radians * 0.5f);
    const float nf_inv = 1.0f / (camera.z_near - camera.z_far);

    mat4 m{};
    set_col(m, 0, f / aspect, 0.0f, 0.0f, 0.0f);
    set_col(m, 1, 0.0f, -f, 0.0f, 0.0f);
    set_col(m, 2, 0.0f, 0.0f, camera.z_far * nf_inv, -1.0f);
    set_col(m, 3, 0.0f, 0.0f, camera.z_far * camera.z_near * nf_inv, 0.0f);
    return m;
}

float project_ndc_depth(float3 world, const mat4& view_projection) {
    const float4 world_h{ world.x, world.y, world.z, 1.0f };
    const float4 clip = multiply(view_projection, world_h);
    if (std::abs(clip.w) <= 1.0e-8f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return clip.z / clip.w;
}

std::uint32_t hash_u32(std::uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

float rand01(std::uint32_t& state) {
    state = hash_u32(state + 0x9e3779b9u);
    return static_cast<float>(state & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

struct Ray {
    float3 origin{};
    float3 direction{};
};

struct Hit {
    float t = std::numeric_limits<float>::infinity();
    float3 normal{};
    std::uint32_t primitive_id = invalid_raytrace_primitive_id;
    std::uint32_t material_index = 0;
};

bool intersect_triangle(const Ray& ray, const RayTracingTriangle& triangle, Hit& hit) {
    constexpr float eps = 1.0e-6f;
    const float3 e1 = sub(triangle.v1, triangle.v0);
    const float3 e2 = sub(triangle.v2, triangle.v0);
    const float3 p = cross(ray.direction, e2);
    const float det = dot(e1, p);
    if (std::abs(det) < eps) {
        return false;
    }

    const float inv_det = 1.0f / det;
    const float3 tvec = sub(ray.origin, triangle.v0);
    const float u = dot(tvec, p) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const float3 q = cross(tvec, e1);
    const float v = dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    const float t = dot(e2, q) * inv_det;
    if (t <= eps || t >= hit.t) {
        return false;
    }

    float3 normal = normalize(cross(e1, e2));
    if (dot(normal, ray.direction) > 0.0f) {
        normal = mul(normal, -1.0f);
    }

    hit.t = t;
    hit.normal = normal;
    hit.primitive_id = triangle.primitive_id;
    hit.material_index = triangle.material_index;
    return true;
}

Hit trace_closest_hit(const RayTracingScene& scene, const Ray& ray) {
    Hit hit;
    for (const RayTracingTriangle& triangle : scene.triangles) {
        (void)intersect_triangle(ray, triangle, hit);
    }
    return hit;
}

float4 shade_hit(const RayTracingScene& scene, const Hit& hit, float noise) {
    const RayTracingMaterial fallback{};
    const RayTracingMaterial& material =
        hit.material_index < scene.materials.size() ? scene.materials[hit.material_index] : fallback;

    const float3 light_dir = normalize({ -0.25f, 0.65f, -1.0f });
    const float diffuse = std::max(0.0f, dot(hit.normal, light_dir));
    const float exposure = std::max(0.0f, 0.25f + 0.75f * diffuse + noise);
    return {
        material.base_color.x * exposure,
        material.base_color.y * exposure,
        material.base_color.z * exposure,
        material.base_color.w,
    };
}

Ray make_camera_ray(
    const RayTracingCamera& camera,
    const RayTracingDispatch& dispatch,
    std::uint32_t x,
    std::uint32_t y,
    float jitter_x,
    float jitter_y)
{
    const float3 forward = normalize(sub(camera.target, camera.eye));
    const float3 right = normalize(cross(forward, camera.up));
    const float3 up = normalize(cross(right, forward));
    const float aspect = static_cast<float>(dispatch.width) / static_cast<float>(dispatch.height);
    const float tan_half_fov = std::tan(camera.fov_y_radians * 0.5f);

    const float px = (static_cast<float>(x) + jitter_x) / static_cast<float>(dispatch.width);
    const float py = (static_cast<float>(y) + jitter_y) / static_cast<float>(dispatch.height);
    const float sx = (px * 2.0f - 1.0f) * aspect * tan_half_fov;
    const float sy = (1.0f - py * 2.0f) * tan_half_fov;

    return {
        camera.eye,
        normalize(add(add(forward, mul(right, sx)), mul(up, sy))),
    };
}

} // namespace

RayTracingSeedFrame trace_raytracing_seed(
    const RayTracingScene& scene,
    const RayTracingCamera& camera,
    const RayTracingDispatch& dispatch)
{
    if (dispatch.width == 0 || dispatch.height == 0 || dispatch.samples_per_pixel == 0) {
        throw std::runtime_error("invalid ray tracing dispatch dimensions or sample count");
    }
    if (camera.z_far <= camera.z_near) {
        throw std::runtime_error("invalid ray tracing camera depth range");
    }

    const std::size_t count =
        static_cast<std::size_t>(dispatch.width) * static_cast<std::size_t>(dispatch.height);

    RayTracingSeedFrame frame;
    frame.width = dispatch.width;
    frame.height = dispatch.height;
    frame.color.assign(count, { 0.0f, 0.0f, 0.0f, 1.0f });
    frame.depth.assign(count, std::numeric_limits<float>::infinity());
    frame.ndc_depth.assign(count, std::numeric_limits<float>::quiet_NaN());
    frame.primitive_id.assign(count, invalid_raytrace_primitive_id);
    frame.view_projection = multiply(
        make_projection(camera, static_cast<float>(dispatch.width) / static_cast<float>(dispatch.height)),
        make_view(camera));
    frame.inverse_view_projection = inverse(frame.view_projection);

    for (std::uint32_t y = 0; y < dispatch.height; ++y) {
        for (std::uint32_t x = 0; x < dispatch.width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * dispatch.width + x;

            float4 radiance{ 0.0f, 0.0f, 0.0f, 1.0f };
            Hit closest;
            Ray closest_ray;
            for (std::uint32_t sample = 0; sample < dispatch.samples_per_pixel; ++sample) {
                std::uint32_t rng = hash_u32(dispatch.seed ^
                                             (x * 0x9e3779b9u) ^
                                             (y * 0x85ebca6bu) ^
                                             (sample * 0xc2b2ae35u));
                const float jitter_x = rand01(rng);
                const float jitter_y = rand01(rng);
                const Ray ray = make_camera_ray(camera, dispatch, x, y, jitter_x, jitter_y);
                const Hit hit = trace_closest_hit(scene, ray);
                if (hit.primitive_id == invalid_raytrace_primitive_id) {
                    continue;
                }

                const float noise = (rand01(rng) - 0.5f) * dispatch.radiance_noise;
                const float4 sample_color = shade_hit(scene, hit, noise);
                radiance.x += sample_color.x;
                radiance.y += sample_color.y;
                radiance.z += sample_color.z;

                if (hit.t < closest.t) {
                    closest = hit;
                    closest_ray = ray;
                }
            }

            const float inv_samples = 1.0f / static_cast<float>(dispatch.samples_per_pixel);
            frame.color[idx] = {
                radiance.x * inv_samples,
                radiance.y * inv_samples,
                radiance.z * inv_samples,
                1.0f,
            };

            if (closest.primitive_id != invalid_raytrace_primitive_id) {
                frame.depth[idx] = closest.t;
                const float3 hit_point =
                    add(closest_ray.origin, mul(closest_ray.direction, closest.t));
                frame.ndc_depth[idx] = project_ndc_depth(hit_point, frame.view_projection);
                frame.primitive_id[idx] = closest.primitive_id;
            }
        }
    }

    return frame;
}

ReprojectionFrame as_reprojection_frame(const RayTracingSeedFrame& frame) {
    ReprojectionFrame reprojection;
    reprojection.width = frame.width;
    reprojection.height = frame.height;
    reprojection.color = frame.color;
    reprojection.depth = frame.depth;
    reprojection.primitive_id = frame.primitive_id;
    reprojection.view_projection = frame.view_projection;
    return reprojection;
}

DenoiseFrame as_denoise_frame(const RayTracingSeedFrame& frame) {
    DenoiseFrame denoise;
    denoise.width = frame.width;
    denoise.height = frame.height;
    denoise.color = frame.color;
    denoise.depth = frame.depth;
    denoise.primitive_id = frame.primitive_id;
    return denoise;
}

} // namespace vksplat
