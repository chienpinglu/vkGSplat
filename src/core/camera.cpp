// SPDX-License-Identifier: Apache-2.0
//
// Pinhole camera with the conventions used by the 3DGS reference
// implementation. View matrices are world-from-camera (the Kerbl 2023
// code refers to this as "world_view_transform"); the projection is
// reversed-Z friendly and y-down, matching Vulkan's clip space.

#include "vkgsplat/camera.h"

#include <cmath>

namespace vkgsplat {

namespace {

constexpr float deg_to_rad(float d) { return d * 3.14159265358979323846f / 180.0f; }

float3 sub(float3 a, float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
float3 cross(float3 a, float3 b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 normalize(float3 v) {
    const float n = std::sqrt(dot(v, v));
    return n > 0.0f ? float3{ v.x / n, v.y / n, v.z / n } : float3{ 0.0f, 0.0f, 0.0f };
}

void set_col(mat4& m, int c, float x, float y, float z, float w) {
    m.m[c * 4 + 0] = x;
    m.m[c * 4 + 1] = y;
    m.m[c * 4 + 2] = z;
    m.m[c * 4 + 3] = w;
}

} // namespace

void Camera::set_view(const mat4& view) { view_ = view; }

void Camera::look_at(float3 eye, float3 target, float3 up) {
    const float3 f = normalize(sub(target, eye));
    const float3 s = normalize(cross(f, up));
    const float3 u = cross(s, f);

    mat4 m{};
    set_col(m, 0,  s.x,  u.x, -f.x, 0.0f);
    set_col(m, 1,  s.y,  u.y, -f.y, 0.0f);
    set_col(m, 2,  s.z,  u.z, -f.z, 0.0f);
    set_col(m, 3, -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f);
    view_ = m;
}

void Camera::set_perspective(float fov_y, float aspect, float z_near, float z_far) {
    fov_y_ = fov_y;
    z_near_ = z_near;
    z_far_  = z_far;

    // Vulkan clip space: y points down, NDC z in [0, 1].
    // Reversed-Z would flip near/far in the depth slot; we keep the
    // standard mapping here and let the renderer choose to invert.
    const float f = 1.0f / std::tan(fov_y * 0.5f);
    const float nf_inv = 1.0f / (z_near - z_far);

    mat4 m{};
    set_col(m, 0, f / aspect, 0.0f, 0.0f, 0.0f);
    set_col(m, 1, 0.0f, -f, 0.0f, 0.0f); // y-down
    set_col(m, 2, 0.0f, 0.0f, z_far * nf_inv, -1.0f);
    set_col(m, 3, 0.0f, 0.0f, z_far * z_near * nf_inv, 0.0f);
    projection_ = m;
}

} // namespace vkgsplat
