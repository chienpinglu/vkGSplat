// SPDX-License-Identifier: Apache-2.0
//
// Camera: pinhole projection with the conventions used by 3D Gaussian
// Splatting reference code (Kerbl et al. 2023). View matrices are
// world-from-camera; the projection matrix follows Vulkan/GLSL
// conventions (clip-space y-down, NDC z in [0, 1]).
#pragma once

#include "types.h"

namespace vkgsplat {

class Camera {
public:
    Camera() = default;

    // Set extrinsics directly from a 4x4 world-from-camera matrix.
    void set_view(const mat4& view);

    // Convenience extrinsic constructor: position + look-at target.
    void look_at(float3 eye, float3 target, float3 up);

    // Perspective intrinsics; fov_y in radians; aspect = width / height.
    void set_perspective(float fov_y, float aspect, float z_near, float z_far);

    [[nodiscard]] const mat4& view() const noexcept { return view_; }
    [[nodiscard]] const mat4& projection() const noexcept { return projection_; }

    [[nodiscard]] std::uint32_t width()  const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }
    void set_resolution(std::uint32_t w, std::uint32_t h) { width_ = w; height_ = h; }

    [[nodiscard]] float fov_y() const noexcept { return fov_y_; }

private:
    mat4 view_{};
    mat4 projection_{};
    std::uint32_t width_  = 0;
    std::uint32_t height_ = 0;
    float fov_y_  = 0.0f;
    float z_near_ = 0.0f;
    float z_far_  = 0.0f;
};

} // namespace vkgsplat
