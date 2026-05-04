// SPDX-License-Identifier: Apache-2.0
//
// CPU reference temporal reprojection and history rejection. This is
// the portable contract that the Vulkan capture path and CUDA kernels
// should match.
#pragma once

#include "types.h"

#include <cstdint>
#include <vector>

namespace vksplat {

struct ReprojectionFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float4> color;
    std::vector<float> depth;
    std::vector<std::uint32_t> primitive_id;
    mat4 view_projection{};
};

struct ReprojectionOptions {
    float depth_threshold = 1.0e-3f;
    float history_weight = 0.9f;
    bool reject_primitive_mismatch = true;
};

struct ReprojectionResult {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float4> color;
    std::vector<std::uint8_t> valid_history;
};

struct CameraMotionMap {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float2> current_to_previous_px;
    std::vector<std::uint8_t> valid;
};

// current_ndc_depth stores current-frame depth in Vulkan NDC [0, 1].
// current_inv_view_projection unprojects current pixels to world space;
// previous_view_projection projects those world points to previous pixels.
CameraMotionMap compute_camera_motion_map(
    std::uint32_t width,
    std::uint32_t height,
    const std::vector<float>& current_ndc_depth,
    const mat4& current_inv_view_projection,
    const mat4& previous_view_projection);

// motion_current_to_previous_px is a screen-space map in pixels:
// previous_pixel = current_pixel + motion_current_to_previous_px.
ReprojectionResult reproject_history(
    const ReprojectionFrame& previous,
    const ReprojectionFrame& current,
    const std::vector<float2>& motion_current_to_previous_px,
    const ReprojectionOptions& options = {});

} // namespace vksplat
