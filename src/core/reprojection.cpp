// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/reprojection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace vkgsplat {
namespace {

std::size_t pixel_count(const ReprojectionFrame& frame) {
    return static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height);
}

void validate_frame(const ReprojectionFrame& frame, const char* name) {
    const std::size_t expected = pixel_count(frame);
    if (frame.color.size() != expected ||
        frame.depth.size() != expected ||
        frame.primitive_id.size() != expected) {
        throw std::runtime_error(std::string("invalid reprojection frame sizes: ") + name);
    }
}

float4 lerp(float4 a, float4 b, float t) {
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    };
}

float4 mul(const mat4& m, float4 v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w,
    };
}

float4 sample_color_bilinear(const ReprojectionFrame& frame, float x, float y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, static_cast<int>(frame.width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(frame.height) - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const auto at = [&](int px, int py) -> const float4& {
        return frame.color[static_cast<std::size_t>(py) * frame.width + static_cast<std::size_t>(px)];
    };

    const float4 a = lerp(at(x0, y0), at(x1, y0), tx);
    const float4 b = lerp(at(x0, y1), at(x1, y1), tx);
    return lerp(a, b, ty);
}

std::size_t nearest_index(const ReprojectionFrame& frame, float x, float y) {
    const int px = std::clamp(static_cast<int>(std::floor(x + 0.5f)), 0, static_cast<int>(frame.width) - 1);
    const int py = std::clamp(static_cast<int>(std::floor(y + 0.5f)), 0, static_cast<int>(frame.height) - 1);
    return static_cast<std::size_t>(py) * frame.width + static_cast<std::size_t>(px);
}

bool in_bounds(const ReprojectionFrame& frame, float x, float y) {
    return std::isfinite(x) &&
           std::isfinite(y) &&
           x >= 0.0f &&
           y >= 0.0f &&
           x <= static_cast<float>(frame.width - 1) &&
           y <= static_cast<float>(frame.height - 1);
}

} // namespace

CameraMotionMap compute_camera_motion_map(
    std::uint32_t width,
    std::uint32_t height,
    const std::vector<float>& current_ndc_depth,
    const mat4& current_inv_view_projection,
    const mat4& previous_view_projection)
{
    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (current_ndc_depth.size() != count) {
        throw std::runtime_error("current NDC depth size does not match motion-map dimensions");
    }

    CameraMotionMap result;
    result.width = width;
    result.height = height;
    result.current_to_previous_px.assign(count, {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
    });
    result.valid.assign(count, 0);

    if (width == 0 || height == 0) {
        return result;
    }

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            const float depth = current_ndc_depth[idx];
            if (!std::isfinite(depth)) {
                continue;
            }

            const float current_px = static_cast<float>(x);
            const float current_py = static_cast<float>(y);
            const float ndc_x = ((current_px + 0.5f) / static_cast<float>(width)) * 2.0f - 1.0f;
            const float ndc_y = ((current_py + 0.5f) / static_cast<float>(height)) * 2.0f - 1.0f;

            float4 world = mul(current_inv_view_projection, { ndc_x, ndc_y, depth, 1.0f });
            if (std::abs(world.w) <= 1.0e-8f) {
                continue;
            }
            const float inv_world_w = 1.0f / world.w;
            world = { world.x * inv_world_w, world.y * inv_world_w, world.z * inv_world_w, 1.0f };

            const float4 previous_clip = mul(previous_view_projection, world);
            if (std::abs(previous_clip.w) <= 1.0e-8f) {
                continue;
            }
            const float inv_previous_w = 1.0f / previous_clip.w;
            const float previous_ndc_x = previous_clip.x * inv_previous_w;
            const float previous_ndc_y = previous_clip.y * inv_previous_w;

            const float previous_px = (previous_ndc_x * 0.5f + 0.5f) * static_cast<float>(width) - 0.5f;
            const float previous_py = (previous_ndc_y * 0.5f + 0.5f) * static_cast<float>(height) - 0.5f;
            if (!std::isfinite(previous_px) || !std::isfinite(previous_py)) {
                continue;
            }

            result.current_to_previous_px[idx] = { previous_px - current_px, previous_py - current_py };
            result.valid[idx] = 1;
        }
    }

    return result;
}

ReprojectionResult reproject_history(
    const ReprojectionFrame& previous,
    const ReprojectionFrame& current,
    const std::vector<float2>& motion_current_to_previous_px,
    const ReprojectionOptions& options)
{
    if (previous.width != current.width || previous.height != current.height) {
        throw std::runtime_error("reprojection frames must have matching dimensions");
    }
    validate_frame(previous, "previous");
    validate_frame(current, "current");

    const std::size_t count = pixel_count(current);
    if (motion_current_to_previous_px.size() != count) {
        throw std::runtime_error("motion map size does not match reprojection frame dimensions");
    }

    ReprojectionResult result;
    result.width = current.width;
    result.height = current.height;
    result.color.resize(count);
    result.valid_history.assign(count, 0);

    const float history_weight = std::clamp(options.history_weight, 0.0f, 1.0f);

    for (std::uint32_t y = 0; y < current.height; ++y) {
        for (std::uint32_t x = 0; x < current.width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * current.width + x;
            const float2 motion = motion_current_to_previous_px[idx];
            const float prev_x = static_cast<float>(x) + motion.x;
            const float prev_y = static_cast<float>(y) + motion.y;

            bool valid = in_bounds(previous, prev_x, prev_y);
            float4 history = current.color[idx];

            if (valid) {
                const std::size_t nearest = nearest_index(previous, prev_x, prev_y);
                const float previous_depth = previous.depth[nearest];
                const float current_depth = current.depth[idx];
                valid = std::isfinite(previous_depth) &&
                        std::isfinite(current_depth) &&
                        std::abs(previous_depth - current_depth) <= options.depth_threshold;

                if (valid && options.reject_primitive_mismatch) {
                    valid = previous.primitive_id[nearest] == current.primitive_id[idx];
                }

                if (valid) {
                    history = sample_color_bilinear(previous, prev_x, prev_y);
                    result.valid_history[idx] = 1;
                }
            }

            result.color[idx] = valid ? lerp(current.color[idx], history, history_weight)
                                      : current.color[idx];
        }
    }

    return result;
}

} // namespace vkgsplat
