// SPDX-License-Identifier: Apache-2.0

#include "vksplat/denoise.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vksplat {
namespace {

std::size_t pixel_count(std::uint32_t width, std::uint32_t height) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

void validate_frame(const DenoiseFrame& frame) {
    const std::size_t expected = pixel_count(frame.width, frame.height);
    if (frame.color.size() != expected ||
        frame.depth.size() != expected ||
        frame.primitive_id.size() != expected) {
        throw std::runtime_error("invalid denoise frame sizes");
    }
}

void validate_history(const ReprojectionResult& history, std::uint32_t width, std::uint32_t height) {
    const std::size_t expected = pixel_count(width, height);
    if (history.width != width ||
        history.height != height ||
        history.color.size() != expected ||
        history.valid_history.size() != expected) {
        throw std::runtime_error("invalid denoise history sizes");
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

float luminance(float4 c) {
    return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
}

bool compatible_neighbor(
    const DenoiseFrame& frame,
    std::size_t center,
    std::size_t neighbor,
    const SvgfDenoiseOptions& options)
{
    if (!std::isfinite(frame.depth[center]) || !std::isfinite(frame.depth[neighbor])) {
        return false;
    }
    if (std::abs(frame.depth[center] - frame.depth[neighbor]) > options.depth_threshold) {
        return false;
    }
    if (options.reject_primitive_mismatch &&
        frame.primitive_id[center] != frame.primitive_id[neighbor]) {
        return false;
    }
    return true;
}

} // namespace

SvgfDenoiseResult denoise_svgf_baseline(
    const DenoiseFrame& current,
    const ReprojectionResult& reprojected_history,
    const SvgfDenoiseOptions& options)
{
    validate_frame(current);
    validate_history(reprojected_history, current.width, current.height);

    const std::size_t count = pixel_count(current.width, current.height);
    const float history_weight = std::clamp(options.history_weight, 0.0f, 1.0f);

    SvgfDenoiseResult result;
    result.width = current.width;
    result.height = current.height;
    result.color.resize(count);
    result.variance.assign(count, 0.0f);
    result.history_used.assign(count, 0);

    std::vector<float4> temporal(count);
    std::vector<float> variance(count, 0.0f);
    for (std::size_t i = 0; i < count; ++i) {
        if (reprojected_history.valid_history[i] != 0) {
            temporal[i] = lerp(current.color[i], reprojected_history.color[i], history_weight);
            const float dl = luminance(current.color[i]) - luminance(reprojected_history.color[i]);
            variance[i] = dl * dl;
            result.history_used[i] = 1;
        } else {
            temporal[i] = current.color[i];
            variance[i] = 0.0f;
        }
    }

    const int radius = static_cast<int>(options.spatial_radius);
    for (std::uint32_t y = 0; y < current.height; ++y) {
        for (std::uint32_t x = 0; x < current.width; ++x) {
            const std::size_t center = static_cast<std::size_t>(y) * current.width + x;

            float4 sum{ 0.0f, 0.0f, 0.0f, 0.0f };
            float variance_sum = 0.0f;
            float weight_sum = 0.0f;

            for (int dy = -radius; dy <= radius; ++dy) {
                const int ny = static_cast<int>(y) + dy;
                if (ny < 0 || ny >= static_cast<int>(current.height)) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int nx = static_cast<int>(x) + dx;
                    if (nx < 0 || nx >= static_cast<int>(current.width)) {
                        continue;
                    }

                    const std::size_t neighbor =
                        static_cast<std::size_t>(ny) * current.width + static_cast<std::size_t>(nx);
                    if (!compatible_neighbor(current, center, neighbor, options)) {
                        continue;
                    }

                    const float spatial_distance = static_cast<float>(dx * dx + dy * dy);
                    const float variance_weight = 1.0f / (1.0f + variance[neighbor]);
                    const float weight = variance_weight / (1.0f + spatial_distance);

                    sum.x += temporal[neighbor].x * weight;
                    sum.y += temporal[neighbor].y * weight;
                    sum.z += temporal[neighbor].z * weight;
                    sum.w += temporal[neighbor].w * weight;
                    variance_sum += variance[neighbor] * weight;
                    weight_sum += weight;
                }
            }

            if (weight_sum > 0.0f) {
                const float inv_weight = 1.0f / weight_sum;
                result.color[center] = {
                    sum.x * inv_weight,
                    sum.y * inv_weight,
                    sum.z * inv_weight,
                    sum.w * inv_weight,
                };
                result.variance[center] = variance_sum * inv_weight;
            } else {
                result.color[center] = temporal[center];
                result.variance[center] = variance[center];
            }
        }
    }

    return result;
}

} // namespace vksplat
