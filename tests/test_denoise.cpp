// SPDX-License-Identifier: Apache-2.0
//
// SVGF-style baseline fixture: temporal accumulation pulls stable pixels
// toward reprojected history, while primitive/depth discontinuities keep
// unrelated bright pixels from bleeding across edges.

#include <vksplat/denoise.h>

#include <cmath>
#include <cstdio>

namespace {

bool near(float a, float b, float eps = 1.0e-4f) {
    return std::abs(a - b) <= eps;
}

vksplat::float4 color(float r, float g, float b) {
    return { r, g, b, 1.0f };
}

} // namespace

int main() {
    using namespace vksplat;

    constexpr std::uint32_t width = 3;
    constexpr std::uint32_t height = 3;
    constexpr std::size_t count = width * height;
    constexpr std::size_t center = 4;
    constexpr std::size_t edge = 5;

    DenoiseFrame current;
    current.width = width;
    current.height = height;
    current.color.assign(count, color(1.0f, 1.0f, 1.0f));
    current.depth.assign(count, 1.0f);
    current.primitive_id.assign(count, 7);

    current.color[center] = color(3.0f, 3.0f, 3.0f);
    current.color[edge] = color(10.0f, 0.0f, 0.0f);
    current.depth[edge] = 2.0f;
    current.primitive_id[edge] = 99;

    ReprojectionResult history;
    history.width = width;
    history.height = height;
    history.color.assign(count, color(1.0f, 1.0f, 1.0f));
    history.valid_history.assign(count, 1);
    history.valid_history[0] = 0;

    SvgfDenoiseOptions options;
    options.history_weight = 0.75f;
    options.depth_threshold = 0.01f;
    options.spatial_radius = 1;
    options.reject_primitive_mismatch = true;

    const SvgfDenoiseResult result = denoise_svgf_baseline(current, history, options);

    if (result.width != width || result.height != height ||
        result.color.size() != count || result.variance.size() != count ||
        result.history_used.size() != count) {
        std::fprintf(stderr, "denoise result sizes mismatch\n");
        return 1;
    }
    if (result.history_used[0] != 0 || result.history_used[center] != 1) {
        std::fprintf(stderr, "history usage flags mismatch\n");
        return 1;
    }
    if (!(result.color[center].x > 1.0f && result.color[center].x < current.color[center].x)) {
        std::fprintf(stderr, "center noisy pixel was not denoised: %.4f\n", result.color[center].x);
        return 1;
    }
    if (!(result.color[center].x < 1.25f)) {
        std::fprintf(stderr, "spatial pass did not pull center toward stable neighbors: %.4f\n",
                     result.color[center].x);
        return 1;
    }
    if (!near(result.color[edge].x, 3.25f) || !near(result.color[edge].y, 0.75f)) {
        std::fprintf(stderr, "edge pixel unexpectedly mixed across primitive/depth boundary: %.4f %.4f\n",
                     result.color[edge].x, result.color[edge].y);
        return 1;
    }
    if (!(result.variance[center] > 0.0f)) {
        std::fprintf(stderr, "center variance was not tracked\n");
        return 1;
    }

    return 0;
}
