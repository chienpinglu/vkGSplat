// SPDX-License-Identifier: Apache-2.0
//
// Synthetic two-frame reprojection fixture. The current frame looks one
// pixel to the right in previous-frame space, then deliberately changes
// primitive/depth on two pixels to validate history rejection.

#include <vkgsplat/reprojection.h>

#include <cmath>
#include <cstdio>

namespace {

bool near(float a, float b, float eps = 1.0e-6f) {
    return std::abs(a - b) <= eps;
}

vkgsplat::float4 color(float r, float g, float b) {
    return { r, g, b, 1.0f };
}

vkgsplat::mat4 identity() {
    vkgsplat::mat4 m{};
    m.m[0] = 1.0f;
    m.m[5] = 1.0f;
    m.m[10] = 1.0f;
    m.m[15] = 1.0f;
    return m;
}

} // namespace

int main() {
    using namespace vkgsplat;

    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 2;
    constexpr std::size_t count = width * height;

    ReprojectionFrame previous;
    previous.width = width;
    previous.height = height;
    previous.color = {
        color(1, 0, 0), color(2, 0, 0), color(3, 0, 0), color(4, 0, 0),
        color(1, 1, 0), color(2, 1, 0), color(3, 1, 0), color(4, 1, 0),
    };
    previous.depth.assign(count, 1.0f);
    previous.primitive_id.assign(count, 7);

    ReprojectionFrame current;
    current.width = width;
    current.height = height;
    current.color.assign(count, color(0, 0, 1));
    current.depth.assign(count, 1.0f);
    current.primitive_id.assign(count, 7);

    std::vector<float> ndc_depth(count, 0.5f);
    mat4 current_inv_view_projection = identity();
    mat4 previous_view_projection = identity();
    previous_view_projection.m[12] = -2.0f / static_cast<float>(width);

    const CameraMotionMap motion_map = compute_camera_motion_map(
        width, height, ndc_depth, current_inv_view_projection, previous_view_projection);
    if (motion_map.current_to_previous_px.size() != count || motion_map.valid.size() != count) {
        std::fprintf(stderr, "camera motion map size mismatch\n");
        return 1;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (motion_map.valid[i] != 1 ||
            !near(motion_map.current_to_previous_px[i].x, -1.0f) ||
            !near(motion_map.current_to_previous_px[i].y, 0.0f)) {
            std::fprintf(stderr, "camera-derived motion map mismatch\n");
            return 1;
        }
    }

    // Primitive mismatch rejects current row 0, pixel 2.
    current.primitive_id[2] = 99;

    // Depth mismatch rejects current row 1, pixel 3.
    current.depth[7] = 2.0f;

    ReprojectionOptions options;
    options.history_weight = 1.0f;
    options.depth_threshold = 0.01f;

    const ReprojectionResult result = reproject_history(
        previous, current, motion_map.current_to_previous_px, options);

    if (result.width != width || result.height != height) {
        std::fprintf(stderr, "result dimensions mismatch\n");
        return 1;
    }
    if (result.valid_history.size() != count || result.color.size() != count) {
        std::fprintf(stderr, "result sizes mismatch\n");
        return 1;
    }

    if (result.valid_history[0] != 0 || !near(result.color[0].z, 1.0f)) {
        std::fprintf(stderr, "left-edge disocclusion was not rejected\n");
        return 1;
    }
    if (result.valid_history[1] != 1 || !near(result.color[1].x, 1.0f)) {
        std::fprintf(stderr, "stable pixel did not reuse previous history\n");
        return 1;
    }
    if (result.valid_history[2] != 0 || !near(result.color[2].z, 1.0f)) {
        std::fprintf(stderr, "primitive mismatch was not rejected\n");
        return 1;
    }
    if (result.valid_history[5] != 1 || !near(result.color[5].x, 1.0f) || !near(result.color[5].y, 1.0f)) {
        std::fprintf(stderr, "second-row stable pixel reprojected incorrectly\n");
        return 1;
    }
    if (result.valid_history[7] != 0 || !near(result.color[7].z, 1.0f)) {
        std::fprintf(stderr, "depth mismatch was not rejected\n");
        return 1;
    }

    return 0;
}
