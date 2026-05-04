// SPDX-License-Identifier: Apache-2.0
//
// Native Apple GPU smoke test: run the SVGF-style denoise contract through
// Metal and compare against the CPU reference implementation.

#include <vksplat/denoise.h>
#include <vksplat/metal/denoise.h>

#include <cmath>
#include <cstdio>

namespace {

bool near(float a, float b, float eps = 2.0e-5f) {
    return std::abs(a - b) <= eps;
}

vksplat::float4 color(float r, float g, float b) {
    return { r, g, b, 1.0f };
}

} // namespace

int main() {
    using namespace vksplat;

    if (!metal::is_available()) {
        std::fprintf(stderr, "Metal device unavailable; skipping\n");
        return 77;
    }

    constexpr std::uint32_t width = 5;
    constexpr std::uint32_t height = 4;
    constexpr std::size_t count = width * height;

    DenoiseFrame current;
    current.width = width;
    current.height = height;
    current.color.assign(count, color(0.4f, 0.45f, 0.5f));
    current.depth.assign(count, 1.0f);
    current.primitive_id.assign(count, 3);

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * width + x;
            current.color[i] = color(
                0.2f + 0.08f * static_cast<float>(x),
                0.3f + 0.05f * static_cast<float>(y),
                0.5f + 0.02f * static_cast<float>(x + y));
        }
    }

    const std::size_t noisy = 2 + width;
    current.color[noisy] = color(1.2f, 0.2f, 0.1f);

    const std::size_t edge = 3 + width * 2;
    current.color[edge] = color(0.0f, 1.2f, 0.1f);
    current.depth[edge] = 2.0f;
    current.primitive_id[edge] = 99;

    ReprojectionResult history;
    history.width = width;
    history.height = height;
    history.color.assign(count, color(0.35f, 0.40f, 0.45f));
    history.valid_history.assign(count, 1);
    history.valid_history[0] = 0;
    history.valid_history[edge] = 1;

    SvgfDenoiseOptions options;
    options.history_weight = 0.65f;
    options.depth_threshold = 0.05f;
    options.spatial_radius = 1;
    options.reject_primitive_mismatch = true;

    const SvgfDenoiseResult cpu = denoise_svgf_baseline(current, history, options);
    const SvgfDenoiseResult gpu = metal::denoise_svgf_baseline(current, history, options);

    if (gpu.width != cpu.width || gpu.height != cpu.height ||
        gpu.color.size() != cpu.color.size() ||
        gpu.variance.size() != cpu.variance.size() ||
        gpu.history_used.size() != cpu.history_used.size()) {
        std::fprintf(stderr, "Metal denoise result size mismatch\n");
        return 1;
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (gpu.history_used[i] != cpu.history_used[i]) {
            std::fprintf(stderr, "history flag mismatch at %zu\n", i);
            return 1;
        }
        if (!near(gpu.color[i].x, cpu.color[i].x) ||
            !near(gpu.color[i].y, cpu.color[i].y) ||
            !near(gpu.color[i].z, cpu.color[i].z) ||
            !near(gpu.color[i].w, cpu.color[i].w) ||
            !near(gpu.variance[i], cpu.variance[i])) {
            std::fprintf(stderr,
                         "Metal/CPU denoise mismatch at %zu: gpu=(%.6f %.6f %.6f %.6f var %.6f) "
                         "cpu=(%.6f %.6f %.6f %.6f var %.6f)\n",
                         i,
                         gpu.color[i].x,
                         gpu.color[i].y,
                         gpu.color[i].z,
                         gpu.color[i].w,
                         gpu.variance[i],
                         cpu.color[i].x,
                         cpu.color[i].y,
                         cpu.color[i].z,
                         cpu.color[i].w,
                         cpu.variance[i]);
            return 1;
        }
    }

    return 0;
}
