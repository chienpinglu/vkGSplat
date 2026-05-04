// SPDX-License-Identifier: Apache-2.0
//
// CPU reference denoising contract for low-sample seed frames. The first
// implementation is an SVGF-style baseline: temporal accumulation from
// reprojected history followed by a small edge-aware spatial filter.
#pragma once

#include "reprojection.h"
#include "types.h"

#include <cstdint>
#include <vector>

namespace vksplat {

struct DenoiseFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float4> color;
    std::vector<float> depth;
    std::vector<std::uint32_t> primitive_id;
};

struct SvgfDenoiseOptions {
    float history_weight = 0.85f;
    float depth_threshold = 1.0e-2f;
    std::uint32_t spatial_radius = 1;
    bool reject_primitive_mismatch = true;
};

struct SvgfDenoiseResult {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float4> color;
    std::vector<float> variance;
    std::vector<std::uint8_t> history_used;
};

SvgfDenoiseResult denoise_svgf_baseline(
    const DenoiseFrame& current,
    const ReprojectionResult& reprojected_history,
    const SvgfDenoiseOptions& options = {});

} // namespace vksplat
