// SPDX-License-Identifier: Apache-2.0
//
// Native Apple Metal reconstruction backend. This is the local Mac GPU
// portability target for the CPU reference contracts before CUDA lowering.
#pragma once

#include "../denoise.h"

#include <string>

namespace vkgsplat::metal {

[[nodiscard]] bool is_available();
[[nodiscard]] std::string device_name();

[[nodiscard]] SvgfDenoiseResult denoise_svgf_baseline(
    const DenoiseFrame& current,
    const ReprojectionResult& reprojected_history,
    const SvgfDenoiseOptions& options = {});

} // namespace vkgsplat::metal
