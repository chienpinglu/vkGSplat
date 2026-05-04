// SPDX-License-Identifier: Apache-2.0
//
// Common host-visible data types shared between the Vulkan front-end,
// the CUDA backend, and the interop layer. Kept POD where possible so
// the same struct can be referenced from .cu translation units.
#pragma once

#include <array>
#include <cstdint>

namespace vkgsplat {

// Minimal vector / matrix types so the public surface does not pull in
// glm or Eigen. Internal modules may use richer math libraries.
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct mat4 {
    // Column-major, Vulkan/GLSL convention.
    std::array<float, 16> m{};
};

// A single 3D Gaussian primitive in the scene. This is the unit of work
// for the v1 renderer (Section 4 of the position paper).
//
// Scaling is stored as log-space (kerbl2023 convention) so that
// optimization-time gradients remain well-conditioned; rendering
// converts to linear space.
struct Gaussian {
    float3 position;          // world-space mean
    float3 scale_log;         // log-scale per axis
    float4 rotation;          // unit quaternion (xyzw)
    float  opacity_logit;     // logit-space opacity in [-inf, +inf]
    // Spherical harmonics, RGB. Degree fixed at compile time for v1.
    static constexpr int sh_degree = 3;
    static constexpr int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
    std::array<float3, sh_coeffs> sh{};
};

// Image surface descriptor used for headless render targets and for
// images that cross the CUDA <-> Vulkan boundary via external memory.
enum class PixelFormat : std::uint32_t {
    UNDEFINED = 0,
    R8G8B8A8_UNORM,
    R8G8B8A8_SRGB,
    R16G16B16A16_SFLOAT,
    R32G32B32A32_SFLOAT,
};

struct ImageDesc {
    std::uint32_t width  = 0;
    std::uint32_t height = 0;
    PixelFormat   format = PixelFormat::R8G8B8A8_UNORM;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
};

// Frame index used to drive timeline semaphores on both APIs.
using FrameId = std::uint64_t;

} // namespace vkgsplat
