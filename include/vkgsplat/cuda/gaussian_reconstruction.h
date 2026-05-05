// SPDX-License-Identifier: Apache-2.0
//
// CUDA ingestion for nvdiffrast-style raster outputs. This is the first
// handoff point from a triangle/G-buffer render into vkGSplat's persistent
// Gaussian reconstruction state.
#pragma once

#include "../types.h"

#include <cstdint>

namespace vkgsplat::cuda {

// nvdiffrast rasterize() returns a contiguous tensor shaped
// [batch, height, width, 4]. The channels are:
//   x,y: perspective-correct barycentrics (u, v)
//   z  : z/w depth
//   w  : triangle_id + 1, with 0 meaning background
//
// This launch extracts hit pixels into a compact device-side sample stream.
struct NvDiffrastRasterLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t batch = 1;
    std::uint32_t max_samples = 0;
    float min_confidence = 0.0f;
};

struct NvDiffrastRasterInputs {
    const float4* raster = nullptr;
    const float4* color_rgba = nullptr;
    const float3* world_position = nullptr;
    const float3* normal = nullptr;
    // Pixel motion for temporal reconstruction. Convention: previous
    // pixel coordinate minus current pixel coordinate.
    const float2* motion_px = nullptr;
};

enum GaussianReconstructionSampleFlags : std::uint32_t {
    GAUSSIAN_SAMPLE_HAS_COLOR = 1u << 0,
    GAUSSIAN_SAMPLE_HAS_WORLD_POSITION = 1u << 1,
    GAUSSIAN_SAMPLE_HAS_NORMAL = 1u << 2,
    GAUSSIAN_SAMPLE_HAS_MOTION = 1u << 3,
};

struct GaussianReconstructionSampleIds {
    std::uint32_t primitive_id = 0;
    std::uint32_t pixel_index = 0;
    std::uint32_t batch_index = 0;
    std::uint32_t flags = 0;
};

struct GaussianReconstructionSample {
    float3 position;
    float3 normal;
    float4 radiance;
    float2 pixel;
    float2 motion_px;
    float2 barycentric_uv;
    float depth_ndc = 0.0f;
    float confidence = 0.0f;
    std::uint32_t primitive_id = 0;
    std::uint32_t pixel_index = 0;
    std::uint32_t batch_index = 0;
    std::uint32_t flags = 0;
};

// Tensorized SoA output view. Each field points at contiguous CUDA memory,
// typically a torch.Tensor / nvdiffrast-compatible tensor allocation:
//   position, normal        [max_samples, 3] float32
//   radiance                [max_samples, 4] float32
//   pixel, motion, bary     [max_samples, 2] float32
//   depth_confidence        [max_samples, 2] float32 = (depth_ndc, confidence)
//   ids                     [max_samples, 4] uint32 = (primitive, pixel, batch, flags)
struct GaussianReconstructionTensorOutputs {
    float3* position = nullptr;
    float3* normal = nullptr;
    float4* radiance = nullptr;
    float2* pixel = nullptr;
    float2* motion_px = nullptr;
    float2* barycentric_uv = nullptr;
    float2* depth_confidence = nullptr;
    GaussianReconstructionSampleIds* ids = nullptr;
};

struct NvDiffrastExtractCounters {
    std::uint32_t visited = 0;
    std::uint32_t emitted = 0;
    std::uint32_t background = 0;
    std::uint32_t invalid = 0;
    std::uint32_t low_confidence = 0;
    std::uint32_t capacity_overflow = 0;
};

void launch_extract_nvdiffrast_samples(const NvDiffrastRasterLaunch& launch,
                                       const NvDiffrastRasterInputs& inputs,
                                       GaussianReconstructionSample* samples,
                                       std::uint32_t* sample_count,
                                       NvDiffrastExtractCounters* counters,
                                       void* cuda_stream);

void launch_extract_nvdiffrast_sample_tensors(const NvDiffrastRasterLaunch& launch,
                                              const NvDiffrastRasterInputs& inputs,
                                              const GaussianReconstructionTensorOutputs& outputs,
                                              std::uint32_t* sample_count,
                                              NvDiffrastExtractCounters* counters,
                                              void* cuda_stream);

} // namespace vkgsplat::cuda
