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

// Generic dense seed-buffer ingestion for native Vulkan/Wicked or ray-tracing
// captures. Unlike nvdiffrast ingestion, primitive identity and depth/confidence
// are already explicit guide tensors. Valid pixels are compacted into the same
// SoA sample tensors consumed by the Gaussian update and tile resolve kernels.
struct GaussianSeedFrameTensorLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t batch = 1;
    std::uint32_t max_samples = 0;
    float min_confidence = 0.0f;
};

struct GaussianSeedFrameTensorInputs {
    const float4* radiance = nullptr; // [batch * height * width]
    const float3* world_position = nullptr; // optional [batch * height * width]
    const float3* normal = nullptr; // optional [batch * height * width]
    const float2* motion_px = nullptr; // optional [batch * height * width]
    const float2* depth_confidence = nullptr; // [batch * height * width]
    const GaussianReconstructionSampleIds* ids = nullptr; // [batch * height * width]
};

struct GaussianSeedFrameExtractCounters {
    std::uint32_t visited = 0;
    std::uint32_t emitted = 0;
    std::uint32_t background = 0;
    std::uint32_t invalid = 0;
    std::uint32_t low_confidence = 0;
    std::uint32_t capacity_overflow = 0;
};

void launch_extract_seed_frame_sample_tensors(const GaussianSeedFrameTensorLaunch& launch,
                                              const GaussianSeedFrameTensorInputs& inputs,
                                              const GaussianReconstructionTensorOutputs& outputs,
                                              std::uint32_t* sample_count,
                                              GaussianSeedFrameExtractCounters* counters,
                                              void* cuda_stream);

struct GaussianSampleCountInfoLaunch {
    std::uint32_t max_samples = 0;
};

struct GaussianSampleCountInfo {
    std::uint32_t raw = 0;
    std::uint32_t clamped = 0;
    std::uint32_t overflow = 0;
    std::uint32_t available = 0;
};

void launch_read_gaussian_sample_count_info(const GaussianSampleCountInfoLaunch& launch,
                                            const std::uint32_t* sample_count,
                                            GaussianSampleCountInfo* output,
                                            void* cuda_stream);


// Const tensor sample view consumed by update kernels. This mirrors
// GaussianReconstructionTensorOutputs, but marks the extraction result as input.
struct GaussianReconstructionTensorSamples {
    const float3* position = nullptr;
    const float3* normal = nullptr;
    const float4* radiance = nullptr;
    const float2* pixel = nullptr;
    const float2* motion_px = nullptr;
    const float2* barycentric_uv = nullptr;
    const float2* depth_confidence = nullptr;
    const GaussianReconstructionSampleIds* ids = nullptr;
};

// Minimal persistent Gaussian state for M0. Fields are SoA so that later
// versions can be backed directly by torch.Tensor or exported CUDA memory.
// Before finalize, floating fields are accumulated confidence-weighted sums;
// after finalize, they are normalized means. mass_variance.x stores mass and
// mass_variance.y stores luma variance after finalize.
struct GaussianReconstructionState {
    float3* position = nullptr;
    float3* normal = nullptr;
    float4* radiance = nullptr;
    float2* pixel = nullptr;
    float2* motion_px = nullptr;
    float3* covariance_diag = nullptr;
    float2* depth_confidence = nullptr;
    float2* mass_variance = nullptr;
    GaussianReconstructionSampleIds* ids = nullptr;
};

struct GaussianStateUpdateLaunch {
    std::uint32_t sample_count = 0;
    std::uint32_t gaussian_capacity = 0;
    float min_confidence = 0.0f;
};

// Tensor-count update keeps the compacted sample count on the GPU. max_samples
// is the allocated tensor length; sample_count is a device-side uint32 written
// by launch_extract_nvdiffrast_sample_tensors.
struct GaussianStateTensorCountUpdateLaunch {
    std::uint32_t max_samples = 0;
    std::uint32_t gaussian_capacity = 0;
    float min_confidence = 0.0f;
};

struct GaussianStateUpdateCounters {
    std::uint32_t visited = 0;
    std::uint32_t updated = 0;
    std::uint32_t skipped_no_position = 0;
    std::uint32_t low_confidence = 0;
    std::uint32_t slot_overflow = 0;
    std::uint32_t invalid = 0;
};

struct GaussianFeatureProjectionLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t gaussian_count = 0;
    float min_confidence = 0.0f;
    // 4DGS-lite temporal projection: 0 projects current pixel, 1 projects by
    // the stored previous-minus-current motion vector. Intermediate values
    // support midpoint/frame-generation probes.
    float motion_alpha = 0.0f;
};

struct GaussianFeatureProjectionOutputs {
    float4* radiance = nullptr;
    float2* depth_confidence = nullptr;
    GaussianReconstructionSampleIds* ids = nullptr;
};

struct GaussianFeatureProjectionCounters {
    std::uint32_t visited = 0;
    std::uint32_t emitted = 0;
    std::uint32_t skipped_empty = 0;
    std::uint32_t out_of_bounds = 0;
};

// Weighted projection for the persistent Gaussian state. This is the
// state-space counterpart of the sample-tile weighted resolver: finalized
// Gaussians splat a small image-space footprint into tensor feature planes.
// It is intentionally tile-free for M0 so the PyTorch path can validate
// state -> frame reconstruction before we add persistent-state binning.
struct GaussianStateWeightedProjectionLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t gaussian_count = 0;
    std::uint32_t radius_px = 1;
    float sigma_px = 0.75f;
    float min_confidence = 0.0f;
    float motion_alpha = 0.0f;
};

struct GaussianStateWeightedProjectionCounters {
    std::uint32_t visited = 0;
    std::uint32_t emitted_taps = 0;
    std::uint32_t skipped_empty = 0;
    std::uint32_t out_of_bounds = 0;
    std::uint32_t invalid = 0;
    std::uint32_t normalized_pixels = 0;
    std::uint32_t zero_weight_pixels = 0;
};

// Tensorized tile-binning pass for compact sample streams. This does not sort
// samples yet; it produces a per-sample tile id and per-tile counts so later
// kernels can prefix-sum, compact, or launch tile-local reconstruction work.
struct GaussianSampleTileBinningLaunch {
    std::uint32_t max_samples = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tile_width = 16;
    std::uint32_t tile_height = 16;
};

struct GaussianSampleTileBinningOutputs {
    // [max_samples] uint32. Invalid or unbinned samples are set to 0xffffffff.
    std::uint32_t* sample_tile = nullptr;
    // [ceil(width/tile_width) * ceil(height/tile_height)] uint32.
    std::uint32_t* tile_counts = nullptr;
};

struct GaussianSampleTileBinningCounters {
    std::uint32_t visited = 0;
    std::uint32_t binned = 0;
    std::uint32_t skipped_empty = 0;
    std::uint32_t invalid = 0;
    std::uint32_t out_of_bounds = 0;
};

// Device-side exclusive prefix sum for tile counts. This closes the CUDA
// bin -> offsets -> compact path without a host readback or Python cumsum.
struct GaussianSampleTileOffsetLaunch {
    std::uint32_t tile_count = 0;
};

struct GaussianSampleTileOffsetInputs {
    const std::uint32_t* tile_counts = nullptr; // [tile_count]
};

struct GaussianSampleTileOffsetOutputs {
    std::uint32_t* tile_offsets = nullptr; // [tile_count + 1]
};

struct GaussianSampleTileOffsetCounters {
    std::uint32_t tile_count = 0;
    std::uint32_t total_samples = 0;
};

// Second tile pass. Given per-sample tile ids and an exclusive prefix sum of
// tile_counts, scatter compact sample indices into tile-local contiguous spans:
//   tile_sample_indices[tile_offsets[t] ... tile_offsets[t + 1])
// tile_offsets must have tile_count + 1 entries.
struct GaussianSampleTileCompactionLaunch {
    std::uint32_t max_samples = 0;
    std::uint32_t tile_count = 0;
};

struct GaussianSampleTileCompactionInputs {
    const std::uint32_t* sample_tile = nullptr; // [max_samples]
    const std::uint32_t* tile_offsets = nullptr; // [tile_count + 1]
};

struct GaussianSampleTileCompactionOutputs {
    // [tile_count] reset to zero, then used as per-tile atomic cursors.
    std::uint32_t* tile_write_counts = nullptr;
    // [max_samples] reset to 0xffffffff. Valid prefix length is tile_offsets[tile_count].
    std::uint32_t* tile_sample_indices = nullptr;
};

struct GaussianSampleTileCompactionCounters {
    std::uint32_t visited = 0;
    std::uint32_t compacted = 0;
    std::uint32_t invalid_tile = 0;
    std::uint32_t capacity_overflow = 0;
};

// M0 tile-local resolve for compact sample streams. This consumes the tile spans
// produced by bin/count/scan/compact and writes sample radiance plus guides back
// into image-space feature planes. Later kernels can replace the direct write
// with weighted splat accumulation or learned resolving.
struct GaussianTileSampleResolveLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tile_width = 16;
    std::uint32_t tile_height = 16;
    std::uint32_t tile_count = 0;
    std::uint32_t max_samples = 0;
    float min_confidence = 0.0f;
};

struct GaussianTileSampleResolveInputs {
    const std::uint32_t* tile_offsets = nullptr; // [tile_count + 1]
    const std::uint32_t* tile_sample_indices = nullptr; // [max_samples]
};

struct GaussianTileSampleResolveOutputs {
    float4* radiance = nullptr; // [height * width]
    float2* depth_confidence = nullptr; // [height * width]
    GaussianReconstructionSampleIds* ids = nullptr; // [height * width]
};

struct GaussianTileSampleResolveCounters {
    std::uint32_t visited_tiles = 0;
    std::uint32_t empty_tiles = 0;
    std::uint32_t visited_samples = 0;
    std::uint32_t emitted = 0;
    std::uint32_t invalid = 0;
    std::uint32_t out_of_bounds = 0;
    std::uint32_t low_confidence = 0;
};

// Weighted tile resolve. This is the first reconstruction-shaped tile kernel:
// each compact sample contributes a small Gaussian footprint into image-space
// accumulators, then a normalize pass produces radiance/depth/confidence planes.
struct GaussianTileWeightedResolveLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tile_width = 16;
    std::uint32_t tile_height = 16;
    std::uint32_t tile_count = 0;
    std::uint32_t max_samples = 0;
    std::uint32_t radius_px = 1;
    float sigma_px = 0.75f;
    float min_confidence = 0.0f;
};

struct GaussianTileWeightedResolveCounters {
    std::uint32_t visited_tiles = 0;
    std::uint32_t empty_tiles = 0;
    std::uint32_t visited_samples = 0;
    std::uint32_t emitted_taps = 0;
    std::uint32_t invalid = 0;
    std::uint32_t out_of_bounds = 0;
    std::uint32_t low_confidence = 0;
    std::uint32_t normalized_pixels = 0;
    std::uint32_t zero_weight_pixels = 0;
};

enum GaussianTileResolveGateFlags : std::uint32_t {
    GAUSSIAN_TILE_GATE_PRIMITIVE_ID = 1u << 0,
    GAUSSIAN_TILE_GATE_DEPTH = 1u << 1,
    GAUSSIAN_TILE_GATE_NORMAL = 1u << 2,
    GAUSSIAN_TILE_GATE_MOTION = 1u << 3,
};

struct GaussianTileGatedWeightedResolveLaunch {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tile_width = 16;
    std::uint32_t tile_height = 16;
    std::uint32_t tile_count = 0;
    std::uint32_t max_samples = 0;
    std::uint32_t radius_px = 1;
    std::uint32_t gate_flags = GAUSSIAN_TILE_GATE_PRIMITIVE_ID | GAUSSIAN_TILE_GATE_DEPTH;
    float sigma_px = 0.75f;
    float min_confidence = 0.0f;
    float depth_epsilon = 1.0e-3f;
    float normal_dot_min = 0.5f;
    float motion_epsilon_px = 1.0f;
};

struct GaussianTileResolveGuides {
    const float2* depth_confidence = nullptr; // [height * width]
    const GaussianReconstructionSampleIds* ids = nullptr; // [height * width]
    const float3* normal = nullptr; // optional [height * width]
    const float2* motion_px = nullptr; // optional [height * width]
};

struct GaussianTileGatedWeightedResolveCounters {
    std::uint32_t visited_tiles = 0;
    std::uint32_t empty_tiles = 0;
    std::uint32_t visited_samples = 0;
    std::uint32_t emitted_taps = 0;
    std::uint32_t invalid = 0;
    std::uint32_t out_of_bounds = 0;
    std::uint32_t low_confidence = 0;
    std::uint32_t normalized_pixels = 0;
    std::uint32_t zero_weight_pixels = 0;
    std::uint32_t rejected_primitive = 0;
    std::uint32_t rejected_depth = 0;
    std::uint32_t rejected_normal = 0;
    std::uint32_t rejected_motion = 0;
};

void launch_clear_gaussian_reconstruction_state(const GaussianReconstructionState& state,
                                                std::uint32_t gaussian_capacity,
                                                void* cuda_stream);

// M0 direct-slot update: sample.primitive_id selects the Gaussian slot. This is
// intentionally simple so we can validate extraction -> state -> projection
// before adding hash-grid assignment and merge/split maintenance.
void launch_accumulate_gaussian_state_from_samples(const GaussianStateUpdateLaunch& launch,
                                                   const GaussianReconstructionSample* samples,
                                                   const GaussianReconstructionState& state,
                                                   GaussianStateUpdateCounters* counters,
                                                   void* cuda_stream);

void launch_accumulate_gaussian_state_from_sample_tensors(const GaussianStateUpdateLaunch& launch,
                                                          const GaussianReconstructionTensorSamples& samples,
                                                          const GaussianReconstructionState& state,
                                                          GaussianStateUpdateCounters* counters,
                                                          void* cuda_stream);

void launch_accumulate_gaussian_state_from_sample_tensors_counted(
    const GaussianStateTensorCountUpdateLaunch& launch,
    const GaussianReconstructionTensorSamples& samples,
    const std::uint32_t* sample_count,
    const GaussianReconstructionState& state,
    GaussianStateUpdateCounters* counters,
    void* cuda_stream);

void launch_finalize_gaussian_reconstruction_state(const GaussianReconstructionState& state,
                                                   std::uint32_t gaussian_capacity,
                                                   void* cuda_stream);

// Projects the finalized M0 state back into pixel feature planes using the last
// observed pixel coordinate. Later versions replace this with camera/covariance
// projection, but this already validates the CUDA dataflow and state reuse.
void launch_project_gaussian_state_features(const GaussianFeatureProjectionLaunch& launch,
                                            const GaussianReconstructionState& state,
                                            const GaussianFeatureProjectionOutputs& outputs,
                                            GaussianFeatureProjectionCounters* counters,
                                            void* cuda_stream);

void launch_project_gaussian_state_features_weighted(
    const GaussianStateWeightedProjectionLaunch& launch,
    const GaussianReconstructionState& state,
    const GaussianFeatureProjectionOutputs& outputs,
    GaussianStateWeightedProjectionCounters* counters,
    void* cuda_stream);

void launch_build_gaussian_sample_tile_bins(const GaussianSampleTileBinningLaunch& launch,
                                            const GaussianReconstructionTensorSamples& samples,
                                            const std::uint32_t* sample_count,
                                            const GaussianSampleTileBinningOutputs& outputs,
                                            GaussianSampleTileBinningCounters* counters,
                                            void* cuda_stream);

void launch_build_gaussian_sample_tile_offsets(const GaussianSampleTileOffsetLaunch& launch,
                                               const GaussianSampleTileOffsetInputs& inputs,
                                               const GaussianSampleTileOffsetOutputs& outputs,
                                               GaussianSampleTileOffsetCounters* counters,
                                               void* cuda_stream);

void launch_compact_gaussian_sample_tile_bins(const GaussianSampleTileCompactionLaunch& launch,
                                              const std::uint32_t* sample_count,
                                              const GaussianSampleTileCompactionInputs& inputs,
                                              const GaussianSampleTileCompactionOutputs& outputs,
                                              GaussianSampleTileCompactionCounters* counters,
                                              void* cuda_stream);

void launch_resolve_gaussian_sample_tiles(const GaussianTileSampleResolveLaunch& launch,
                                          const GaussianReconstructionTensorSamples& samples,
                                          const GaussianTileSampleResolveInputs& inputs,
                                          const GaussianTileSampleResolveOutputs& outputs,
                                          GaussianTileSampleResolveCounters* counters,
                                          void* cuda_stream);

void launch_resolve_gaussian_sample_tiles_weighted(
    const GaussianTileWeightedResolveLaunch& launch,
    const GaussianReconstructionTensorSamples& samples,
    const GaussianTileSampleResolveInputs& inputs,
    const GaussianTileSampleResolveOutputs& outputs,
    GaussianTileWeightedResolveCounters* counters,
    void* cuda_stream);

void launch_resolve_gaussian_sample_tiles_weighted_gated(
    const GaussianTileGatedWeightedResolveLaunch& launch,
    const GaussianReconstructionTensorSamples& samples,
    const GaussianTileSampleResolveInputs& inputs,
    const GaussianTileResolveGuides& guides,
    const GaussianTileSampleResolveOutputs& outputs,
    GaussianTileGatedWeightedResolveCounters* counters,
    void* cuda_stream);

} // namespace vkgsplat::cuda
