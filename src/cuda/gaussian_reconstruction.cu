// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/cuda/gaussian_reconstruction.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace vkgsplat::cuda {
namespace {

__device__ bool finite_float(float v) {
    return isfinite(v);
}

__device__ bool finite_float2(float2 v) {
    return finite_float(v.x) && finite_float(v.y);
}

__device__ bool finite_float3(float3 v) {
    return finite_float(v.x) && finite_float(v.y) && finite_float(v.z);
}

__device__ bool finite_float4(float4 v) {
    return finite_float(v.x) && finite_float(v.y) && finite_float(v.z) && finite_float(v.w);
}

__device__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ void increment(std::uint32_t* counter) {
    if (counter) {
        atomicAdd(counter, 1u);
    }
}

__device__ bool reserve_sample(std::uint32_t* sample_count,
                               std::uint32_t max_samples,
                               std::uint32_t* out_slot) {
    if (!sample_count || !out_slot || max_samples == 0) {
        return false;
    }

    std::uint32_t old = *sample_count;
    while (old < max_samples) {
        const std::uint32_t assumed = old;
        old = atomicCAS(sample_count, assumed, assumed + 1u);
        if (old == assumed) {
            *out_slot = assumed;
            return true;
        }
    }
    return false;
}

__device__ std::uint32_t raster_triangle_id(float tri_plus_one) {
    if (!finite_float(tri_plus_one) || tri_plus_one < 0.5f) {
        return 0u;
    }
    return static_cast<std::uint32_t>(tri_plus_one);
}

__device__ bool build_sample(NvDiffrastRasterLaunch launch,
                             NvDiffrastRasterInputs inputs,
                             std::uint32_t i,
                             NvDiffrastExtractCounters* counters,
                             GaussianReconstructionSample* out_sample) {
    if (counters) {
        increment(&counters->visited);
    }

    const float4 raster = inputs.raster[i];
    if (!finite_float4(raster)) {
        if (counters) {
            increment(&counters->invalid);
        }
        return false;
    }

    const std::uint32_t tri_plus_one = raster_triangle_id(raster.w);
    if (tri_plus_one == 0u) {
        if (counters) {
            increment(&counters->background);
        }
        return false;
    }

    if (raster.z < -1.0001f || raster.z > 1.0001f) {
        if (counters) {
            increment(&counters->invalid);
        }
        return false;
    }

    const float4 color = inputs.color_rgba ? inputs.color_rgba[i] : float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    if (inputs.color_rgba && !finite_float4(color)) {
        if (counters) {
            increment(&counters->invalid);
        }
        return false;
    }

    const float confidence = inputs.color_rgba ? clamp01(color.w) : 1.0f;
    if (confidence < launch.min_confidence) {
        if (counters) {
            increment(&counters->low_confidence);
        }
        return false;
    }

    const std::uint32_t pixels_per_batch = launch.width * launch.height;
    const std::uint32_t pixel_index = i % pixels_per_batch;
    const std::uint32_t batch_index = i / pixels_per_batch;
    const std::uint32_t px = pixel_index % launch.width;
    const std::uint32_t py = pixel_index / launch.width;

    GaussianReconstructionSample sample{};
    sample.pixel = { static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f };
    sample.barycentric_uv = { raster.x, raster.y };
    sample.depth_ndc = raster.z;
    sample.radiance = color;
    sample.confidence = confidence;
    sample.primitive_id = tri_plus_one - 1u;
    sample.pixel_index = pixel_index;
    sample.batch_index = batch_index;
    sample.flags = inputs.color_rgba ? GAUSSIAN_SAMPLE_HAS_COLOR : 0u;

    if (inputs.world_position) {
        const float3 p = inputs.world_position[i];
        if (!finite_float3(p)) {
            if (counters) {
                increment(&counters->invalid);
            }
            return false;
        }
        sample.position = p;
        sample.flags |= GAUSSIAN_SAMPLE_HAS_WORLD_POSITION;
    } else {
        sample.position = { sample.pixel.x, sample.pixel.y, raster.z };
    }

    if (inputs.normal) {
        const float3 n = inputs.normal[i];
        if (finite_float3(n)) {
            sample.normal = n;
            sample.flags |= GAUSSIAN_SAMPLE_HAS_NORMAL;
        } else {
            sample.normal = { 0.0f, 0.0f, 1.0f };
        }
    } else {
        sample.normal = { 0.0f, 0.0f, 1.0f };
    }

    if (inputs.motion_px) {
        const float2 motion = inputs.motion_px[i];
        if (finite_float2(motion)) {
            sample.motion_px = motion;
            sample.flags |= GAUSSIAN_SAMPLE_HAS_MOTION;
        }
    }

    *out_sample = sample;
    return true;
}

__device__ void write_tensor_sample(const GaussianReconstructionTensorOutputs& outputs,
                                    std::uint32_t slot,
                                    const GaussianReconstructionSample& sample) {
    if (outputs.position) {
        outputs.position[slot] = sample.position;
    }
    if (outputs.normal) {
        outputs.normal[slot] = sample.normal;
    }
    if (outputs.radiance) {
        outputs.radiance[slot] = sample.radiance;
    }
    if (outputs.pixel) {
        outputs.pixel[slot] = sample.pixel;
    }
    if (outputs.motion_px) {
        outputs.motion_px[slot] = sample.motion_px;
    }
    if (outputs.barycentric_uv) {
        outputs.barycentric_uv[slot] = sample.barycentric_uv;
    }
    if (outputs.depth_confidence) {
        outputs.depth_confidence[slot] = { sample.depth_ndc, sample.confidence };
    }
    if (outputs.ids) {
        outputs.ids[slot] = {
            sample.primitive_id,
            sample.pixel_index,
            sample.batch_index,
            sample.flags,
        };
    }
}

__global__ void extract_nvdiffrast_samples_kernel(NvDiffrastRasterLaunch launch,
                                                  NvDiffrastRasterInputs inputs,
                                                  GaussianReconstructionSample* samples,
                                                  std::uint32_t* sample_count,
                                                  NvDiffrastExtractCounters* counters) {
    const std::uint32_t total_pixels = launch.width * launch.height * launch.batch;
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;

    for (std::uint32_t i = thread_id; i < total_pixels; i += stride) {
        GaussianReconstructionSample sample{};
        if (!build_sample(launch, inputs, i, counters, &sample)) {
            continue;
        }

        std::uint32_t slot = 0;
        if (!reserve_sample(sample_count, launch.max_samples, &slot)) {
            if (counters) {
                increment(&counters->capacity_overflow);
            }
            continue;
        }

        samples[slot] = sample;
        if (counters) {
            increment(&counters->emitted);
        }
    }
}

__global__ void extract_nvdiffrast_sample_tensors_kernel(NvDiffrastRasterLaunch launch,
                                                         NvDiffrastRasterInputs inputs,
                                                         GaussianReconstructionTensorOutputs outputs,
                                                         std::uint32_t* sample_count,
                                                         NvDiffrastExtractCounters* counters) {
    const std::uint32_t total_pixels = launch.width * launch.height * launch.batch;
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;

    for (std::uint32_t i = thread_id; i < total_pixels; i += stride) {
        GaussianReconstructionSample sample{};
        if (!build_sample(launch, inputs, i, counters, &sample)) {
            continue;
        }

        std::uint32_t slot = 0;
        if (!reserve_sample(sample_count, launch.max_samples, &slot)) {
            if (counters) {
                increment(&counters->capacity_overflow);
            }
            continue;
        }

        write_tensor_sample(outputs, slot, sample);
        if (counters) {
            increment(&counters->emitted);
        }
    }
}

__global__ void extract_seed_frame_sample_tensors_kernel(
    GaussianSeedFrameTensorLaunch launch,
    GaussianSeedFrameTensorInputs inputs,
    GaussianReconstructionTensorOutputs outputs,
    std::uint32_t* sample_count,
    GaussianSeedFrameExtractCounters* counters) {
    const std::uint32_t total_pixels = launch.width * launch.height * launch.batch;
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pixels) {
        return;
    }

    if (counters) {
        increment(&counters->visited);
    }

    const GaussianReconstructionSampleIds dense_ids = inputs.ids[i];
    if (dense_ids.flags == 0u) {
        if (counters) {
            increment(&counters->background);
        }
        return;
    }

    const float4 radiance = inputs.radiance[i];
    const float2 depth_confidence = inputs.depth_confidence[i];
    if (!finite_float4(radiance) || !finite_float2(depth_confidence)) {
        if (counters) {
            increment(&counters->invalid);
        }
        return;
    }

    const float confidence = clamp01(depth_confidence.y);
    if (confidence < launch.min_confidence) {
        if (counters) {
            increment(&counters->low_confidence);
        }
        return;
    }

    float3 world_position{ 0.0f, 0.0f, 0.0f };
    float3 normal{ 0.0f, 0.0f, 1.0f };
    float2 motion_px{ 0.0f, 0.0f };
    std::uint32_t flags = GAUSSIAN_SAMPLE_HAS_COLOR;
    if (inputs.world_position) {
        world_position = inputs.world_position[i];
        if (!finite_float3(world_position)) {
            if (counters) {
                increment(&counters->invalid);
            }
            return;
        }
        flags |= GAUSSIAN_SAMPLE_HAS_WORLD_POSITION;
    }
    if (inputs.normal) {
        normal = inputs.normal[i];
        if (!finite_float3(normal)) {
            if (counters) {
                increment(&counters->invalid);
            }
            return;
        }
        flags |= GAUSSIAN_SAMPLE_HAS_NORMAL;
    }
    if (inputs.motion_px) {
        motion_px = inputs.motion_px[i];
        if (!finite_float2(motion_px)) {
            if (counters) {
                increment(&counters->invalid);
            }
            return;
        }
        flags |= GAUSSIAN_SAMPLE_HAS_MOTION;
    }

    std::uint32_t slot = 0;
    if (!reserve_sample(sample_count, launch.max_samples, &slot)) {
        if (counters) {
            increment(&counters->capacity_overflow);
        }
        return;
    }

    const std::uint32_t pixels_per_batch = launch.width * launch.height;
    const std::uint32_t pixel_index = i % pixels_per_batch;
    const std::uint32_t batch_index = i / pixels_per_batch;
    const std::uint32_t px = pixel_index % launch.width;
    const std::uint32_t py = pixel_index / launch.width;

    outputs.position[slot] = world_position;
    outputs.normal[slot] = normal;
    outputs.radiance[slot] = { radiance.x, radiance.y, radiance.z, confidence };
    outputs.pixel[slot] = { static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f };
    outputs.motion_px[slot] = motion_px;
    outputs.barycentric_uv[slot] = { 0.0f, 0.0f };
    outputs.depth_confidence[slot] = { depth_confidence.x, confidence };
    outputs.ids[slot] = {
        dense_ids.primitive_id,
        pixel_index,
        batch_index,
        dense_ids.flags | flags,
    };

    if (counters) {
        increment(&counters->emitted);
    }
}

__global__ void read_gaussian_sample_count_info_kernel(
    GaussianSampleCountInfoLaunch launch,
    const std::uint32_t* sample_count,
    GaussianSampleCountInfo* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !output) {
        return;
    }

    const std::uint32_t raw = sample_count ? *sample_count : 0u;
    const std::uint32_t clamped = raw < launch.max_samples ? raw : launch.max_samples;
    output->raw = raw;
    output->clamped = clamped;
    output->overflow = raw - clamped;
    output->available = launch.max_samples - clamped;
}

bool valid_launch(const NvDiffrastRasterLaunch& launch,
                  const NvDiffrastRasterInputs& inputs,
                  std::uint32_t* sample_count) {
    return inputs.raster && sample_count && launch.width > 0 && launch.height > 0 &&
           launch.batch > 0 && launch.max_samples > 0;
}

void reset_outputs(std::uint32_t* sample_count,
                   NvDiffrastExtractCounters* counters,
                   cudaStream_t stream) {
    if (sample_count) {
        cudaMemsetAsync(sample_count, 0, sizeof(std::uint32_t), stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(NvDiffrastExtractCounters), stream);
    }
}

bool valid_seed_frame_launch(const GaussianSeedFrameTensorLaunch& launch,
                             const GaussianSeedFrameTensorInputs& inputs,
                             const GaussianReconstructionTensorOutputs& outputs,
                             std::uint32_t* sample_count) {
    return launch.width > 0 && launch.height > 0 && launch.batch > 0 &&
           launch.max_samples > 0 && sample_count && inputs.radiance &&
           inputs.depth_confidence && inputs.ids && outputs.position &&
           outputs.normal && outputs.radiance && outputs.pixel && outputs.motion_px &&
           outputs.barycentric_uv && outputs.depth_confidence && outputs.ids;
}

void reset_seed_frame_outputs(std::uint32_t* sample_count,
                              GaussianSeedFrameExtractCounters* counters,
                              cudaStream_t stream) {
    if (sample_count) {
        cudaMemsetAsync(sample_count, 0, sizeof(std::uint32_t), stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianSeedFrameExtractCounters), stream);
    }
}

__device__ float luma(float4 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ void atomic_add_float2(float2* dst, float2 value) {
    atomicAdd(&dst->x, value.x);
    atomicAdd(&dst->y, value.y);
}

__device__ void atomic_add_float3(float3* dst, float3 value) {
    atomicAdd(&dst->x, value.x);
    atomicAdd(&dst->y, value.y);
    atomicAdd(&dst->z, value.z);
}

__device__ void atomic_add_float4(float4* dst, float4 value) {
    atomicAdd(&dst->x, value.x);
    atomicAdd(&dst->y, value.y);
    atomicAdd(&dst->z, value.z);
    atomicAdd(&dst->w, value.w);
}

__device__ float3 mul(float3 v, float s) {
    return { v.x * s, v.y * s, v.z * s };
}

__device__ float4 mul(float4 v, float s) {
    return { v.x * s, v.y * s, v.z * s, v.w * s };
}

__device__ float2 mul(float2 v, float s) {
    return { v.x * s, v.y * s };
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 sqr(float3 v) {
    return { v.x * v.x, v.y * v.y, v.z * v.z };
}

__device__ bool valid_state(const GaussianReconstructionState& state) {
    return state.position && state.normal && state.radiance && state.pixel &&
           state.motion_px && state.covariance_diag && state.depth_confidence &&
           state.mass_variance && state.ids;
}

__device__ bool valid_sample_for_state(const GaussianReconstructionSample& sample) {
    return (sample.flags & GAUSSIAN_SAMPLE_HAS_WORLD_POSITION) &&
           finite_float3(sample.position) && finite_float4(sample.radiance) &&
           finite_float3(sample.normal) && finite_float2(sample.pixel) &&
           finite_float2(sample.motion_px) && finite_float(sample.depth_ndc) &&
           finite_float(sample.confidence);
}

__host__ __device__ std::uint32_t div_up_u32(std::uint32_t value, std::uint32_t divisor) {
    return divisor == 0u ? 0u : (value + divisor - 1u) / divisor;
}

constexpr std::uint32_t kInvalidTileId = 0xffffffffu;
constexpr std::uint32_t kMaxWeightedResolveRadiusPx = 8u;
constexpr std::uint32_t kTileOffsetScanElementsPerThread = 4u;

__device__ bool sample_is_empty(GaussianReconstructionSampleIds ids) {
    return ids.flags == 0u;
}

__device__ bool normal_gate_passes(float3 sample_normal,
                                   float3 guide_normal,
                                   float normal_dot_min) {
    if (!finite_float3(sample_normal) || !finite_float3(guide_normal)) {
        return false;
    }
    const float sample_len2 = dot(sample_normal, sample_normal);
    const float guide_len2 = dot(guide_normal, guide_normal);
    if (sample_len2 <= 1.0e-12f || guide_len2 <= 1.0e-12f) {
        return false;
    }
    const float inv_len = rsqrtf(sample_len2 * guide_len2);
    return dot(sample_normal, guide_normal) * inv_len >= normal_dot_min;
}

__device__ bool motion_gate_passes(float2 sample_motion,
                                   float2 guide_motion,
                                   float motion_epsilon_px) {
    if (!finite_float2(sample_motion) || !finite_float2(guide_motion)) {
        return false;
    }
    const float dx = sample_motion.x - guide_motion.x;
    const float dy = sample_motion.y - guide_motion.y;
    return dx * dx + dy * dy <= motion_epsilon_px * motion_epsilon_px;
}

__device__ bool tile_resolve_guides_pass(
    GaussianTileGatedWeightedResolveLaunch launch,
    GaussianTileResolveGuides guides,
    const GaussianReconstructionTensorSamples& samples,
    std::uint32_t sample_index,
    std::uint32_t out_i,
    GaussianReconstructionSampleIds sample_ids,
    float2 sample_depth_confidence,
    GaussianTileGatedWeightedResolveCounters* counters) {
    if (launch.gate_flags & GAUSSIAN_TILE_GATE_PRIMITIVE_ID) {
        const GaussianReconstructionSampleIds guide_ids = guides.ids[out_i];
        if (sample_is_empty(guide_ids) || guide_ids.primitive_id != sample_ids.primitive_id) {
            if (counters) {
                increment(&counters->rejected_primitive);
            }
            return false;
        }
    }

    if (launch.gate_flags & GAUSSIAN_TILE_GATE_DEPTH) {
        const float2 guide_depth_confidence = guides.depth_confidence[out_i];
        if (!finite_float2(guide_depth_confidence) || guide_depth_confidence.y <= 0.0f ||
            fabsf(sample_depth_confidence.x - guide_depth_confidence.x) > launch.depth_epsilon) {
            if (counters) {
                increment(&counters->rejected_depth);
            }
            return false;
        }
    }

    if (launch.gate_flags & GAUSSIAN_TILE_GATE_NORMAL) {
        if (!normal_gate_passes(samples.normal[sample_index], guides.normal[out_i],
                                launch.normal_dot_min)) {
            if (counters) {
                increment(&counters->rejected_normal);
            }
            return false;
        }
    }

    if (launch.gate_flags & GAUSSIAN_TILE_GATE_MOTION) {
        if (!motion_gate_passes(samples.motion_px[sample_index], guides.motion_px[out_i],
                                launch.motion_epsilon_px)) {
            if (counters) {
                increment(&counters->rejected_motion);
            }
            return false;
        }
    }

    return true;
}

__device__ GaussianReconstructionSample read_tensor_sample(const GaussianReconstructionTensorSamples& samples,
                                                           std::uint32_t i) {
    GaussianReconstructionSample sample{};
    const GaussianReconstructionSampleIds ids = samples.ids[i];
    sample.position = samples.position[i];
    sample.normal = samples.normal ? samples.normal[i] : float3{ 0.0f, 0.0f, 1.0f };
    sample.radiance = samples.radiance[i];
    sample.pixel = samples.pixel ? samples.pixel[i] : float2{ 0.0f, 0.0f };
    sample.motion_px = samples.motion_px ? samples.motion_px[i] : float2{ 0.0f, 0.0f };
    const float2 depth_confidence = samples.depth_confidence[i];
    sample.depth_ndc = depth_confidence.x;
    sample.confidence = depth_confidence.y;
    sample.primitive_id = ids.primitive_id;
    sample.pixel_index = ids.pixel_index;
    sample.batch_index = ids.batch_index;
    sample.flags = ids.flags;
    return sample;
}

__global__ void clear_gaussian_state_kernel(GaussianReconstructionState state,
                                            std::uint32_t gaussian_capacity) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gaussian_capacity) {
        return;
    }
    state.position[i] = { 0.0f, 0.0f, 0.0f };
    state.normal[i] = { 0.0f, 0.0f, 0.0f };
    state.radiance[i] = { 0.0f, 0.0f, 0.0f, 0.0f };
    state.pixel[i] = { 0.0f, 0.0f };
    state.motion_px[i] = { 0.0f, 0.0f };
    state.covariance_diag[i] = { 0.0f, 0.0f, 0.0f };
    state.depth_confidence[i] = { 0.0f, 0.0f };
    state.mass_variance[i] = { 0.0f, 0.0f };
    state.ids[i] = { 0u, 0u, 0u, 0u };
}

__device__ void accumulate_one_sample(const GaussianStateUpdateLaunch& launch,
                                      const GaussianReconstructionSample& sample,
                                      GaussianReconstructionState state,
                                      GaussianStateUpdateCounters* counters) {
    if (counters) {
        increment(&counters->visited);
    }
    if (!valid_sample_for_state(sample)) {
        if (counters) {
            increment((sample.flags & GAUSSIAN_SAMPLE_HAS_WORLD_POSITION) ? &counters->invalid : &counters->skipped_no_position);
        }
        return;
    }
    if (sample.confidence < launch.min_confidence) {
        if (counters) {
            increment(&counters->low_confidence);
        }
        return;
    }
    if (sample.primitive_id >= launch.gaussian_capacity) {
        if (counters) {
            increment(&counters->slot_overflow);
        }
        return;
    }

    const std::uint32_t slot = sample.primitive_id;
    const float w = clamp01(sample.confidence);
    atomic_add_float3(&state.position[slot], mul(sample.position, w));
    atomic_add_float3(&state.normal[slot], mul(sample.normal, w));
    atomic_add_float4(&state.radiance[slot], mul(sample.radiance, w));
    atomic_add_float2(&state.pixel[slot], mul(sample.pixel, w));
    atomic_add_float2(&state.motion_px[slot], mul(sample.motion_px, w));
    atomic_add_float3(&state.covariance_diag[slot], mul(sqr(sample.position), w));
    atomic_add_float2(&state.depth_confidence[slot], { sample.depth_ndc * w, 0.0f });
    const float y = luma(sample.radiance);
    atomic_add_float2(&state.mass_variance[slot], { w, y * y * w });
    state.ids[slot] = { sample.primitive_id, sample.pixel_index, sample.batch_index, sample.flags };
    if (counters) {
        increment(&counters->updated);
    }
}

__global__ void accumulate_gaussian_state_from_samples_kernel(GaussianStateUpdateLaunch launch,
                                                              const GaussianReconstructionSample* samples,
                                                              GaussianReconstructionState state,
                                                              GaussianStateUpdateCounters* counters) {
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    for (std::uint32_t i = thread_id; i < launch.sample_count; i += stride) {
        accumulate_one_sample(launch, samples[i], state, counters);
    }
}

__global__ void accumulate_gaussian_state_from_tensor_samples_kernel(GaussianStateUpdateLaunch launch,
                                                                     GaussianReconstructionTensorSamples samples,
                                                                     GaussianReconstructionState state,
                                                                     GaussianStateUpdateCounters* counters) {
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    for (std::uint32_t i = thread_id; i < launch.sample_count; i += stride) {
        accumulate_one_sample(launch, read_tensor_sample(samples, i), state, counters);
    }
}

__global__ void accumulate_gaussian_state_from_counted_tensor_samples_kernel(
    GaussianStateTensorCountUpdateLaunch launch,
    GaussianReconstructionTensorSamples samples,
    const std::uint32_t* sample_count,
    GaussianReconstructionState state,
    GaussianStateUpdateCounters* counters) {
    const std::uint32_t raw_count = sample_count ? *sample_count : 0u;
    const std::uint32_t clamped_count = raw_count < launch.max_samples ? raw_count : launch.max_samples;
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;
    const GaussianStateUpdateLaunch scalar_launch{
        clamped_count,
        launch.gaussian_capacity,
        launch.min_confidence,
    };
    for (std::uint32_t i = thread_id; i < clamped_count; i += stride) {
        accumulate_one_sample(scalar_launch, read_tensor_sample(samples, i), state, counters);
    }
}

__global__ void build_gaussian_sample_tile_bins_kernel(
    GaussianSampleTileBinningLaunch launch,
    GaussianReconstructionTensorSamples samples,
    const std::uint32_t* sample_count,
    GaussianSampleTileBinningOutputs outputs,
    GaussianSampleTileBinningCounters* counters) {
    const std::uint32_t raw_count = sample_count ? *sample_count : 0u;
    const std::uint32_t clamped_count = raw_count < launch.max_samples ? raw_count : launch.max_samples;
    const std::uint32_t tiles_x = div_up_u32(launch.width, launch.tile_width);
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;

    for (std::uint32_t i = thread_id; i < clamped_count; i += stride) {
        if (counters) {
            increment(&counters->visited);
        }

        const GaussianReconstructionSampleIds ids = samples.ids[i];
        if (sample_is_empty(ids)) {
            if (counters) {
                increment(&counters->skipped_empty);
            }
            continue;
        }

        const float2 pixel = samples.pixel[i];
        if (!finite_float2(pixel)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const int px = static_cast<int>(floorf(pixel.x));
        const int py = static_cast<int>(floorf(pixel.y));
        if (px < 0 || py < 0 || px >= static_cast<int>(launch.width) ||
            py >= static_cast<int>(launch.height)) {
            if (counters) {
                increment(&counters->out_of_bounds);
            }
            continue;
        }

        const std::uint32_t tile_x = static_cast<std::uint32_t>(px) / launch.tile_width;
        const std::uint32_t tile_y = static_cast<std::uint32_t>(py) / launch.tile_height;
        const std::uint32_t tile = tile_y * tiles_x + tile_x;
        outputs.sample_tile[i] = tile;
        atomicAdd(&outputs.tile_counts[tile], 1u);
        if (counters) {
            increment(&counters->binned);
        }
    }
}

__global__ void build_gaussian_sample_tile_offsets_kernel(
    GaussianSampleTileOffsetLaunch launch,
    GaussianSampleTileOffsetInputs inputs,
    GaussianSampleTileOffsetOutputs outputs,
    GaussianSampleTileOffsetCounters* counters) {
    extern __shared__ std::uint32_t thread_sums[];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t thread_capacity = blockDim.x * kTileOffsetScanElementsPerThread;

    if (launch.tile_count == 0 || !inputs.tile_counts || !outputs.tile_offsets) {
        return;
    }

    if (launch.tile_count > thread_capacity) {
        if (tid == 0) {
            std::uint32_t sum = 0;
            outputs.tile_offsets[0] = 0;
            for (std::uint32_t i = 0; i < launch.tile_count; ++i) {
                sum += inputs.tile_counts[i];
                outputs.tile_offsets[i + 1u] = sum;
            }
            if (counters) {
                counters->tile_count = launch.tile_count;
                counters->total_samples = sum;
            }
        }
        return;
    }

    const std::uint32_t base = tid * kTileOffsetScanElementsPerThread;
    std::uint32_t local_prefix[kTileOffsetScanElementsPerThread] = {};
    std::uint32_t running = 0;
    for (std::uint32_t j = 0; j < kTileOffsetScanElementsPerThread; ++j) {
        const std::uint32_t idx = base + j;
        local_prefix[j] = running;
        running += idx < launch.tile_count ? inputs.tile_counts[idx] : 0u;
    }

    thread_sums[tid] = running;
    __syncthreads();

    for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1u) {
        const std::uint32_t add = tid >= offset ? thread_sums[tid - offset] : 0u;
        __syncthreads();
        if (tid >= offset) {
            thread_sums[tid] += add;
        }
        __syncthreads();
    }

    const std::uint32_t thread_base = tid == 0 ? 0u : thread_sums[tid - 1u];
    for (std::uint32_t j = 0; j < kTileOffsetScanElementsPerThread; ++j) {
        const std::uint32_t idx = base + j;
        if (idx < launch.tile_count) {
            outputs.tile_offsets[idx] = thread_base + local_prefix[j];
        }
    }

    if (tid == blockDim.x - 1u) {
        outputs.tile_offsets[launch.tile_count] = thread_sums[tid];
        if (counters) {
            counters->tile_count = launch.tile_count;
            counters->total_samples = thread_sums[tid];
        }
    }
}

__global__ void compact_gaussian_sample_tile_bins_kernel(
    GaussianSampleTileCompactionLaunch launch,
    const std::uint32_t* sample_count,
    GaussianSampleTileCompactionInputs inputs,
    GaussianSampleTileCompactionOutputs outputs,
    GaussianSampleTileCompactionCounters* counters) {
    const std::uint32_t raw_count = sample_count ? *sample_count : 0u;
    const std::uint32_t clamped_count = raw_count < launch.max_samples ? raw_count : launch.max_samples;
    const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = blockDim.x * gridDim.x;

    for (std::uint32_t i = thread_id; i < clamped_count; i += stride) {
        if (counters) {
            increment(&counters->visited);
        }

        const std::uint32_t tile = inputs.sample_tile[i];
        if (tile == kInvalidTileId || tile >= launch.tile_count) {
            if (counters) {
                increment(&counters->invalid_tile);
            }
            continue;
        }

        const std::uint32_t local_slot = atomicAdd(&outputs.tile_write_counts[tile], 1u);
        const std::uint32_t begin = inputs.tile_offsets[tile];
        const std::uint32_t end = inputs.tile_offsets[tile + 1u];
        const std::uint32_t out_slot = begin + local_slot;
        if (out_slot >= end || out_slot >= launch.max_samples) {
            if (counters) {
                increment(&counters->capacity_overflow);
            }
            continue;
        }

        outputs.tile_sample_indices[out_slot] = i;
        if (counters) {
            increment(&counters->compacted);
        }
    }
}

__global__ void resolve_gaussian_sample_tiles_kernel(
    GaussianTileSampleResolveLaunch launch,
    GaussianReconstructionTensorSamples samples,
    GaussianTileSampleResolveInputs inputs,
    GaussianTileSampleResolveOutputs outputs,
    GaussianTileSampleResolveCounters* counters) {
    const std::uint32_t tile = blockIdx.x;
    if (tile >= launch.tile_count) {
        return;
    }

    if (threadIdx.x == 0 && counters) {
        increment(&counters->visited_tiles);
    }

    const std::uint32_t raw_begin = inputs.tile_offsets[tile];
    const std::uint32_t raw_end = inputs.tile_offsets[tile + 1u];
    if (raw_begin >= raw_end) {
        if (threadIdx.x == 0 && counters) {
            increment(&counters->empty_tiles);
        }
        return;
    }
    const std::uint32_t begin = raw_begin < launch.max_samples ? raw_begin : launch.max_samples;
    const std::uint32_t end = raw_end < launch.max_samples ? raw_end : launch.max_samples;
    if (begin >= end) {
        if (threadIdx.x == 0 && counters) {
            increment(&counters->invalid);
        }
        return;
    }

    for (std::uint32_t cursor = begin + threadIdx.x; cursor < end; cursor += blockDim.x) {
        const std::uint32_t sample_index = inputs.tile_sample_indices[cursor];
        if (counters) {
            increment(&counters->visited_samples);
        }
        if (sample_index == kInvalidTileId || sample_index >= launch.max_samples) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const GaussianReconstructionSampleIds ids = samples.ids[sample_index];
        if (sample_is_empty(ids)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const float2 pixel = samples.pixel[sample_index];
        const float2 depth_confidence = samples.depth_confidence[sample_index];
        const float4 radiance = samples.radiance[sample_index];
        if (!finite_float2(pixel) || !finite_float2(depth_confidence) || !finite_float4(radiance)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }
        if (depth_confidence.y < launch.min_confidence) {
            if (counters) {
                increment(&counters->low_confidence);
            }
            continue;
        }

        const int px = static_cast<int>(floorf(pixel.x));
        const int py = static_cast<int>(floorf(pixel.y));
        if (px < 0 || py < 0 || px >= static_cast<int>(launch.width) ||
            py >= static_cast<int>(launch.height)) {
            if (counters) {
                increment(&counters->out_of_bounds);
            }
            continue;
        }

        const std::uint32_t out_i = static_cast<std::uint32_t>(py) * launch.width +
                                    static_cast<std::uint32_t>(px);
        outputs.radiance[out_i] = { radiance.x, radiance.y, radiance.z, depth_confidence.y };
        outputs.depth_confidence[out_i] = depth_confidence;
        outputs.ids[out_i] = ids;
        if (counters) {
            increment(&counters->emitted);
        }
    }
}

__global__ void resolve_gaussian_sample_tiles_weighted_accumulate_kernel(
    GaussianTileWeightedResolveLaunch launch,
    GaussianReconstructionTensorSamples samples,
    GaussianTileSampleResolveInputs inputs,
    GaussianTileSampleResolveOutputs outputs,
    GaussianTileWeightedResolveCounters* counters) {
    const std::uint32_t tile = blockIdx.x;
    if (tile >= launch.tile_count) {
        return;
    }

    if (threadIdx.x == 0 && counters) {
        increment(&counters->visited_tiles);
    }

    const std::uint32_t raw_begin = inputs.tile_offsets[tile];
    const std::uint32_t raw_end = inputs.tile_offsets[tile + 1u];
    if (raw_begin >= raw_end) {
        if (threadIdx.x == 0 && counters) {
            increment(&counters->empty_tiles);
        }
        return;
    }
    const std::uint32_t begin = raw_begin < launch.max_samples ? raw_begin : launch.max_samples;
    const std::uint32_t end = raw_end < launch.max_samples ? raw_end : launch.max_samples;
    if (begin >= end) {
        if (threadIdx.x == 0 && counters) {
            increment(&counters->invalid);
        }
        return;
    }

    const int radius = static_cast<int>(launch.radius_px);
    const float sigma = fmaxf(launch.sigma_px, 1.0e-6f);
    const float inv_two_sigma2 = 0.5f / (sigma * sigma);

    for (std::uint32_t cursor = begin + threadIdx.x; cursor < end; cursor += blockDim.x) {
        const std::uint32_t sample_index = inputs.tile_sample_indices[cursor];
        if (counters) {
            increment(&counters->visited_samples);
        }
        if (sample_index == kInvalidTileId || sample_index >= launch.max_samples) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const GaussianReconstructionSampleIds ids = samples.ids[sample_index];
        if (sample_is_empty(ids)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const float2 pixel = samples.pixel[sample_index];
        const float2 depth_confidence = samples.depth_confidence[sample_index];
        const float4 radiance = samples.radiance[sample_index];
        if (!finite_float2(pixel) || !finite_float2(depth_confidence) || !finite_float4(radiance)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }
        const float confidence = clamp01(depth_confidence.y);
        if (confidence < launch.min_confidence) {
            if (counters) {
                increment(&counters->low_confidence);
            }
            continue;
        }

        const int center_px = static_cast<int>(floorf(pixel.x));
        const int center_py = static_cast<int>(floorf(pixel.y));
        for (int dy = -radius; dy <= radius; ++dy) {
            const int py = center_py + dy;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int px = center_px + dx;
                if (px < 0 || py < 0 || px >= static_cast<int>(launch.width) ||
                    py >= static_cast<int>(launch.height)) {
                    if (counters) {
                        increment(&counters->out_of_bounds);
                    }
                    continue;
                }

                const float pixel_center_x = static_cast<float>(px) + 0.5f;
                const float pixel_center_y = static_cast<float>(py) + 0.5f;
                const float delta_x = pixel_center_x - pixel.x;
                const float delta_y = pixel_center_y - pixel.y;
                const float tap_weight = confidence * expf(-(delta_x * delta_x + delta_y * delta_y) *
                                                          inv_two_sigma2);
                if (tap_weight <= 0.0f || !finite_float(tap_weight)) {
                    if (counters) {
                        increment(&counters->invalid);
                    }
                    continue;
                }

                const std::uint32_t out_i = static_cast<std::uint32_t>(py) * launch.width +
                                            static_cast<std::uint32_t>(px);
                atomicAdd(&outputs.radiance[out_i].x, radiance.x * tap_weight);
                atomicAdd(&outputs.radiance[out_i].y, radiance.y * tap_weight);
                atomicAdd(&outputs.radiance[out_i].z, radiance.z * tap_weight);
                atomicAdd(&outputs.radiance[out_i].w, tap_weight);
                atomicAdd(&outputs.depth_confidence[out_i].x, depth_confidence.x * tap_weight);
                atomicAdd(&outputs.depth_confidence[out_i].y, tap_weight);
                outputs.ids[out_i] = ids;
                if (counters) {
                    increment(&counters->emitted_taps);
                }
            }
        }
    }
}

__global__ void resolve_gaussian_sample_tiles_weighted_normalize_kernel(
    GaussianTileWeightedResolveLaunch launch,
    GaussianTileSampleResolveOutputs outputs,
    GaussianTileWeightedResolveCounters* counters) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (i >= pixel_count) {
        return;
    }

    const float weight = outputs.radiance[i].w;
    if (weight <= 1.0e-12f || !finite_float(weight)) {
        if (counters) {
            increment(&counters->zero_weight_pixels);
        }
        return;
    }

    const float inv_weight = 1.0f / weight;
    outputs.radiance[i] = {
        outputs.radiance[i].x * inv_weight,
        outputs.radiance[i].y * inv_weight,
        outputs.radiance[i].z * inv_weight,
        clamp01(weight),
    };
    outputs.depth_confidence[i] = {
        outputs.depth_confidence[i].x * inv_weight,
        clamp01(outputs.depth_confidence[i].y),
    };
    if (counters) {
        increment(&counters->normalized_pixels);
    }
}

__global__ void resolve_gaussian_sample_tiles_weighted_gated_accumulate_kernel(
    GaussianTileGatedWeightedResolveLaunch launch,
    GaussianReconstructionTensorSamples samples,
    GaussianTileSampleResolveInputs inputs,
    GaussianTileResolveGuides guides,
    GaussianTileSampleResolveOutputs outputs,
    GaussianTileGatedWeightedResolveCounters* counters) {
    const std::uint32_t tile = blockIdx.x;
    if (tile >= launch.tile_count) {
        return;
    }

    if (threadIdx.x == 0 && counters) {
        increment(&counters->visited_tiles);
    }

    const std::uint32_t raw_begin = inputs.tile_offsets[tile];
    const std::uint32_t raw_end = inputs.tile_offsets[tile + 1u];
    if (raw_begin >= raw_end) {
        if (threadIdx.x == 0 && counters) {
            increment(&counters->empty_tiles);
        }
        return;
    }
    const std::uint32_t begin = raw_begin < launch.max_samples ? raw_begin : launch.max_samples;
    const std::uint32_t end = raw_end < launch.max_samples ? raw_end : launch.max_samples;
    if (begin >= end) {
        if (threadIdx.x == 0 && counters) {
            increment(&counters->invalid);
        }
        return;
    }

    const int radius = static_cast<int>(launch.radius_px);
    const float sigma = fmaxf(launch.sigma_px, 1.0e-6f);
    const float inv_two_sigma2 = 0.5f / (sigma * sigma);

    for (std::uint32_t cursor = begin + threadIdx.x; cursor < end; cursor += blockDim.x) {
        const std::uint32_t sample_index = inputs.tile_sample_indices[cursor];
        if (counters) {
            increment(&counters->visited_samples);
        }
        if (sample_index == kInvalidTileId || sample_index >= launch.max_samples) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const GaussianReconstructionSampleIds ids = samples.ids[sample_index];
        if (sample_is_empty(ids)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }

        const float2 pixel = samples.pixel[sample_index];
        const float2 depth_confidence = samples.depth_confidence[sample_index];
        const float4 radiance = samples.radiance[sample_index];
        if (!finite_float2(pixel) || !finite_float2(depth_confidence) || !finite_float4(radiance)) {
            if (counters) {
                increment(&counters->invalid);
            }
            continue;
        }
        const float confidence = clamp01(depth_confidence.y);
        if (confidence < launch.min_confidence) {
            if (counters) {
                increment(&counters->low_confidence);
            }
            continue;
        }

        const int center_px = static_cast<int>(floorf(pixel.x));
        const int center_py = static_cast<int>(floorf(pixel.y));
        for (int dy = -radius; dy <= radius; ++dy) {
            const int py = center_py + dy;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int px = center_px + dx;
                if (px < 0 || py < 0 || px >= static_cast<int>(launch.width) ||
                    py >= static_cast<int>(launch.height)) {
                    if (counters) {
                        increment(&counters->out_of_bounds);
                    }
                    continue;
                }

                const std::uint32_t out_i = static_cast<std::uint32_t>(py) * launch.width +
                                            static_cast<std::uint32_t>(px);
                if (!tile_resolve_guides_pass(launch, guides, samples, sample_index, out_i, ids,
                                              depth_confidence, counters)) {
                    continue;
                }

                const float pixel_center_x = static_cast<float>(px) + 0.5f;
                const float pixel_center_y = static_cast<float>(py) + 0.5f;
                const float delta_x = pixel_center_x - pixel.x;
                const float delta_y = pixel_center_y - pixel.y;
                const float tap_weight = confidence * expf(-(delta_x * delta_x + delta_y * delta_y) *
                                                          inv_two_sigma2);
                if (tap_weight <= 0.0f || !finite_float(tap_weight)) {
                    if (counters) {
                        increment(&counters->invalid);
                    }
                    continue;
                }

                atomicAdd(&outputs.radiance[out_i].x, radiance.x * tap_weight);
                atomicAdd(&outputs.radiance[out_i].y, radiance.y * tap_weight);
                atomicAdd(&outputs.radiance[out_i].z, radiance.z * tap_weight);
                atomicAdd(&outputs.radiance[out_i].w, tap_weight);
                atomicAdd(&outputs.depth_confidence[out_i].x, depth_confidence.x * tap_weight);
                atomicAdd(&outputs.depth_confidence[out_i].y, tap_weight);
                outputs.ids[out_i] = ids;
                if (counters) {
                    increment(&counters->emitted_taps);
                }
            }
        }
    }
}

__global__ void resolve_gaussian_sample_tiles_weighted_gated_normalize_kernel(
    GaussianTileGatedWeightedResolveLaunch launch,
    GaussianTileSampleResolveOutputs outputs,
    GaussianTileGatedWeightedResolveCounters* counters) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (i >= pixel_count) {
        return;
    }

    const float weight = outputs.radiance[i].w;
    if (weight <= 1.0e-12f || !finite_float(weight)) {
        if (counters) {
            increment(&counters->zero_weight_pixels);
        }
        return;
    }

    const float inv_weight = 1.0f / weight;
    outputs.radiance[i] = {
        outputs.radiance[i].x * inv_weight,
        outputs.radiance[i].y * inv_weight,
        outputs.radiance[i].z * inv_weight,
        clamp01(weight),
    };
    outputs.depth_confidence[i] = {
        outputs.depth_confidence[i].x * inv_weight,
        clamp01(outputs.depth_confidence[i].y),
    };
    if (counters) {
        increment(&counters->normalized_pixels);
    }
}

__global__ void finalize_gaussian_state_kernel(GaussianReconstructionState state,
                                               std::uint32_t gaussian_capacity) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gaussian_capacity) {
        return;
    }
    const float mass = state.mass_variance[i].x;
    if (mass <= 0.0f) {
        return;
    }
    const float inv_mass = 1.0f / mass;
    const float3 mean = mul(state.position[i], inv_mass);
    const float3 second = mul(state.covariance_diag[i], inv_mass);
    state.position[i] = mean;
    state.covariance_diag[i] = {
        fmaxf(second.x - mean.x * mean.x, 1.0e-6f),
        fmaxf(second.y - mean.y * mean.y, 1.0e-6f),
        fmaxf(second.z - mean.z * mean.z, 1.0e-6f),
    };
    const float3 n = mul(state.normal[i], inv_mass);
    const float n_len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    state.normal[i] = n_len > 1.0e-6f ? float3{ n.x / n_len, n.y / n_len, n.z / n_len }
                                      : float3{ 0.0f, 0.0f, 1.0f };
    state.radiance[i] = mul(state.radiance[i], inv_mass);
    state.pixel[i] = mul(state.pixel[i], inv_mass);
    state.motion_px[i] = mul(state.motion_px[i], inv_mass);
    state.depth_confidence[i].x *= inv_mass;
    state.depth_confidence[i].y = clamp01(mass);
    const float mean_luma = luma(state.radiance[i]);
    const float second_luma = state.mass_variance[i].y * inv_mass;
    state.mass_variance[i].y = fmaxf(second_luma - mean_luma * mean_luma, 0.0f);
}

__global__ void project_gaussian_state_features_kernel(GaussianFeatureProjectionLaunch launch,
                                                       GaussianReconstructionState state,
                                                       GaussianFeatureProjectionOutputs outputs,
                                                       GaussianFeatureProjectionCounters* counters) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= launch.gaussian_count) {
        return;
    }
    if (counters) {
        increment(&counters->visited);
    }
    const float mass = state.mass_variance[i].x;
    const float confidence = state.depth_confidence[i].y;
    if (mass <= 0.0f || confidence < launch.min_confidence || state.ids[i].flags == 0u) {
        if (counters) {
            increment(&counters->skipped_empty);
        }
        return;
    }
    const float2 projected_pixel = {
        state.pixel[i].x + launch.motion_alpha * state.motion_px[i].x,
        state.pixel[i].y + launch.motion_alpha * state.motion_px[i].y,
    };
    const int px = static_cast<int>(floorf(projected_pixel.x));
    const int py = static_cast<int>(floorf(projected_pixel.y));
    if (px < 0 || py < 0 || px >= static_cast<int>(launch.width) || py >= static_cast<int>(launch.height)) {
        if (counters) {
            increment(&counters->out_of_bounds);
        }
        return;
    }
    const std::uint32_t out_i = static_cast<std::uint32_t>(py) * launch.width + static_cast<std::uint32_t>(px);
    outputs.radiance[out_i] = { state.radiance[i].x, state.radiance[i].y, state.radiance[i].z, confidence };
    outputs.depth_confidence[out_i] = state.depth_confidence[i];
    outputs.ids[out_i] = state.ids[i];
    if (counters) {
        increment(&counters->emitted);
    }
}

__global__ void project_gaussian_state_features_weighted_accumulate_kernel(
    GaussianStateWeightedProjectionLaunch launch,
    GaussianReconstructionState state,
    GaussianFeatureProjectionOutputs outputs,
    GaussianStateWeightedProjectionCounters* counters) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= launch.gaussian_count) {
        return;
    }
    if (counters) {
        increment(&counters->visited);
    }

    const float mass = state.mass_variance[i].x;
    const float confidence = state.depth_confidence[i].y;
    if (mass <= 0.0f || confidence < launch.min_confidence || state.ids[i].flags == 0u) {
        if (counters) {
            increment(&counters->skipped_empty);
        }
        return;
    }

    const float2 projected_pixel = {
        state.pixel[i].x + launch.motion_alpha * state.motion_px[i].x,
        state.pixel[i].y + launch.motion_alpha * state.motion_px[i].y,
    };
    const float4 radiance = state.radiance[i];
    const float2 depth_confidence = state.depth_confidence[i];
    if (!finite_float2(projected_pixel) || !finite_float4(radiance) ||
        !finite_float2(depth_confidence)) {
        if (counters) {
            increment(&counters->invalid);
        }
        return;
    }

    const int center_px = static_cast<int>(floorf(projected_pixel.x));
    const int center_py = static_cast<int>(floorf(projected_pixel.y));
    const int radius = static_cast<int>(launch.radius_px);
    const float sigma = fmaxf(launch.sigma_px, 1.0e-6f);
    const float inv_two_sigma2 = 0.5f / (sigma * sigma);

    for (int dy = -radius; dy <= radius; ++dy) {
        const int py = center_py + dy;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int px = center_px + dx;
            if (px < 0 || py < 0 || px >= static_cast<int>(launch.width) ||
                py >= static_cast<int>(launch.height)) {
                if (counters) {
                    increment(&counters->out_of_bounds);
                }
                continue;
            }

            const float pixel_center_x = static_cast<float>(px) + 0.5f;
            const float pixel_center_y = static_cast<float>(py) + 0.5f;
            const float delta_x = pixel_center_x - projected_pixel.x;
            const float delta_y = pixel_center_y - projected_pixel.y;
            const float tap_weight = confidence * expf(-(delta_x * delta_x + delta_y * delta_y) *
                                                       inv_two_sigma2);
            if (tap_weight <= 0.0f || !finite_float(tap_weight)) {
                if (counters) {
                    increment(&counters->invalid);
                }
                continue;
            }

            const std::uint32_t out_i = static_cast<std::uint32_t>(py) * launch.width +
                                        static_cast<std::uint32_t>(px);
            atomicAdd(&outputs.radiance[out_i].x, radiance.x * tap_weight);
            atomicAdd(&outputs.radiance[out_i].y, radiance.y * tap_weight);
            atomicAdd(&outputs.radiance[out_i].z, radiance.z * tap_weight);
            atomicAdd(&outputs.radiance[out_i].w, tap_weight);
            atomicAdd(&outputs.depth_confidence[out_i].x, depth_confidence.x * tap_weight);
            atomicAdd(&outputs.depth_confidence[out_i].y, tap_weight);
            outputs.ids[out_i] = state.ids[i];
            if (counters) {
                increment(&counters->emitted_taps);
            }
        }
    }
}

__global__ void project_gaussian_state_features_weighted_normalize_kernel(
    GaussianStateWeightedProjectionLaunch launch,
    GaussianFeatureProjectionOutputs outputs,
    GaussianStateWeightedProjectionCounters* counters) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (i >= pixel_count) {
        return;
    }

    const float weight = outputs.radiance[i].w;
    if (weight <= 1.0e-12f || !finite_float(weight)) {
        if (counters) {
            increment(&counters->zero_weight_pixels);
        }
        return;
    }

    const float inv_weight = 1.0f / weight;
    outputs.radiance[i] = {
        outputs.radiance[i].x * inv_weight,
        outputs.radiance[i].y * inv_weight,
        outputs.radiance[i].z * inv_weight,
        clamp01(weight),
    };
    outputs.depth_confidence[i] = {
        outputs.depth_confidence[i].x * inv_weight,
        clamp01(outputs.depth_confidence[i].y),
    };
    if (counters) {
        increment(&counters->normalized_pixels);
    }
}

bool valid_host_state(const GaussianReconstructionState& state) {
    return state.position && state.normal && state.radiance && state.pixel &&
           state.motion_px && state.covariance_diag && state.depth_confidence &&
           state.mass_variance && state.ids;
}

bool valid_tensor_samples(const GaussianReconstructionTensorSamples& samples) {
    return samples.position && samples.radiance && samples.depth_confidence && samples.ids;
}

bool valid_tile_binning_samples(const GaussianReconstructionTensorSamples& samples) {
    return samples.pixel && samples.ids;
}

bool valid_tile_binning_launch(const GaussianSampleTileBinningLaunch& launch,
                               const GaussianReconstructionTensorSamples& samples,
                               const std::uint32_t* sample_count,
                               const GaussianSampleTileBinningOutputs& outputs) {
    return launch.max_samples > 0 && launch.width > 0 && launch.height > 0 &&
           launch.tile_width > 0 && launch.tile_height > 0 && sample_count &&
           valid_tile_binning_samples(samples) && outputs.sample_tile && outputs.tile_counts;
}

bool valid_tile_compaction_launch(const GaussianSampleTileCompactionLaunch& launch,
                                  const std::uint32_t* sample_count,
                                  const GaussianSampleTileCompactionInputs& inputs,
                                  const GaussianSampleTileCompactionOutputs& outputs) {
    return launch.max_samples > 0 && launch.tile_count > 0 && sample_count &&
           inputs.sample_tile && inputs.tile_offsets && outputs.tile_write_counts &&
           outputs.tile_sample_indices;
}

bool valid_tile_offset_launch(const GaussianSampleTileOffsetLaunch& launch,
                              const GaussianSampleTileOffsetInputs& inputs,
                              const GaussianSampleTileOffsetOutputs& outputs) {
    return launch.tile_count > 0 && inputs.tile_counts && outputs.tile_offsets;
}

bool valid_tile_sample_resolve_launch(const GaussianTileSampleResolveLaunch& launch,
                                      const GaussianReconstructionTensorSamples& samples,
                                      const GaussianTileSampleResolveInputs& inputs,
                                      const GaussianTileSampleResolveOutputs& outputs) {
    return launch.width > 0 && launch.height > 0 && launch.tile_width > 0 &&
           launch.tile_height > 0 && launch.tile_count > 0 && launch.max_samples > 0 &&
           samples.pixel && samples.radiance && samples.depth_confidence && samples.ids &&
           inputs.tile_offsets && inputs.tile_sample_indices && outputs.radiance &&
           outputs.depth_confidence && outputs.ids;
}

bool valid_tile_weighted_resolve_launch(const GaussianTileWeightedResolveLaunch& launch,
                                        const GaussianReconstructionTensorSamples& samples,
                                        const GaussianTileSampleResolveInputs& inputs,
                                        const GaussianTileSampleResolveOutputs& outputs) {
    return launch.width > 0 && launch.height > 0 && launch.tile_width > 0 &&
           launch.tile_height > 0 && launch.tile_count > 0 && launch.max_samples > 0 &&
           launch.radius_px <= kMaxWeightedResolveRadiusPx && launch.sigma_px > 0.0f &&
           samples.pixel && samples.radiance && samples.depth_confidence && samples.ids &&
           inputs.tile_offsets && inputs.tile_sample_indices && outputs.radiance &&
           outputs.depth_confidence && outputs.ids;
}

bool valid_tile_gated_weighted_resolve_launch(const GaussianTileGatedWeightedResolveLaunch& launch,
                                              const GaussianReconstructionTensorSamples& samples,
                                              const GaussianTileSampleResolveInputs& inputs,
                                              const GaussianTileResolveGuides& guides,
                                              const GaussianTileSampleResolveOutputs& outputs) {
    if (!(launch.width > 0 && launch.height > 0 && launch.tile_width > 0 &&
          launch.tile_height > 0 && launch.tile_count > 0 && launch.max_samples > 0 &&
          launch.radius_px <= kMaxWeightedResolveRadiusPx && launch.sigma_px > 0.0f &&
          samples.pixel && samples.radiance && samples.depth_confidence && samples.ids &&
          inputs.tile_offsets && inputs.tile_sample_indices && outputs.radiance &&
          outputs.depth_confidence && outputs.ids)) {
        return false;
    }
    if ((launch.gate_flags & GAUSSIAN_TILE_GATE_PRIMITIVE_ID) && !guides.ids) {
        return false;
    }
    if ((launch.gate_flags & GAUSSIAN_TILE_GATE_DEPTH) && !guides.depth_confidence) {
        return false;
    }
    if ((launch.gate_flags & GAUSSIAN_TILE_GATE_NORMAL) && (!samples.normal || !guides.normal)) {
        return false;
    }
    if ((launch.gate_flags & GAUSSIAN_TILE_GATE_MOTION) && (!samples.motion_px || !guides.motion_px)) {
        return false;
    }
    return true;
}

bool valid_update_launch(const GaussianStateUpdateLaunch& launch) {
    return launch.sample_count > 0 && launch.gaussian_capacity > 0;
}

bool valid_counted_update_launch(const GaussianStateTensorCountUpdateLaunch& launch,
                                 const std::uint32_t* sample_count) {
    return sample_count && launch.max_samples > 0 && launch.gaussian_capacity > 0;
}

bool valid_state_weighted_projection_launch(const GaussianStateWeightedProjectionLaunch& launch,
                                            const GaussianReconstructionState& state,
                                            const GaussianFeatureProjectionOutputs& outputs) {
    return launch.width > 0 && launch.height > 0 && launch.gaussian_count > 0 &&
           launch.radius_px <= kMaxWeightedResolveRadiusPx && launch.sigma_px > 0.0f &&
           valid_host_state(state) && outputs.radiance && outputs.depth_confidence &&
           outputs.ids;
}

void reset_update_counters(GaussianStateUpdateCounters* counters, cudaStream_t stream) {
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianStateUpdateCounters), stream);
    }
}

void reset_tile_binning_outputs(const GaussianSampleTileBinningLaunch& launch,
                                const GaussianSampleTileBinningOutputs& outputs,
                                GaussianSampleTileBinningCounters* counters,
                                cudaStream_t stream) {
    const std::uint32_t tiles_x = div_up_u32(launch.width, launch.tile_width);
    const std::uint32_t tiles_y = div_up_u32(launch.height, launch.tile_height);
    const std::uint32_t tile_count = tiles_x * tiles_y;
    if (outputs.sample_tile) {
        cudaMemsetAsync(outputs.sample_tile, 0xff, sizeof(std::uint32_t) * launch.max_samples, stream);
    }
    if (outputs.tile_counts) {
        cudaMemsetAsync(outputs.tile_counts, 0, sizeof(std::uint32_t) * tile_count, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianSampleTileBinningCounters), stream);
    }
}

void reset_tile_compaction_outputs(const GaussianSampleTileCompactionLaunch& launch,
                                   const GaussianSampleTileCompactionOutputs& outputs,
                                   GaussianSampleTileCompactionCounters* counters,
                                   cudaStream_t stream) {
    if (outputs.tile_write_counts) {
        cudaMemsetAsync(outputs.tile_write_counts, 0, sizeof(std::uint32_t) * launch.tile_count, stream);
    }
    if (outputs.tile_sample_indices) {
        cudaMemsetAsync(outputs.tile_sample_indices, 0xff, sizeof(std::uint32_t) * launch.max_samples, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianSampleTileCompactionCounters), stream);
    }
}

void reset_tile_offset_outputs(const GaussianSampleTileOffsetLaunch& launch,
                               const GaussianSampleTileOffsetOutputs& outputs,
                               GaussianSampleTileOffsetCounters* counters,
                               cudaStream_t stream) {
    if (outputs.tile_offsets) {
        cudaMemsetAsync(outputs.tile_offsets, 0, sizeof(std::uint32_t) * (launch.tile_count + 1u), stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianSampleTileOffsetCounters), stream);
    }
}

void reset_tile_sample_resolve_outputs(const GaussianTileSampleResolveLaunch& launch,
                                       const GaussianTileSampleResolveOutputs& outputs,
                                       GaussianTileSampleResolveCounters* counters,
                                       cudaStream_t stream) {
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (outputs.radiance) {
        cudaMemsetAsync(outputs.radiance, 0, sizeof(float4) * pixel_count, stream);
    }
    if (outputs.depth_confidence) {
        cudaMemsetAsync(outputs.depth_confidence, 0, sizeof(float2) * pixel_count, stream);
    }
    if (outputs.ids) {
        cudaMemsetAsync(outputs.ids, 0, sizeof(GaussianReconstructionSampleIds) * pixel_count, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianTileSampleResolveCounters), stream);
    }
}

void reset_tile_weighted_resolve_outputs(const GaussianTileWeightedResolveLaunch& launch,
                                         const GaussianTileSampleResolveOutputs& outputs,
                                         GaussianTileWeightedResolveCounters* counters,
                                         cudaStream_t stream) {
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (outputs.radiance) {
        cudaMemsetAsync(outputs.radiance, 0, sizeof(float4) * pixel_count, stream);
    }
    if (outputs.depth_confidence) {
        cudaMemsetAsync(outputs.depth_confidence, 0, sizeof(float2) * pixel_count, stream);
    }
    if (outputs.ids) {
        cudaMemsetAsync(outputs.ids, 0, sizeof(GaussianReconstructionSampleIds) * pixel_count, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianTileWeightedResolveCounters), stream);
    }
}

void reset_tile_gated_weighted_resolve_outputs(const GaussianTileGatedWeightedResolveLaunch& launch,
                                               const GaussianTileSampleResolveOutputs& outputs,
                                               GaussianTileGatedWeightedResolveCounters* counters,
                                               cudaStream_t stream) {
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (outputs.radiance) {
        cudaMemsetAsync(outputs.radiance, 0, sizeof(float4) * pixel_count, stream);
    }
    if (outputs.depth_confidence) {
        cudaMemsetAsync(outputs.depth_confidence, 0, sizeof(float2) * pixel_count, stream);
    }
    if (outputs.ids) {
        cudaMemsetAsync(outputs.ids, 0, sizeof(GaussianReconstructionSampleIds) * pixel_count, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianTileGatedWeightedResolveCounters), stream);
    }
}

void reset_projection_outputs(const GaussianFeatureProjectionLaunch& launch,
                              const GaussianFeatureProjectionOutputs& outputs,
                              GaussianFeatureProjectionCounters* counters,
                              cudaStream_t stream) {
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (outputs.radiance) {
        cudaMemsetAsync(outputs.radiance, 0, sizeof(float4) * pixel_count, stream);
    }
    if (outputs.depth_confidence) {
        cudaMemsetAsync(outputs.depth_confidence, 0, sizeof(float2) * pixel_count, stream);
    }
    if (outputs.ids) {
        cudaMemsetAsync(outputs.ids, 0, sizeof(GaussianReconstructionSampleIds) * pixel_count, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianFeatureProjectionCounters), stream);
    }
}

void reset_weighted_projection_outputs(const GaussianStateWeightedProjectionLaunch& launch,
                                       const GaussianFeatureProjectionOutputs& outputs,
                                       GaussianStateWeightedProjectionCounters* counters,
                                       cudaStream_t stream) {
    const std::uint32_t pixel_count = launch.width * launch.height;
    if (outputs.radiance) {
        cudaMemsetAsync(outputs.radiance, 0, sizeof(float4) * pixel_count, stream);
    }
    if (outputs.depth_confidence) {
        cudaMemsetAsync(outputs.depth_confidence, 0, sizeof(float2) * pixel_count, stream);
    }
    if (outputs.ids) {
        cudaMemsetAsync(outputs.ids, 0, sizeof(GaussianReconstructionSampleIds) * pixel_count, stream);
    }
    if (counters) {
        cudaMemsetAsync(counters, 0, sizeof(GaussianStateWeightedProjectionCounters), stream);
    }
}

} // namespace

void launch_extract_nvdiffrast_samples(const NvDiffrastRasterLaunch& launch,
                                       const NvDiffrastRasterInputs& inputs,
                                       GaussianReconstructionSample* samples,
                                       std::uint32_t* sample_count,
                                       NvDiffrastExtractCounters* counters,
                                       void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_outputs(sample_count, counters, stream);
    if (!valid_launch(launch, inputs, sample_count) || !samples) {
        return;
    }

    const std::uint32_t total_pixels = launch.width * launch.height * launch.batch;
    const int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1u) / threads);
    extract_nvdiffrast_samples_kernel<<<blocks, threads, 0, stream>>>(
        launch, inputs, samples, sample_count, counters);
}

void launch_extract_nvdiffrast_sample_tensors(const NvDiffrastRasterLaunch& launch,
                                              const NvDiffrastRasterInputs& inputs,
                                              const GaussianReconstructionTensorOutputs& outputs,
                                              std::uint32_t* sample_count,
                                              NvDiffrastExtractCounters* counters,
                                              void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_outputs(sample_count, counters, stream);
    if (!valid_launch(launch, inputs, sample_count)) {
        return;
    }

    const std::uint32_t total_pixels = launch.width * launch.height * launch.batch;
    const int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1u) / threads);
    extract_nvdiffrast_sample_tensors_kernel<<<blocks, threads, 0, stream>>>(
        launch, inputs, outputs, sample_count, counters);
}

void launch_extract_seed_frame_sample_tensors(const GaussianSeedFrameTensorLaunch& launch,
                                              const GaussianSeedFrameTensorInputs& inputs,
                                              const GaussianReconstructionTensorOutputs& outputs,
                                              std::uint32_t* sample_count,
                                              GaussianSeedFrameExtractCounters* counters,
                                              void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_seed_frame_outputs(sample_count, counters, stream);
    if (!valid_seed_frame_launch(launch, inputs, outputs, sample_count)) {
        return;
    }

    const std::uint32_t total_pixels = launch.width * launch.height * launch.batch;
    const int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1u) / threads);
    extract_seed_frame_sample_tensors_kernel<<<blocks, threads, 0, stream>>>(
        launch, inputs, outputs, sample_count, counters);
}

void launch_read_gaussian_sample_count_info(const GaussianSampleCountInfoLaunch& launch,
                                            const std::uint32_t* sample_count,
                                            GaussianSampleCountInfo* output,
                                            void* cuda_stream) {
    if (!output) {
        return;
    }
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    read_gaussian_sample_count_info_kernel<<<1, 1, 0, stream>>>(launch, sample_count, output);
}

void launch_clear_gaussian_reconstruction_state(const GaussianReconstructionState& state,
                                                std::uint32_t gaussian_capacity,
                                                void* cuda_stream) {
    if (!valid_host_state(state) || gaussian_capacity == 0) {
        return;
    }
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    const int threads = 256;
    const int blocks = static_cast<int>((gaussian_capacity + threads - 1u) / threads);
    clear_gaussian_state_kernel<<<blocks, threads, 0, stream>>>(state, gaussian_capacity);
}

void launch_accumulate_gaussian_state_from_samples(const GaussianStateUpdateLaunch& launch,
                                                   const GaussianReconstructionSample* samples,
                                                   const GaussianReconstructionState& state,
                                                   GaussianStateUpdateCounters* counters,
                                                   void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_update_counters(counters, stream);
    if (!valid_update_launch(launch) || !samples || !valid_host_state(state)) {
        return;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((launch.sample_count + threads - 1u) / threads);
    accumulate_gaussian_state_from_samples_kernel<<<blocks, threads, 0, stream>>>(
        launch, samples, state, counters);
}

void launch_accumulate_gaussian_state_from_sample_tensors(const GaussianStateUpdateLaunch& launch,
                                                          const GaussianReconstructionTensorSamples& samples,
                                                          const GaussianReconstructionState& state,
                                                          GaussianStateUpdateCounters* counters,
                                                          void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_update_counters(counters, stream);
    if (!valid_update_launch(launch) || !valid_tensor_samples(samples) || !valid_host_state(state)) {
        return;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((launch.sample_count + threads - 1u) / threads);
    accumulate_gaussian_state_from_tensor_samples_kernel<<<blocks, threads, 0, stream>>>(
        launch, samples, state, counters);
}

void launch_accumulate_gaussian_state_from_sample_tensors_counted(
    const GaussianStateTensorCountUpdateLaunch& launch,
    const GaussianReconstructionTensorSamples& samples,
    const std::uint32_t* sample_count,
    const GaussianReconstructionState& state,
    GaussianStateUpdateCounters* counters,
    void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_update_counters(counters, stream);
    if (!valid_counted_update_launch(launch, sample_count) || !valid_tensor_samples(samples) ||
        !valid_host_state(state)) {
        return;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((launch.max_samples + threads - 1u) / threads);
    accumulate_gaussian_state_from_counted_tensor_samples_kernel<<<blocks, threads, 0, stream>>>(
        launch, samples, sample_count, state, counters);
}

void launch_finalize_gaussian_reconstruction_state(const GaussianReconstructionState& state,
                                                   std::uint32_t gaussian_capacity,
                                                   void* cuda_stream) {
    if (!valid_host_state(state) || gaussian_capacity == 0) {
        return;
    }
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    const int threads = 256;
    const int blocks = static_cast<int>((gaussian_capacity + threads - 1u) / threads);
    finalize_gaussian_state_kernel<<<blocks, threads, 0, stream>>>(state, gaussian_capacity);
}

void launch_build_gaussian_sample_tile_bins(const GaussianSampleTileBinningLaunch& launch,
                                            const GaussianReconstructionTensorSamples& samples,
                                            const std::uint32_t* sample_count,
                                            const GaussianSampleTileBinningOutputs& outputs,
                                            GaussianSampleTileBinningCounters* counters,
                                            void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_tile_binning_outputs(launch, outputs, counters, stream);
    if (!valid_tile_binning_launch(launch, samples, sample_count, outputs)) {
        return;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((launch.max_samples + threads - 1u) / threads);
    build_gaussian_sample_tile_bins_kernel<<<blocks, threads, 0, stream>>>(
        launch, samples, sample_count, outputs, counters);
}

void launch_build_gaussian_sample_tile_offsets(const GaussianSampleTileOffsetLaunch& launch,
                                               const GaussianSampleTileOffsetInputs& inputs,
                                               const GaussianSampleTileOffsetOutputs& outputs,
                                               GaussianSampleTileOffsetCounters* counters,
                                               void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_tile_offset_outputs(launch, outputs, counters, stream);
    if (!valid_tile_offset_launch(launch, inputs, outputs)) {
        return;
    }
    const int threads = 1024;
    const std::size_t shared_bytes = sizeof(std::uint32_t) * threads;
    build_gaussian_sample_tile_offsets_kernel<<<1, threads, shared_bytes, stream>>>(
        launch, inputs, outputs, counters);
}

void launch_compact_gaussian_sample_tile_bins(const GaussianSampleTileCompactionLaunch& launch,
                                              const std::uint32_t* sample_count,
                                              const GaussianSampleTileCompactionInputs& inputs,
                                              const GaussianSampleTileCompactionOutputs& outputs,
                                              GaussianSampleTileCompactionCounters* counters,
                                              void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_tile_compaction_outputs(launch, outputs, counters, stream);
    if (!valid_tile_compaction_launch(launch, sample_count, inputs, outputs)) {
        return;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((launch.max_samples + threads - 1u) / threads);
    compact_gaussian_sample_tile_bins_kernel<<<blocks, threads, 0, stream>>>(
        launch, sample_count, inputs, outputs, counters);
}

void launch_resolve_gaussian_sample_tiles(const GaussianTileSampleResolveLaunch& launch,
                                          const GaussianReconstructionTensorSamples& samples,
                                          const GaussianTileSampleResolveInputs& inputs,
                                          const GaussianTileSampleResolveOutputs& outputs,
                                          GaussianTileSampleResolveCounters* counters,
                                          void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_tile_sample_resolve_outputs(launch, outputs, counters, stream);
    if (!valid_tile_sample_resolve_launch(launch, samples, inputs, outputs)) {
        return;
    }
    const int threads = 128;
    const int blocks = static_cast<int>(launch.tile_count);
    resolve_gaussian_sample_tiles_kernel<<<blocks, threads, 0, stream>>>(
        launch, samples, inputs, outputs, counters);
}

void launch_resolve_gaussian_sample_tiles_weighted(
    const GaussianTileWeightedResolveLaunch& launch,
    const GaussianReconstructionTensorSamples& samples,
    const GaussianTileSampleResolveInputs& inputs,
    const GaussianTileSampleResolveOutputs& outputs,
    GaussianTileWeightedResolveCounters* counters,
    void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_tile_weighted_resolve_outputs(launch, outputs, counters, stream);
    if (!valid_tile_weighted_resolve_launch(launch, samples, inputs, outputs)) {
        return;
    }
    const int accumulate_threads = 128;
    resolve_gaussian_sample_tiles_weighted_accumulate_kernel<<<
        static_cast<int>(launch.tile_count), accumulate_threads, 0, stream>>>(
        launch, samples, inputs, outputs, counters);

    const int normalize_threads = 256;
    const std::uint32_t pixel_count = launch.width * launch.height;
    const int normalize_blocks = static_cast<int>((pixel_count + normalize_threads - 1u) /
                                                  normalize_threads);
    resolve_gaussian_sample_tiles_weighted_normalize_kernel<<<
        normalize_blocks, normalize_threads, 0, stream>>>(launch, outputs, counters);
}

void launch_resolve_gaussian_sample_tiles_weighted_gated(
    const GaussianTileGatedWeightedResolveLaunch& launch,
    const GaussianReconstructionTensorSamples& samples,
    const GaussianTileSampleResolveInputs& inputs,
    const GaussianTileResolveGuides& guides,
    const GaussianTileSampleResolveOutputs& outputs,
    GaussianTileGatedWeightedResolveCounters* counters,
    void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_tile_gated_weighted_resolve_outputs(launch, outputs, counters, stream);
    if (!valid_tile_gated_weighted_resolve_launch(launch, samples, inputs, guides, outputs)) {
        return;
    }
    const int accumulate_threads = 128;
    resolve_gaussian_sample_tiles_weighted_gated_accumulate_kernel<<<
        static_cast<int>(launch.tile_count), accumulate_threads, 0, stream>>>(
        launch, samples, inputs, guides, outputs, counters);

    const int normalize_threads = 256;
    const std::uint32_t pixel_count = launch.width * launch.height;
    const int normalize_blocks = static_cast<int>((pixel_count + normalize_threads - 1u) /
                                                  normalize_threads);
    resolve_gaussian_sample_tiles_weighted_gated_normalize_kernel<<<
        normalize_blocks, normalize_threads, 0, stream>>>(launch, outputs, counters);
}

void launch_project_gaussian_state_features(const GaussianFeatureProjectionLaunch& launch,
                                            const GaussianReconstructionState& state,
                                            const GaussianFeatureProjectionOutputs& outputs,
                                            GaussianFeatureProjectionCounters* counters,
                                            void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_projection_outputs(launch, outputs, counters, stream);
    if (!valid_host_state(state) || !outputs.radiance || !outputs.depth_confidence || !outputs.ids ||
        launch.width == 0 || launch.height == 0 || launch.gaussian_count == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((launch.gaussian_count + threads - 1u) / threads);
    project_gaussian_state_features_kernel<<<blocks, threads, 0, stream>>>(
        launch, state, outputs, counters);
}

void launch_project_gaussian_state_features_weighted(
    const GaussianStateWeightedProjectionLaunch& launch,
    const GaussianReconstructionState& state,
    const GaussianFeatureProjectionOutputs& outputs,
    GaussianStateWeightedProjectionCounters* counters,
    void* cuda_stream) {
    auto* stream = static_cast<cudaStream_t>(cuda_stream);
    reset_weighted_projection_outputs(launch, outputs, counters, stream);
    if (!valid_state_weighted_projection_launch(launch, state, outputs)) {
        return;
    }

    const int accumulate_threads = 256;
    const int accumulate_blocks = static_cast<int>((launch.gaussian_count + accumulate_threads - 1u) /
                                                   accumulate_threads);
    project_gaussian_state_features_weighted_accumulate_kernel<<<
        accumulate_blocks, accumulate_threads, 0, stream>>>(
        launch, state, outputs, counters);

    const int normalize_threads = 256;
    const std::uint32_t pixel_count = launch.width * launch.height;
    const int normalize_blocks = static_cast<int>((pixel_count + normalize_threads - 1u) /
                                                  normalize_threads);
    project_gaussian_state_features_weighted_normalize_kernel<<<
        normalize_blocks, normalize_threads, 0, stream>>>(
        launch, outputs, counters);
}

} // namespace vkgsplat::cuda
