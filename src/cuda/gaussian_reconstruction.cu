// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/cuda/gaussian_reconstruction.h"

#include <cuda_runtime.h>

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

} // namespace vkgsplat::cuda
