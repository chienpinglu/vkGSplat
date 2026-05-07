// SPDX-License-Identifier: Apache-2.0
//
// CUDA ingestion fixture for nvdiffrast-style raster outputs. This does
// not depend on Python or torch; it constructs the tensors that
// dr.rasterize() and dr.interpolate() would hand to CUDA.

#include <vkgsplat/cuda/gaussian_reconstruction.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error in %s: %s\n", #expr,              \
                         cudaGetErrorString(err__));                            \
            return 1;                                                           \
        }                                                                       \
    } while (0)

bool near(float a, float b, float eps = 1.0e-5f) {
    return std::abs(a - b) <= eps;
}

} // namespace

int main() {
    using namespace vkgsplat::cuda;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices available\n");
        return 77; // CTest skip
    }

    const std::uint32_t width = 2;
    const std::uint32_t height = 2;
    const std::uint32_t pixel_count = width * height;

    std::vector<vkgsplat::float4> raster(pixel_count, { 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<vkgsplat::float4> color(pixel_count, { 0.0f, 0.0f, 0.0f, 1.0f });
    std::vector<vkgsplat::float3> world(pixel_count, { 0.0f, 0.0f, 0.0f });
    std::vector<vkgsplat::float3> normal(pixel_count, { 0.0f, 0.0f, 1.0f });
    std::vector<vkgsplat::float2> motion(pixel_count, { 0.0f, 0.0f });

    raster[1] = { 0.25f, 0.50f, 0.10f, 3.0f };
    color[1] = { 1.0f, 0.25f, 0.0f, 0.8f };
    world[1] = { 2.0f, 3.0f, 4.0f };
    normal[1] = { 0.0f, 1.0f, 0.0f };
    motion[1] = { -0.5f, 0.25f };

    raster[2] = { 0.75f, 0.10f, 0.20f, 7.0f };
    color[2] = { 0.0f, 0.5f, 1.0f, 1.0f };
    world[2] = { 5.0f, 6.0f, 7.0f };
    normal[2] = { 1.0f, 0.0f, 0.0f };
    motion[2] = { 1.0f, -1.0f };

    vkgsplat::float4* d_raster = nullptr;
    vkgsplat::float4* d_color = nullptr;
    vkgsplat::float3* d_world = nullptr;
    vkgsplat::float3* d_normal = nullptr;
    vkgsplat::float2* d_motion = nullptr;
    GaussianReconstructionSample* d_samples = nullptr;
    vkgsplat::float3* d_tensor_position = nullptr;
    vkgsplat::float3* d_tensor_normal = nullptr;
    vkgsplat::float4* d_tensor_radiance = nullptr;
    vkgsplat::float2* d_tensor_pixel = nullptr;
    vkgsplat::float2* d_tensor_motion = nullptr;
    vkgsplat::float2* d_tensor_bary = nullptr;
    vkgsplat::float2* d_tensor_depth_conf = nullptr;
    GaussianReconstructionSampleIds* d_tensor_ids = nullptr;
    vkgsplat::float2* d_seed_depth_conf = nullptr;
    GaussianReconstructionSampleIds* d_seed_ids = nullptr;
    std::uint32_t* d_sample_tile = nullptr;
    std::uint32_t* d_tile_counts = nullptr;
    std::uint32_t* d_tile_offsets = nullptr;
    std::uint32_t* d_tile_write_counts = nullptr;
    std::uint32_t* d_tile_sample_indices = nullptr;
    GaussianSampleTileBinningCounters* d_tile_counters = nullptr;
    GaussianSampleTileOffsetCounters* d_tile_offset_counters = nullptr;
    GaussianSampleTileCompactionCounters* d_tile_compaction_counters = nullptr;
    GaussianTileSampleResolveCounters* d_tile_resolve_counters = nullptr;
    GaussianTileWeightedResolveCounters* d_tile_weighted_resolve_counters = nullptr;
    std::uint32_t* d_count = nullptr;
    GaussianSampleCountInfo* d_count_info = nullptr;
    NvDiffrastExtractCounters* d_counters = nullptr;
    GaussianSeedFrameExtractCounters* d_seed_counters = nullptr;
    constexpr std::uint32_t gaussian_capacity = 8;
    vkgsplat::float3* d_state_position = nullptr;
    vkgsplat::float3* d_state_normal = nullptr;
    vkgsplat::float4* d_state_radiance = nullptr;
    vkgsplat::float2* d_state_pixel = nullptr;
    vkgsplat::float2* d_state_motion = nullptr;
    vkgsplat::float3* d_state_covariance = nullptr;
    vkgsplat::float2* d_state_depth_conf = nullptr;
    vkgsplat::float2* d_state_mass_variance = nullptr;
    GaussianReconstructionSampleIds* d_state_ids = nullptr;
    GaussianStateUpdateCounters* d_update_counters = nullptr;
    vkgsplat::float4* d_project_radiance = nullptr;
    vkgsplat::float2* d_project_depth_conf = nullptr;
    GaussianReconstructionSampleIds* d_project_ids = nullptr;
    GaussianFeatureProjectionCounters* d_project_counters = nullptr;
    GaussianStateWeightedProjectionCounters* d_state_weighted_projection_counters = nullptr;
    vkgsplat::float2* d_guide_depth_conf = nullptr;
    GaussianReconstructionSampleIds* d_guide_ids = nullptr;
    GaussianTileGatedWeightedResolveCounters* d_tile_gated_resolve_counters = nullptr;

    CHECK_CUDA(cudaMalloc(&d_raster, sizeof(vkgsplat::float4) * raster.size()));
    CHECK_CUDA(cudaMalloc(&d_color, sizeof(vkgsplat::float4) * color.size()));
    CHECK_CUDA(cudaMalloc(&d_world, sizeof(vkgsplat::float3) * world.size()));
    CHECK_CUDA(cudaMalloc(&d_normal, sizeof(vkgsplat::float3) * normal.size()));
    CHECK_CUDA(cudaMalloc(&d_motion, sizeof(vkgsplat::float2) * motion.size()));
    CHECK_CUDA(cudaMalloc(&d_samples, sizeof(GaussianReconstructionSample) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_position, sizeof(vkgsplat::float3) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_normal, sizeof(vkgsplat::float3) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_radiance, sizeof(vkgsplat::float4) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_pixel, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_motion, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_bary, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_depth_conf, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tensor_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_seed_depth_conf, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_seed_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_sample_tile, sizeof(std::uint32_t) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tile_counts, sizeof(std::uint32_t) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tile_offsets, sizeof(std::uint32_t) * (pixel_count + 1u)));
    CHECK_CUDA(cudaMalloc(&d_tile_write_counts, sizeof(std::uint32_t) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tile_sample_indices, sizeof(std::uint32_t) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tile_counters, sizeof(GaussianSampleTileBinningCounters)));
    CHECK_CUDA(cudaMalloc(&d_tile_offset_counters, sizeof(GaussianSampleTileOffsetCounters)));
    CHECK_CUDA(cudaMalloc(&d_tile_compaction_counters, sizeof(GaussianSampleTileCompactionCounters)));
    CHECK_CUDA(cudaMalloc(&d_tile_resolve_counters, sizeof(GaussianTileSampleResolveCounters)));
    CHECK_CUDA(cudaMalloc(&d_tile_weighted_resolve_counters, sizeof(GaussianTileWeightedResolveCounters)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_count_info, sizeof(GaussianSampleCountInfo)));
    CHECK_CUDA(cudaMalloc(&d_counters, sizeof(NvDiffrastExtractCounters)));
    CHECK_CUDA(cudaMalloc(&d_seed_counters, sizeof(GaussianSeedFrameExtractCounters)));
    CHECK_CUDA(cudaMalloc(&d_state_position, sizeof(vkgsplat::float3) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_normal, sizeof(vkgsplat::float3) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_radiance, sizeof(vkgsplat::float4) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_pixel, sizeof(vkgsplat::float2) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_motion, sizeof(vkgsplat::float2) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_covariance, sizeof(vkgsplat::float3) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_depth_conf, sizeof(vkgsplat::float2) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_mass_variance, sizeof(vkgsplat::float2) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_state_ids, sizeof(GaussianReconstructionSampleIds) * gaussian_capacity));
    CHECK_CUDA(cudaMalloc(&d_update_counters, sizeof(GaussianStateUpdateCounters)));
    CHECK_CUDA(cudaMalloc(&d_project_radiance, sizeof(vkgsplat::float4) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_project_depth_conf, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_project_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_project_counters, sizeof(GaussianFeatureProjectionCounters)));
    CHECK_CUDA(cudaMalloc(&d_state_weighted_projection_counters,
                          sizeof(GaussianStateWeightedProjectionCounters)));
    CHECK_CUDA(cudaMalloc(&d_guide_depth_conf, sizeof(vkgsplat::float2) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_guide_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count));
    CHECK_CUDA(cudaMalloc(&d_tile_gated_resolve_counters, sizeof(GaussianTileGatedWeightedResolveCounters)));

    CHECK_CUDA(cudaMemcpy(d_raster, raster.data(), sizeof(vkgsplat::float4) * raster.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_color, color.data(), sizeof(vkgsplat::float4) * color.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_world, world.data(), sizeof(vkgsplat::float3) * world.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_normal, normal.data(), sizeof(vkgsplat::float3) * normal.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_motion, motion.data(), sizeof(vkgsplat::float2) * motion.size(), cudaMemcpyHostToDevice));

    const NvDiffrastRasterLaunch launch{ width, height, 1, pixel_count, 0.0f };
    const NvDiffrastRasterInputs inputs{ d_raster, d_color, d_world, d_normal, d_motion };
    launch_extract_nvdiffrast_samples(launch, inputs, d_samples, d_count, d_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::uint32_t count = 0;
    NvDiffrastExtractCounters counters{};
    std::vector<GaussianReconstructionSample> samples(pixel_count);
    CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(samples.data(), d_samples, sizeof(GaussianReconstructionSample) * samples.size(), cudaMemcpyDeviceToHost));

    if (count != 2 || counters.emitted != 2 || counters.background != 2 || counters.invalid != 0) {
        std::fprintf(stderr, "unexpected AoS counters: count=%u emitted=%u background=%u invalid=%u\n",
                     count, counters.emitted, counters.background, counters.invalid);
        return 1;
    }

    std::sort(samples.begin(), samples.begin() + count,
              [](const GaussianReconstructionSample& a, const GaussianReconstructionSample& b) {
                  return a.primitive_id < b.primitive_id;
              });

    const GaussianReconstructionSample& a = samples[0];
    const GaussianReconstructionSample& b = samples[1];
    constexpr std::uint32_t expected_flags = GAUSSIAN_SAMPLE_HAS_COLOR |
                                             GAUSSIAN_SAMPLE_HAS_WORLD_POSITION |
                                             GAUSSIAN_SAMPLE_HAS_NORMAL |
                                             GAUSSIAN_SAMPLE_HAS_MOTION;
    if (a.primitive_id != 2 || a.pixel_index != 1 || !near(a.pixel.x, 1.5f) || !near(a.pixel.y, 0.5f) ||
        !near(a.depth_ndc, 0.10f) || !near(a.barycentric_uv.x, 0.25f) || !near(a.confidence, 0.8f) ||
        !near(a.position.z, 4.0f) || !near(a.normal.y, 1.0f) || !near(a.motion_px.x, -0.5f) ||
        (a.flags & expected_flags) != expected_flags) {
        std::fprintf(stderr, "sample A mismatch\n");
        return 1;
    }

    if (b.primitive_id != 6 || b.pixel_index != 2 || !near(b.pixel.x, 0.5f) || !near(b.pixel.y, 1.5f) ||
        !near(b.depth_ndc, 0.20f) || !near(b.radiance.z, 1.0f) || !near(b.position.x, 5.0f) ||
        !near(b.normal.x, 1.0f) || !near(b.motion_px.y, -1.0f)) {
        std::fprintf(stderr, "sample B mismatch\n");
        return 1;
    }

    const GaussianReconstructionTensorOutputs tensor_outputs{
        d_tensor_position,
        d_tensor_normal,
        d_tensor_radiance,
        d_tensor_pixel,
        d_tensor_motion,
        d_tensor_bary,
        d_tensor_depth_conf,
        d_tensor_ids,
    };
    launch_extract_nvdiffrast_sample_tensors(launch, inputs, tensor_outputs, d_count, d_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<vkgsplat::float3> tensor_position(pixel_count);
    std::vector<vkgsplat::float2> tensor_depth_conf(pixel_count);
    std::vector<GaussianReconstructionSampleIds> tensor_ids(pixel_count);
    CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tensor_position.data(), d_tensor_position, sizeof(vkgsplat::float3) * tensor_position.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tensor_depth_conf.data(), d_tensor_depth_conf, sizeof(vkgsplat::float2) * tensor_depth_conf.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tensor_ids.data(), d_tensor_ids, sizeof(GaussianReconstructionSampleIds) * tensor_ids.size(), cudaMemcpyDeviceToHost));


    if (count != 2 || counters.emitted != 2 || counters.background != 2 || counters.invalid != 0) {
        std::fprintf(stderr, "unexpected tensor counters: count=%u emitted=%u background=%u invalid=%u\n",
                     count, counters.emitted, counters.background, counters.invalid);
        return 1;
    }

    const bool first_is_tri2 = tensor_ids[0].primitive_id == 2;
    std::uint32_t tri2_slot = first_is_tri2 ? 0u : 1u;
    std::uint32_t tri6_slot = first_is_tri2 ? 1u : 0u;
    if (tensor_ids[tri2_slot].primitive_id != 2 || !near(tensor_position[tri2_slot].z, 4.0f) ||
        !near(tensor_depth_conf[tri2_slot].x, 0.10f) || !near(tensor_depth_conf[tri2_slot].y, 0.8f) ||
        tensor_ids[tri6_slot].primitive_id != 6 || !near(tensor_position[tri6_slot].x, 5.0f) ||
        !near(tensor_depth_conf[tri6_slot].x, 0.20f)) {
        std::fprintf(stderr, "tensorized output mismatch\n");
        return 1;
    }

    std::vector<vkgsplat::float2> seed_depth_conf(pixel_count, { 0.0f, 0.0f });
    std::vector<GaussianReconstructionSampleIds> seed_ids(pixel_count, { 0u, 0u, 0u, 0u });
    seed_depth_conf[1] = { 0.10f, 0.8f };
    seed_depth_conf[2] = { 0.20f, 1.0f };
    seed_ids[1] = { 2u, 1u, 0u, expected_flags };
    seed_ids[2] = { 6u, 2u, 0u, expected_flags };
    CHECK_CUDA(cudaMemcpy(d_seed_depth_conf, seed_depth_conf.data(),
                          sizeof(vkgsplat::float2) * seed_depth_conf.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_seed_ids, seed_ids.data(),
                          sizeof(GaussianReconstructionSampleIds) * seed_ids.size(), cudaMemcpyHostToDevice));

    const GaussianSeedFrameTensorLaunch seed_launch{ width, height, 1u, pixel_count, 0.0f };
    const GaussianSeedFrameTensorInputs seed_inputs{
        d_color,
        d_world,
        d_normal,
        d_motion,
        d_seed_depth_conf,
        d_seed_ids,
    };
    launch_extract_seed_frame_sample_tensors(
        seed_launch, seed_inputs, tensor_outputs, d_count, d_seed_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianSeedFrameExtractCounters seed_counters{};
    CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&seed_counters, d_seed_counters,
                          sizeof(seed_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tensor_position.data(), d_tensor_position,
                          sizeof(vkgsplat::float3) * tensor_position.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tensor_depth_conf.data(), d_tensor_depth_conf,
                          sizeof(vkgsplat::float2) * tensor_depth_conf.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tensor_ids.data(), d_tensor_ids,
                          sizeof(GaussianReconstructionSampleIds) * tensor_ids.size(), cudaMemcpyDeviceToHost));

    if (count != 2 || seed_counters.visited != pixel_count || seed_counters.emitted != 2 ||
        seed_counters.background != 2 || seed_counters.invalid != 0 ||
        seed_counters.low_confidence != 0 || seed_counters.capacity_overflow != 0) {
        std::fprintf(stderr,
                     "seed-buffer extraction counters mismatch: count=%u visited=%u emitted=%u background=%u invalid=%u low=%u overflow=%u\n",
                     count, seed_counters.visited, seed_counters.emitted,
                     seed_counters.background, seed_counters.invalid,
                     seed_counters.low_confidence, seed_counters.capacity_overflow);
        return 1;
    }

    const bool seed_first_is_tri2 = tensor_ids[0].primitive_id == 2;
    const std::uint32_t seed_tri2_slot = seed_first_is_tri2 ? 0u : 1u;
    const std::uint32_t seed_tri6_slot = seed_first_is_tri2 ? 1u : 0u;
    if (tensor_ids[seed_tri2_slot].primitive_id != 2 ||
        tensor_ids[seed_tri2_slot].pixel_index != 1 ||
        !near(tensor_position[seed_tri2_slot].z, 4.0f) ||
        !near(tensor_depth_conf[seed_tri2_slot].x, 0.10f) ||
        !near(tensor_depth_conf[seed_tri2_slot].y, 0.8f) ||
        tensor_ids[seed_tri6_slot].primitive_id != 6 ||
        tensor_ids[seed_tri6_slot].pixel_index != 2 ||
        !near(tensor_position[seed_tri6_slot].x, 5.0f) ||
        !near(tensor_depth_conf[seed_tri6_slot].x, 0.20f)) {
        std::fprintf(stderr, "seed-buffer tensorized output mismatch\n");
        return 1;
    }
    GaussianSampleCountInfo count_info{};
    const GaussianSampleCountInfoLaunch count_info_launch{ pixel_count };
    launch_read_gaussian_sample_count_info(count_info_launch, d_count, d_count_info, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&count_info, d_count_info, sizeof(count_info), cudaMemcpyDeviceToHost));
    if (count_info.raw != 2 || count_info.clamped != 2 || count_info.overflow != 0 ||
        count_info.available != pixel_count - 2u) {
        std::fprintf(stderr,
                     "sample-count info mismatch: raw=%u clamped=%u overflow=%u available=%u\n",
                     count_info.raw, count_info.clamped, count_info.overflow, count_info.available);
        return 1;
    }

    const std::uint32_t overflow_count = pixel_count + 3u;
    CHECK_CUDA(cudaMemcpy(d_count, &overflow_count, sizeof(overflow_count), cudaMemcpyHostToDevice));
    launch_read_gaussian_sample_count_info(count_info_launch, d_count, d_count_info, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&count_info, d_count_info, sizeof(count_info), cudaMemcpyDeviceToHost));
    if (count_info.raw != overflow_count || count_info.clamped != pixel_count ||
        count_info.overflow != 3u || count_info.available != 0u) {
        std::fprintf(stderr,
                     "sample-count overflow info mismatch: raw=%u clamped=%u overflow=%u available=%u\n",
                     count_info.raw, count_info.clamped, count_info.overflow, count_info.available);
        return 1;
    }
    CHECK_CUDA(cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice));

    tri2_slot = seed_tri2_slot;
    tri6_slot = seed_tri6_slot;


    const GaussianReconstructionState state{
        d_state_position,
        d_state_normal,
        d_state_radiance,
        d_state_pixel,
        d_state_motion,
        d_state_covariance,
        d_state_depth_conf,
        d_state_mass_variance,
        d_state_ids,
    };
    const GaussianStateUpdateLaunch update_launch{ count, gaussian_capacity, 0.0f };
    launch_clear_gaussian_reconstruction_state(state, gaussian_capacity, nullptr);
    launch_accumulate_gaussian_state_from_samples(update_launch, d_samples, state, d_update_counters, nullptr);
    launch_finalize_gaussian_reconstruction_state(state, gaussian_capacity, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianStateUpdateCounters update_counters{};
    std::vector<vkgsplat::float3> state_position(gaussian_capacity);
    std::vector<vkgsplat::float3> state_normal(gaussian_capacity);
    std::vector<vkgsplat::float4> state_radiance(gaussian_capacity);
    std::vector<vkgsplat::float2> state_pixel(gaussian_capacity);
    std::vector<vkgsplat::float2> state_depth_conf(gaussian_capacity);
    std::vector<vkgsplat::float2> state_mass_variance(gaussian_capacity);
    std::vector<GaussianReconstructionSampleIds> state_ids(gaussian_capacity);
    CHECK_CUDA(cudaMemcpy(&update_counters, d_update_counters, sizeof(update_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_position.data(), d_state_position, sizeof(vkgsplat::float3) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_normal.data(), d_state_normal, sizeof(vkgsplat::float3) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_radiance.data(), d_state_radiance, sizeof(vkgsplat::float4) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_pixel.data(), d_state_pixel, sizeof(vkgsplat::float2) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_depth_conf.data(), d_state_depth_conf, sizeof(vkgsplat::float2) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_mass_variance.data(), d_state_mass_variance, sizeof(vkgsplat::float2) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_ids.data(), d_state_ids, sizeof(GaussianReconstructionSampleIds) * gaussian_capacity, cudaMemcpyDeviceToHost));

    if (update_counters.visited != 2 || update_counters.updated != 2 || update_counters.invalid != 0 ||
        update_counters.slot_overflow != 0) {
        std::fprintf(stderr, "unexpected AoS update counters: visited=%u updated=%u invalid=%u overflow=%u\n",
                     update_counters.visited, update_counters.updated,
                     update_counters.invalid, update_counters.slot_overflow);
        return 1;
    }

    if (!near(state_position[2].z, 4.0f) || !near(state_normal[2].y, 1.0f) ||
        !near(state_radiance[2].x, 1.0f) || !near(state_pixel[2].x, 1.5f) ||
        !near(state_depth_conf[2].x, 0.10f) || !near(state_depth_conf[2].y, 0.8f) ||
        !near(state_mass_variance[2].x, 0.8f) || !near(state_mass_variance[2].y, 0.0f) ||
        state_ids[2].primitive_id != 2 || !near(state_position[6].x, 5.0f) ||
        !near(state_normal[6].x, 1.0f) || !near(state_radiance[6].z, 1.0f) ||
        !near(state_depth_conf[6].x, 0.20f) || !near(state_depth_conf[6].y, 1.0f) ||
        !near(state_mass_variance[6].x, 1.0f) || state_ids[6].primitive_id != 6) {
        std::fprintf(stderr, "finalized AoS Gaussian state mismatch\n");
        return 1;
    }

    const GaussianFeatureProjectionLaunch projection_launch{ width, height, gaussian_capacity, 0.0f };
    const GaussianFeatureProjectionOutputs projection_outputs{
        d_project_radiance,
        d_project_depth_conf,
        d_project_ids,
    };
    launch_project_gaussian_state_features(
        projection_launch, state, projection_outputs, d_project_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianFeatureProjectionCounters project_counters{};
    std::vector<vkgsplat::float4> projected_radiance(pixel_count);
    std::vector<vkgsplat::float2> projected_depth_conf(pixel_count);
    std::vector<GaussianReconstructionSampleIds> projected_ids(pixel_count);
    CHECK_CUDA(cudaMemcpy(&project_counters, d_project_counters, sizeof(project_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance, sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf, sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_ids.data(), d_project_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count, cudaMemcpyDeviceToHost));

    if (project_counters.visited != gaussian_capacity || project_counters.emitted != 2 ||
        project_counters.skipped_empty != gaussian_capacity - 2 || project_counters.out_of_bounds != 0) {
        std::fprintf(stderr, "unexpected projection counters: visited=%u emitted=%u empty=%u oob=%u\n",
                     project_counters.visited, project_counters.emitted,
                     project_counters.skipped_empty, project_counters.out_of_bounds);
        return 1;
    }

    if (projected_ids[1].primitive_id != 2 || !near(projected_radiance[1].x, 1.0f) ||
        !near(projected_radiance[1].w, 0.8f) || !near(projected_depth_conf[1].x, 0.10f) ||
        projected_ids[2].primitive_id != 6 || !near(projected_radiance[2].z, 1.0f) ||
        !near(projected_radiance[2].w, 1.0f) || !near(projected_depth_conf[2].x, 0.20f)) {
        std::fprintf(stderr, "projected Gaussian features mismatch\n");
        return 1;
    }

    const GaussianStateWeightedProjectionLaunch weighted_projection_launch{
        width, height, gaussian_capacity, 0u, 1.0f, 0.0f, 0.0f
    };
    launch_project_gaussian_state_features_weighted(
        weighted_projection_launch, state, projection_outputs,
        d_state_weighted_projection_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianStateWeightedProjectionCounters weighted_project_counters{};
    CHECK_CUDA(cudaMemcpy(&weighted_project_counters, d_state_weighted_projection_counters,
                          sizeof(weighted_project_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance, sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf, sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_ids.data(), d_project_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count, cudaMemcpyDeviceToHost));

    if (weighted_project_counters.visited != gaussian_capacity ||
        weighted_project_counters.emitted_taps != 2 ||
        weighted_project_counters.skipped_empty != gaussian_capacity - 2 ||
        weighted_project_counters.out_of_bounds != 0 ||
        weighted_project_counters.invalid != 0 ||
        weighted_project_counters.normalized_pixels != 2 ||
        weighted_project_counters.zero_weight_pixels != 2 ||
        projected_ids[1].primitive_id != 2 || !near(projected_radiance[1].x, 1.0f) ||
        !near(projected_radiance[1].w, 0.8f) || !near(projected_depth_conf[1].x, 0.10f) ||
        projected_ids[2].primitive_id != 6 || !near(projected_radiance[2].z, 1.0f) ||
        !near(projected_radiance[2].w, 1.0f) || !near(projected_depth_conf[2].x, 0.20f)) {
        std::fprintf(stderr,
                     "weighted Gaussian state projection mismatch: visited=%u taps=%u empty=%u oob=%u invalid=%u norm=%u zero=%u\n",
                     weighted_project_counters.visited,
                     weighted_project_counters.emitted_taps,
                     weighted_project_counters.skipped_empty,
                     weighted_project_counters.out_of_bounds,
                     weighted_project_counters.invalid,
                     weighted_project_counters.normalized_pixels,
                     weighted_project_counters.zero_weight_pixels);
        return 1;
    }

    const GaussianStateWeightedProjectionLaunch wide_weighted_projection_launch{
        width, height, gaussian_capacity, 1u, 1.0f, 0.0f, 0.0f
    };
    launch_project_gaussian_state_features_weighted(
        wide_weighted_projection_launch, state, projection_outputs,
        d_state_weighted_projection_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&weighted_project_counters, d_state_weighted_projection_counters,
                          sizeof(weighted_project_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance,
                          sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf,
                          sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));

    const float state_wa_center = 0.8f;
    const float state_wa_edge = 0.8f * static_cast<float>(std::exp(-0.5f));
    const float state_wa_diag = 0.8f * static_cast<float>(std::exp(-1.0f));
    const float state_wb_center = 1.0f;
    const float state_wb_edge = static_cast<float>(std::exp(-0.5f));
    const float state_wb_diag = static_cast<float>(std::exp(-1.0f));
    const auto state_red = [](float wa, float wb) { return wa / (wa + wb); };
    const auto state_blue = [](float wa, float wb) { return wb / (wa + wb); };
    const auto state_depth = [](float wa, float wb) {
        return (0.10f * wa + 0.20f * wb) / (wa + wb);
    };

    if (weighted_project_counters.visited != gaussian_capacity ||
        weighted_project_counters.emitted_taps != 8 ||
        weighted_project_counters.skipped_empty != gaussian_capacity - 2 ||
        weighted_project_counters.out_of_bounds != 10 ||
        weighted_project_counters.invalid != 0 ||
        weighted_project_counters.normalized_pixels != 4 ||
        weighted_project_counters.zero_weight_pixels != 0 ||
        !near(projected_radiance[0].x, state_red(state_wa_edge, state_wb_edge), 1.0e-4f) ||
        !near(projected_radiance[1].x, state_red(state_wa_center, state_wb_diag), 1.0e-4f) ||
        !near(projected_radiance[1].z, state_blue(state_wa_center, state_wb_diag), 1.0e-4f) ||
        !near(projected_depth_conf[1].x, state_depth(state_wa_center, state_wb_diag), 1.0e-4f) ||
        !near(projected_radiance[2].x, state_red(state_wa_diag, state_wb_center), 1.0e-4f) ||
        !near(projected_depth_conf[2].x, state_depth(state_wa_diag, state_wb_center), 1.0e-4f)) {
        std::fprintf(stderr,
                     "wide weighted Gaussian state projection mismatch: visited=%u taps=%u empty=%u oob=%u invalid=%u norm=%u zero=%u\n",
                     weighted_project_counters.visited,
                     weighted_project_counters.emitted_taps,
                     weighted_project_counters.skipped_empty,
                     weighted_project_counters.out_of_bounds,
                     weighted_project_counters.invalid,
                     weighted_project_counters.normalized_pixels,
                     weighted_project_counters.zero_weight_pixels);
        return 1;
    }

    const GaussianFeatureProjectionLaunch midpoint_projection_launch{
        width, height, gaussian_capacity, 0.0f, 0.5f
    };
    launch_project_gaussian_state_features(
        midpoint_projection_launch, state, projection_outputs, d_project_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&project_counters, d_project_counters, sizeof(project_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance, sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf, sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_ids.data(), d_project_ids, sizeof(GaussianReconstructionSampleIds) * pixel_count, cudaMemcpyDeviceToHost));

    if (project_counters.emitted != 2 || projected_ids[1].primitive_id != 2 ||
        !near(projected_radiance[1].w, 0.8f) || projected_ids[3].primitive_id != 6 ||
        !near(projected_radiance[3].z, 1.0f) || !near(projected_depth_conf[3].x, 0.20f)) {
        std::fprintf(stderr, "motion-alpha Gaussian projection mismatch\n");
        return 1;
    }

    const GaussianReconstructionTensorSamples tensor_samples{
        d_tensor_position,
        d_tensor_normal,
        d_tensor_radiance,
        d_tensor_pixel,
        d_tensor_motion,
        d_tensor_bary,
        d_tensor_depth_conf,
        d_tensor_ids,
    };
    const GaussianSampleTileBinningLaunch tile_binning_launch{ pixel_count, width, height, 1u, 1u };
    const GaussianSampleTileBinningOutputs tile_binning_outputs{ d_sample_tile, d_tile_counts };
    launch_build_gaussian_sample_tile_bins(
        tile_binning_launch, tensor_samples, d_count, tile_binning_outputs, d_tile_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianSampleTileBinningCounters tile_counters{};
    std::vector<std::uint32_t> sample_tiles(pixel_count);
    std::vector<std::uint32_t> tile_counts(pixel_count);
    CHECK_CUDA(cudaMemcpy(&tile_counters, d_tile_counters, sizeof(tile_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sample_tiles.data(), d_sample_tile, sizeof(std::uint32_t) * sample_tiles.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tile_counts.data(), d_tile_counts, sizeof(std::uint32_t) * tile_counts.size(), cudaMemcpyDeviceToHost));

    if (tile_counters.visited != 2 || tile_counters.binned != 2 || tile_counters.invalid != 0 ||
        tile_counters.out_of_bounds != 0 || sample_tiles[tri2_slot] != 1u || sample_tiles[tri6_slot] != 2u ||
        tile_counts[1] != 1u || tile_counts[2] != 1u) {
        std::fprintf(stderr,
                     "tensorized tile binning mismatch: visited=%u binned=%u invalid=%u oob=%u tiles=(%u,%u) counts=(%u,%u)\n",
                     tile_counters.visited, tile_counters.binned, tile_counters.invalid,
                     tile_counters.out_of_bounds, sample_tiles[tri2_slot], sample_tiles[tri6_slot],
                     tile_counts[1], tile_counts[2]);
        return 1;
    }

    const GaussianSampleTileOffsetLaunch tile_offset_launch{ pixel_count };
    const GaussianSampleTileOffsetInputs tile_offset_inputs{ d_tile_counts };
    const GaussianSampleTileOffsetOutputs tile_offset_outputs{ d_tile_offsets };
    launch_build_gaussian_sample_tile_offsets(
        tile_offset_launch, tile_offset_inputs, tile_offset_outputs,
        d_tile_offset_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianSampleTileOffsetCounters tile_offset_counters{};
    std::vector<std::uint32_t> tile_offsets(pixel_count + 1u);
    CHECK_CUDA(cudaMemcpy(&tile_offset_counters, d_tile_offset_counters,
                          sizeof(tile_offset_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tile_offsets.data(), d_tile_offsets,
                          sizeof(std::uint32_t) * tile_offsets.size(), cudaMemcpyDeviceToHost));
    if (tile_offset_counters.tile_count != pixel_count ||
        tile_offset_counters.total_samples != 2u ||
        tile_offsets[0] != 0u || tile_offsets[1] != 0u ||
        tile_offsets[2] != 1u || tile_offsets[3] != 2u ||
        tile_offsets[4] != 2u) {
        std::fprintf(stderr,
                     "tensorized tile offset mismatch: tiles=%u total=%u offsets=(%u,%u,%u,%u,%u)\n",
                     tile_offset_counters.tile_count, tile_offset_counters.total_samples,
                     tile_offsets[0], tile_offsets[1], tile_offsets[2],
                     tile_offsets[3], tile_offsets[4]);
        return 1;
    }

    const GaussianSampleTileCompactionLaunch tile_compaction_launch{ pixel_count, pixel_count };
    const GaussianSampleTileCompactionInputs tile_compaction_inputs{ d_sample_tile, d_tile_offsets };
    const GaussianSampleTileCompactionOutputs tile_compaction_outputs{
        d_tile_write_counts,
        d_tile_sample_indices,
    };
    launch_compact_gaussian_sample_tile_bins(
        tile_compaction_launch, d_count, tile_compaction_inputs, tile_compaction_outputs,
        d_tile_compaction_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianSampleTileCompactionCounters compaction_counters{};
    std::vector<std::uint32_t> tile_write_counts(pixel_count);
    std::vector<std::uint32_t> tile_sample_indices(pixel_count);
    CHECK_CUDA(cudaMemcpy(&compaction_counters, d_tile_compaction_counters,
                          sizeof(compaction_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tile_write_counts.data(), d_tile_write_counts,
                          sizeof(std::uint32_t) * tile_write_counts.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tile_sample_indices.data(), d_tile_sample_indices,
                          sizeof(std::uint32_t) * tile_sample_indices.size(), cudaMemcpyDeviceToHost));

    if (compaction_counters.visited != 2 || compaction_counters.compacted != 2 ||
        compaction_counters.invalid_tile != 0 || compaction_counters.capacity_overflow != 0 ||
        tile_write_counts[1] != 1u || tile_write_counts[2] != 1u ||
        tile_sample_indices[0] != tri2_slot || tile_sample_indices[1] != tri6_slot) {
        std::fprintf(stderr,
                     "tensorized tile compaction mismatch: visited=%u compacted=%u invalid=%u overflow=%u writes=(%u,%u) indices=(%u,%u)\n",
                     compaction_counters.visited, compaction_counters.compacted,
                     compaction_counters.invalid_tile, compaction_counters.capacity_overflow,
                     tile_write_counts[1], tile_write_counts[2],
                     tile_sample_indices[0], tile_sample_indices[1]);
        return 1;
    }

    const GaussianTileSampleResolveLaunch tile_resolve_launch{
        width, height, 1u, 1u, pixel_count, pixel_count, 0.0f
    };
    const GaussianTileSampleResolveInputs tile_resolve_inputs{ d_tile_offsets, d_tile_sample_indices };
    const GaussianTileSampleResolveOutputs tile_resolve_outputs{
        d_project_radiance,
        d_project_depth_conf,
        d_project_ids,
    };
    launch_resolve_gaussian_sample_tiles(
        tile_resolve_launch, tensor_samples, tile_resolve_inputs, tile_resolve_outputs,
        d_tile_resolve_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianTileSampleResolveCounters resolve_counters{};
    CHECK_CUDA(cudaMemcpy(&resolve_counters, d_tile_resolve_counters,
                          sizeof(resolve_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance,
                          sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf,
                          sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_ids.data(), d_project_ids,
                          sizeof(GaussianReconstructionSampleIds) * pixel_count, cudaMemcpyDeviceToHost));

    if (resolve_counters.visited_tiles != pixel_count || resolve_counters.empty_tiles != 2 ||
        resolve_counters.visited_samples != 2 || resolve_counters.emitted != 2 ||
        resolve_counters.invalid != 0 || resolve_counters.out_of_bounds != 0 ||
        projected_ids[1].primitive_id != 2 || !near(projected_radiance[1].x, 1.0f) ||
        !near(projected_radiance[1].w, 0.8f) || projected_ids[2].primitive_id != 6 ||
        !near(projected_radiance[2].z, 1.0f) || !near(projected_depth_conf[2].x, 0.20f)) {
        std::fprintf(stderr,
                     "tile-local sample resolve mismatch: tiles=%u empty=%u samples=%u emitted=%u invalid=%u oob=%u\n",
                     resolve_counters.visited_tiles, resolve_counters.empty_tiles,
                     resolve_counters.visited_samples, resolve_counters.emitted,
                     resolve_counters.invalid, resolve_counters.out_of_bounds);
        return 1;
    }

    const GaussianTileWeightedResolveLaunch weighted_resolve_launch{
        width, height, 1u, 1u, pixel_count, pixel_count, 0u, 1.0f, 0.0f
    };
    launch_resolve_gaussian_sample_tiles_weighted(
        weighted_resolve_launch, tensor_samples, tile_resolve_inputs, tile_resolve_outputs,
        d_tile_weighted_resolve_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianTileWeightedResolveCounters weighted_counters{};
    CHECK_CUDA(cudaMemcpy(&weighted_counters, d_tile_weighted_resolve_counters,
                          sizeof(weighted_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance,
                          sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf,
                          sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_ids.data(), d_project_ids,
                          sizeof(GaussianReconstructionSampleIds) * pixel_count, cudaMemcpyDeviceToHost));

    if (weighted_counters.visited_tiles != pixel_count || weighted_counters.empty_tiles != 2 ||
        weighted_counters.visited_samples != 2 || weighted_counters.emitted_taps != 2 ||
        weighted_counters.normalized_pixels != 2 || weighted_counters.zero_weight_pixels != 2 ||
        weighted_counters.invalid != 0 || weighted_counters.out_of_bounds != 0 ||
        projected_ids[1].primitive_id != 2 || !near(projected_radiance[1].x, 1.0f) ||
        !near(projected_radiance[1].w, 0.8f) || projected_ids[2].primitive_id != 6 ||
        !near(projected_radiance[2].z, 1.0f) || !near(projected_depth_conf[2].x, 0.20f)) {
        std::fprintf(stderr,
                     "weighted tile resolve mismatch: tiles=%u empty=%u samples=%u taps=%u norm=%u zero=%u invalid=%u oob=%u\n",
                     weighted_counters.visited_tiles, weighted_counters.empty_tiles,
                     weighted_counters.visited_samples, weighted_counters.emitted_taps,
                     weighted_counters.normalized_pixels, weighted_counters.zero_weight_pixels,
                     weighted_counters.invalid, weighted_counters.out_of_bounds);
        return 1;
    }

    const GaussianTileWeightedResolveLaunch wide_weighted_resolve_launch{
        width, height, 1u, 1u, pixel_count, pixel_count, 1u, 1.0f, 0.0f
    };
    launch_resolve_gaussian_sample_tiles_weighted(
        wide_weighted_resolve_launch, tensor_samples, tile_resolve_inputs, tile_resolve_outputs,
        d_tile_weighted_resolve_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&weighted_counters, d_tile_weighted_resolve_counters,
                          sizeof(weighted_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance,
                          sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf,
                          sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));

    const float wa_center = 0.8f;
    const float wa_edge = 0.8f * static_cast<float>(std::exp(-0.5f));
    const float wa_diag = 0.8f * static_cast<float>(std::exp(-1.0f));
    const float wb_center = 1.0f;
    const float wb_edge = static_cast<float>(std::exp(-0.5f));
    const float wb_diag = static_cast<float>(std::exp(-1.0f));
    const auto red = [](float wa, float wb) { return wa / (wa + wb); };
    const auto blue = [](float wa, float wb) { return wb / (wa + wb); };
    const auto depth = [](float wa, float wb) {
        return (0.10f * wa + 0.20f * wb) / (wa + wb);
    };

    if (weighted_counters.visited_tiles != pixel_count || weighted_counters.empty_tiles != 2 ||
        weighted_counters.visited_samples != 2 || weighted_counters.emitted_taps != 8 ||
        weighted_counters.normalized_pixels != 4 || weighted_counters.zero_weight_pixels != 0 ||
        weighted_counters.invalid != 0 || weighted_counters.out_of_bounds != 10 ||
        !near(projected_radiance[0].x, red(wa_edge, wb_edge), 1.0e-4f) ||
        !near(projected_radiance[1].x, red(wa_center, wb_diag), 1.0e-4f) ||
        !near(projected_radiance[1].z, blue(wa_center, wb_diag), 1.0e-4f) ||
        !near(projected_depth_conf[1].x, depth(wa_center, wb_diag), 1.0e-4f) ||
        !near(projected_radiance[2].x, red(wa_diag, wb_center), 1.0e-4f) ||
        !near(projected_depth_conf[2].x, depth(wa_diag, wb_center), 1.0e-4f)) {
        std::fprintf(stderr,
                     "wide weighted tile resolve mismatch: tiles=%u empty=%u samples=%u taps=%u norm=%u zero=%u invalid=%u oob=%u\n",
                     weighted_counters.visited_tiles, weighted_counters.empty_tiles,
                     weighted_counters.visited_samples, weighted_counters.emitted_taps,
                     weighted_counters.normalized_pixels, weighted_counters.zero_weight_pixels,
                     weighted_counters.invalid, weighted_counters.out_of_bounds);
        return 1;
    }

    std::vector<vkgsplat::float2> guide_depth_conf(pixel_count, { 0.0f, 0.0f });
    std::vector<GaussianReconstructionSampleIds> guide_ids(pixel_count, { 0u, 0u, 0u, 0u });
    guide_depth_conf[1] = { 0.10f, 0.8f };
    guide_depth_conf[2] = { 0.20f, 1.0f };
    guide_ids[1] = { 2u, 1u, 0u, expected_flags };
    guide_ids[2] = { 6u, 2u, 0u, expected_flags };
    CHECK_CUDA(cudaMemcpy(d_guide_depth_conf, guide_depth_conf.data(),
                          sizeof(vkgsplat::float2) * guide_depth_conf.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_guide_ids, guide_ids.data(),
                          sizeof(GaussianReconstructionSampleIds) * guide_ids.size(), cudaMemcpyHostToDevice));

    const GaussianTileResolveGuides primitive_depth_guides{
        d_guide_depth_conf,
        d_guide_ids,
        nullptr,
        nullptr,
    };
    const GaussianTileGatedWeightedResolveLaunch gated_resolve_launch{
        width,
        height,
        1u,
        1u,
        pixel_count,
        pixel_count,
        1u,
        GAUSSIAN_TILE_GATE_PRIMITIVE_ID | GAUSSIAN_TILE_GATE_DEPTH,
        1.0f,
        0.0f,
        1.0e-3f,
        0.5f,
        1.0f,
    };
    launch_resolve_gaussian_sample_tiles_weighted_gated(
        gated_resolve_launch, tensor_samples, tile_resolve_inputs, primitive_depth_guides,
        tile_resolve_outputs, d_tile_gated_resolve_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    GaussianTileGatedWeightedResolveCounters gated_counters{};
    CHECK_CUDA(cudaMemcpy(&gated_counters, d_tile_gated_resolve_counters,
                          sizeof(gated_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_radiance.data(), d_project_radiance,
                          sizeof(vkgsplat::float4) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_depth_conf.data(), d_project_depth_conf,
                          sizeof(vkgsplat::float2) * pixel_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(projected_ids.data(), d_project_ids,
                          sizeof(GaussianReconstructionSampleIds) * pixel_count, cudaMemcpyDeviceToHost));

    if (gated_counters.visited_tiles != pixel_count || gated_counters.empty_tiles != 2 ||
        gated_counters.visited_samples != 2 || gated_counters.emitted_taps != 2 ||
        gated_counters.normalized_pixels != 2 || gated_counters.zero_weight_pixels != 2 ||
        gated_counters.out_of_bounds != 10 || gated_counters.rejected_primitive != 6 ||
        gated_counters.rejected_depth != 0 || gated_counters.invalid != 0 ||
        projected_ids[1].primitive_id != 2 || !near(projected_radiance[1].x, 1.0f) ||
        !near(projected_depth_conf[1].x, 0.10f) ||
        projected_ids[2].primitive_id != 6 || !near(projected_radiance[2].z, 1.0f) ||
        !near(projected_depth_conf[2].x, 0.20f)) {
        std::fprintf(stderr,
                     "gated weighted resolve mismatch: tiles=%u empty=%u samples=%u taps=%u norm=%u zero=%u oob=%u prim=%u depth=%u invalid=%u\n",
                     gated_counters.visited_tiles, gated_counters.empty_tiles,
                     gated_counters.visited_samples, gated_counters.emitted_taps,
                     gated_counters.normalized_pixels, gated_counters.zero_weight_pixels,
                     gated_counters.out_of_bounds, gated_counters.rejected_primitive,
                     gated_counters.rejected_depth, gated_counters.invalid);
        return 1;
    }

    guide_depth_conf[1] = { 0.50f, 0.8f };
    CHECK_CUDA(cudaMemcpy(d_guide_depth_conf, guide_depth_conf.data(),
                          sizeof(vkgsplat::float2) * guide_depth_conf.size(), cudaMemcpyHostToDevice));
    const GaussianTileGatedWeightedResolveLaunch depth_reject_launch{
        width,
        height,
        1u,
        1u,
        pixel_count,
        pixel_count,
        0u,
        GAUSSIAN_TILE_GATE_PRIMITIVE_ID | GAUSSIAN_TILE_GATE_DEPTH,
        1.0f,
        0.0f,
        1.0e-3f,
        0.5f,
        1.0f,
    };
    launch_resolve_gaussian_sample_tiles_weighted_gated(
        depth_reject_launch, tensor_samples, tile_resolve_inputs, primitive_depth_guides,
        tile_resolve_outputs, d_tile_gated_resolve_counters, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&gated_counters, d_tile_gated_resolve_counters,
                          sizeof(gated_counters), cudaMemcpyDeviceToHost));

    if (gated_counters.emitted_taps != 1 || gated_counters.rejected_depth != 1 ||
        gated_counters.rejected_primitive != 0 || gated_counters.normalized_pixels != 1 ||
        gated_counters.zero_weight_pixels != 3 || gated_counters.out_of_bounds != 0) {
        std::fprintf(stderr,
                     "gated depth rejection mismatch: taps=%u depth=%u prim=%u norm=%u zero=%u oob=%u\n",
                     gated_counters.emitted_taps, gated_counters.rejected_depth,
                     gated_counters.rejected_primitive, gated_counters.normalized_pixels,
                     gated_counters.zero_weight_pixels, gated_counters.out_of_bounds);
        return 1;
    }

    const GaussianStateTensorCountUpdateLaunch counted_update_launch{ pixel_count, gaussian_capacity, 0.0f };
    launch_clear_gaussian_reconstruction_state(state, gaussian_capacity, nullptr);
    launch_accumulate_gaussian_state_from_sample_tensors_counted(
        counted_update_launch, tensor_samples, d_count, state, d_update_counters, nullptr);
    launch_finalize_gaussian_reconstruction_state(state, gaussian_capacity, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&update_counters, d_update_counters, sizeof(update_counters), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_position.data(), d_state_position, sizeof(vkgsplat::float3) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_radiance.data(), d_state_radiance, sizeof(vkgsplat::float4) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_depth_conf.data(), d_state_depth_conf, sizeof(vkgsplat::float2) * gaussian_capacity, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(state_ids.data(), d_state_ids, sizeof(GaussianReconstructionSampleIds) * gaussian_capacity, cudaMemcpyDeviceToHost));

    if (update_counters.visited != 2 || update_counters.updated != 2 ||
        state_ids[2].primitive_id != 2 || !near(state_position[2].z, 4.0f) ||
        !near(state_radiance[2].x, 1.0f) || !near(state_depth_conf[2].y, 0.8f) ||
        state_ids[6].primitive_id != 6 || !near(state_position[6].x, 5.0f) ||
        !near(state_radiance[6].z, 1.0f) || !near(state_depth_conf[6].y, 1.0f)) {
        std::fprintf(stderr, "tensorized Gaussian state update mismatch\n");
        return 1;
    }

    cudaFree(d_raster);
    cudaFree(d_color);
    cudaFree(d_world);
    cudaFree(d_normal);
    cudaFree(d_motion);
    cudaFree(d_samples);
    cudaFree(d_tensor_position);
    cudaFree(d_tensor_normal);
    cudaFree(d_tensor_radiance);
    cudaFree(d_tensor_pixel);
    cudaFree(d_tensor_motion);
    cudaFree(d_tensor_bary);
    cudaFree(d_tensor_depth_conf);
    cudaFree(d_tensor_ids);
    cudaFree(d_seed_depth_conf);
    cudaFree(d_seed_ids);
    cudaFree(d_sample_tile);
    cudaFree(d_tile_counts);
    cudaFree(d_tile_offsets);
    cudaFree(d_tile_write_counts);
    cudaFree(d_tile_sample_indices);
    cudaFree(d_tile_counters);
    cudaFree(d_tile_offset_counters);
    cudaFree(d_tile_compaction_counters);
    cudaFree(d_tile_resolve_counters);
    cudaFree(d_tile_weighted_resolve_counters);
    cudaFree(d_count);
    cudaFree(d_count_info);
    cudaFree(d_counters);
    cudaFree(d_seed_counters);
    cudaFree(d_state_position);
    cudaFree(d_state_normal);
    cudaFree(d_state_radiance);
    cudaFree(d_state_pixel);
    cudaFree(d_state_motion);
    cudaFree(d_state_covariance);
    cudaFree(d_state_depth_conf);
    cudaFree(d_state_mass_variance);
    cudaFree(d_state_ids);
    cudaFree(d_update_counters);
    cudaFree(d_project_radiance);
    cudaFree(d_project_depth_conf);
    cudaFree(d_project_ids);
    cudaFree(d_project_counters);
    cudaFree(d_state_weighted_projection_counters);
    cudaFree(d_guide_depth_conf);
    cudaFree(d_guide_ids);
    cudaFree(d_tile_gated_resolve_counters);

    return 0;
}
