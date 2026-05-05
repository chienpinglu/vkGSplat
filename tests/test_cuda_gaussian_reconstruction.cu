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
    std::uint32_t* d_count = nullptr;
    NvDiffrastExtractCounters* d_counters = nullptr;

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
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_counters, sizeof(NvDiffrastExtractCounters)));

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
    cudaFree(d_count);
    cudaFree(d_counters);

    if (count != 2 || counters.emitted != 2 || counters.background != 2 || counters.invalid != 0) {
        std::fprintf(stderr, "unexpected tensor counters: count=%u emitted=%u background=%u invalid=%u\n",
                     count, counters.emitted, counters.background, counters.invalid);
        return 1;
    }

    const bool first_is_tri2 = tensor_ids[0].primitive_id == 2;
    const std::uint32_t tri2_slot = first_is_tri2 ? 0u : 1u;
    const std::uint32_t tri6_slot = first_is_tri2 ? 1u : 0u;
    if (tensor_ids[tri2_slot].primitive_id != 2 || !near(tensor_position[tri2_slot].z, 4.0f) ||
        !near(tensor_depth_conf[tri2_slot].x, 0.10f) || !near(tensor_depth_conf[tri2_slot].y, 0.8f) ||
        tensor_ids[tri6_slot].primitive_id != 6 || !near(tensor_position[tri6_slot].x, 5.0f) ||
        !near(tensor_depth_conf[tri6_slot].x, 0.20f)) {
        std::fprintf(stderr, "tensorized output mismatch\n");
        return 1;
    }

    return 0;
}
