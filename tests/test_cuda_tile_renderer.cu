// SPDX-License-Identifier: Apache-2.0
//
// M8: CUDA-only tile renderer comparison. No Vulkan interop here: copy
// a tiny projected-splat/tile-range fixture to CUDA, render, copy back,
// and validate the same ordering invariant as the CPU reference.

#include <vkgsplat/cuda/tile_renderer.h>

#include <cuda_runtime.h>

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

} // namespace

int main() {
    using namespace vkgsplat;
    using namespace vkgsplat::cuda;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices available\n");
        return 77; // CTest skip
    }

    const TileRendererLaunch launch{ 16, 16, 16, 1, 1 };
    const GpuProjectedSplat projected[] = {
        {
            { 8.0f, 8.0f }, 0.75f,
            { 3.0f, 0.0f }, { 0.0f, 3.0f },
            { 1.0f / 9.0f, 0.0f, 1.0f / 9.0f },
            0.98f, { 0.0f, 0.0f, 1.0f }, 0,
        },
        {
            { 8.0f, 8.0f }, 0.50f,
            { 3.0f, 0.0f }, { 0.0f, 3.0f },
            { 1.0f / 9.0f, 0.0f, 1.0f / 9.0f },
            0.98f, { 1.0f, 0.0f, 0.0f }, 1,
        },
    };
    const std::uint32_t sorted_indices[] = { 0, 1 }; // far blue then near red
    const GpuTileRange ranges[] = { { 0, 2 } };
    std::vector<float4> pixels(launch.width * launch.height, { 0.0f, 0.0f, 0.0f, 1.0f });

    GpuProjectedSplat* d_projected = nullptr;
    std::uint32_t* d_indices = nullptr;
    GpuTileRange* d_ranges = nullptr;
    float4* d_pixels = nullptr;

    CHECK_CUDA(cudaMalloc(&d_projected, sizeof(projected)));
    CHECK_CUDA(cudaMalloc(&d_indices, sizeof(sorted_indices)));
    CHECK_CUDA(cudaMalloc(&d_ranges, sizeof(ranges)));
    CHECK_CUDA(cudaMalloc(&d_pixels, sizeof(float4) * pixels.size()));
    CHECK_CUDA(cudaMemcpy(d_projected, projected, sizeof(projected), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, sorted_indices, sizeof(sorted_indices), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ranges, ranges, sizeof(ranges), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pixels, pixels.data(), sizeof(float4) * pixels.size(), cudaMemcpyHostToDevice));

    launch_tile_renderer(launch, d_projected, d_indices, d_ranges, d_pixels, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(pixels.data(), d_pixels, sizeof(float4) * pixels.size(), cudaMemcpyDeviceToHost));

    cudaFree(d_projected);
    cudaFree(d_indices);
    cudaFree(d_ranges);
    cudaFree(d_pixels);

    const float4 center = pixels[8 * launch.width + 8];
    if (!(center.x > 0.35f && center.z > 0.05f && center.x > center.z)) {
        std::fprintf(stderr, "CUDA tile renderer mismatch: center=(%.4f %.4f %.4f %.4f)\n",
                     center.x, center.y, center.z, center.w);
        return 1;
    }

    return 0;
}
