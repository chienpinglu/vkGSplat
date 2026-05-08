// SPDX-License-Identifier: Apache-2.0
//
// M8: CUDA-only tile renderer comparison. No Vulkan interop here: copy
// a tiny projected-splat/tile-range fixture to CUDA, render, copy back,
// and validate the same ordering invariant as the CPU reference.

#include <vkgsplat/cuda/tile_renderer.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
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

bool near_u8(unsigned char a, unsigned char b, unsigned char eps = 1) {
    const int delta = static_cast<int>(a) - static_cast<int>(b);
    return std::abs(delta) <= static_cast<int>(eps);
}

unsigned char pack_unorm8(float v) {
    const float clamped = std::fmin(std::fmax(v, 0.0f), 1.0f);
    return static_cast<unsigned char>(clamped * 255.0f + 0.5f);
}

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
    std::vector<vkgsplat::float4> pixels(
        launch.width * launch.height, { 0.0f, 0.0f, 0.0f, 1.0f });

    GpuProjectedSplat* d_projected = nullptr;
    std::uint32_t* d_indices = nullptr;
    GpuTileRange* d_ranges = nullptr;
    vkgsplat::float4* d_pixels = nullptr;

    CHECK_CUDA(cudaMalloc(&d_projected, sizeof(projected)));
    CHECK_CUDA(cudaMalloc(&d_indices, sizeof(sorted_indices)));
    CHECK_CUDA(cudaMalloc(&d_ranges, sizeof(ranges)));
    CHECK_CUDA(cudaMalloc(&d_pixels, sizeof(vkgsplat::float4) * pixels.size()));
    CHECK_CUDA(cudaMemcpy(d_projected, projected, sizeof(projected), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, sorted_indices, sizeof(sorted_indices), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ranges, ranges, sizeof(ranges), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pixels, pixels.data(), sizeof(vkgsplat::float4) * pixels.size(), cudaMemcpyHostToDevice));

    launch_tile_renderer(launch, d_projected, d_indices, d_ranges, d_pixels, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(pixels.data(), d_pixels, sizeof(vkgsplat::float4) * pixels.size(), cudaMemcpyDeviceToHost));

    const vkgsplat::float4 corner = pixels[0];
    if (!near(corner.x, 0.0f) || !near(corner.y, 0.0f) ||
        !near(corner.z, 0.0f) || !near(corner.w, 0.0f)) {
        std::fprintf(stderr, "CUDA tile renderer did not clear background: corner=(%.4f %.4f %.4f %.4f)\n",
                     corner.x, corner.y, corner.z, corner.w);
        return 1;
    }

    const vkgsplat::float4 center = pixels[8 * launch.width + 8];
    if (!(center.x > 0.90f && center.z > 0.04f && center.x > center.z)) {
        std::fprintf(stderr, "CUDA tile renderer mismatch: center=(%.4f %.4f %.4f %.4f)\n",
                     center.x, center.y, center.z, center.w);
        return 1;
    }

    cudaArray_t surface_array = nullptr;
    cudaSurfaceObject_t surface = 0;
    const cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    CHECK_CUDA(cudaMallocArray(&surface_array,
                               &channel_desc,
                               launch.width,
                               launch.height,
                               cudaArraySurfaceLoadStore));

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = surface_array;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resource_desc));

    launch_tile_renderer_surface_rgba8(
        launch,
        d_projected,
        d_indices,
        d_ranges,
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(surface)),
        nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uchar4> surface_pixels(launch.width * launch.height);
    CHECK_CUDA(cudaMemcpy2DFromArray(surface_pixels.data(),
                                     launch.width * sizeof(uchar4),
                                     surface_array,
                                     0,
                                     0,
                                     launch.width * sizeof(uchar4),
                                     launch.height,
                                     cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaFreeArray(surface_array));

    const uchar4 surface_corner = surface_pixels[0];
    if (surface_corner.x != 0 || surface_corner.y != 0 ||
        surface_corner.z != 0 || surface_corner.w != 0) {
        std::fprintf(stderr,
                     "CUDA tile surface renderer did not clear background: corner=(%u %u %u %u)\n",
                     static_cast<unsigned>(surface_corner.x),
                     static_cast<unsigned>(surface_corner.y),
                     static_cast<unsigned>(surface_corner.z),
                     static_cast<unsigned>(surface_corner.w));
        return 1;
    }

    const uchar4 surface_center = surface_pixels[8 * launch.width + 8];
    if (!near_u8(surface_center.x, pack_unorm8(center.x), 2) ||
        !near_u8(surface_center.y, pack_unorm8(center.y), 2) ||
        !near_u8(surface_center.z, pack_unorm8(center.z), 2) ||
        !near_u8(surface_center.w, pack_unorm8(center.w), 2)) {
        std::fprintf(stderr,
                     "CUDA tile surface renderer mismatch: surface=(%u %u %u %u) float=(%.4f %.4f %.4f %.4f)\n",
                     static_cast<unsigned>(surface_center.x),
                     static_cast<unsigned>(surface_center.y),
                     static_cast<unsigned>(surface_center.z),
                     static_cast<unsigned>(surface_center.w),
                     center.x,
                     center.y,
                     center.z,
                     center.w);
        return 1;
    }

    cudaFree(d_projected);
    cudaFree(d_indices);
    cudaFree(d_ranges);
    cudaFree(d_pixels);

    return 0;
}
