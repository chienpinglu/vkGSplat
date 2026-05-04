// SPDX-License-Identifier: Apache-2.0
//
// ABI sanity checks for the shared CPU/Vulkan/CUDA 3DGS pipeline records.

#include <vkgsplat/gpu_pipeline.h>

#include <cstdio>
#include <type_traits>

int main() {
    using namespace vkgsplat;

    static_assert(std::is_trivially_copyable_v<GpuGaussian>);
    static_assert(std::is_trivially_copyable_v<GpuProjectedSplat>);
    static_assert(std::is_trivially_copyable_v<GpuTileRange>);
    static_assert(std::is_trivially_copyable_v<GpuIndirectParams>);

    if (sizeof(GpuTileRange) != sizeof(std::uint32_t) * 2) {
        std::fprintf(stderr, "GpuTileRange size mismatch\n");
        return 1;
    }
    if (sizeof(GpuIndirectParams) != sizeof(std::uint32_t) * 4) {
        std::fprintf(stderr, "GpuIndirectParams size mismatch\n");
        return 1;
    }
    return 0;
}
