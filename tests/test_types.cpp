// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for core POD types — guards against accidental layout
// changes that would break the .cu / .cpp boundary.

#include <vksplat/types.h>

#include <cstddef>
#include <cstdio>
#include <type_traits>

int main() {
    using namespace vksplat;

    static_assert(std::is_trivially_copyable_v<float3>);
    static_assert(std::is_trivially_copyable_v<float4>);
    static_assert(std::is_trivially_copyable_v<mat4>);

    if (sizeof(mat4) != sizeof(float) * 16) {
        std::fprintf(stderr, "mat4 size mismatch: %zu\n", sizeof(mat4));
        return 1;
    }

    if (Gaussian::sh_coeffs != 16) {
        std::fprintf(stderr, "sh_coeffs mismatch: %d\n", Gaussian::sh_coeffs);
        return 1;
    }
    return 0;
}
