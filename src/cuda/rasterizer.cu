// SPDX-License-Identifier: Apache-2.0
//
// CUDA backend for vkGSplat: 3D Gaussian Splatting forward rasterizer.
//
// Pipeline (Kerbl et al. 2023, with SDG-flavoured tweaks):
//
//   preprocess_kernel   per-Gaussian: cull, project mean, build 2D conic,
//                       compute screen-space tile coverage.
//   sort_pairs          production target: radix-sort
//                       (key,value) = (depth|tile_id, gaussian_id).
//   identify_ranges     production target: scan to find each tile's
//                       (start,end) span. M1 currently uses count/scatter
//                       kernels with a CUB exclusive scan; production will
//                       replace tile-local insertion sorting with radix or
//                       cooperative tile sorting.
//   blend_kernel        per-tile, per-pixel front-to-back alpha blend.
//
// The implementation is being moved stage-by-stage from a host oracle
// to CUDA. The tile compositor is device-side; preprocessing is now a
// real device kernel; tile-list construction has an M0 deterministic
// device path with fixed per-tile capacity and an M1 compact path with
// CUB-scanned tile offsets. The next production step is depth-key sorting.

#include "vkgsplat/cuda/rasterizer.h"
#include "vkgsplat/cuda/tile_renderer.h"
#include "vkgsplat/tile_raster.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace vkgsplat::cuda {

namespace {

#define VKGSPLAT_CUDA_CHECK(expr)                                       \
    do {                                                               \
        cudaError_t err__ = (expr);                                    \
        if (err__ != cudaSuccess) {                                    \
            throw std::runtime_error(std::string{"CUDA error in "} +   \
                #expr + ": " + cudaGetErrorString(err__));             \
        }                                                              \
    } while (0)

// Device-side per-Gaussian preprocessed state. Two-pass: fill, then
// the binning pass derives tile spans. Layout-of-arrays would cut
// memory traffic; the AoS form here is for clarity.
struct PreprocessedGaussian {
    float2 mean_screen;     // pixel-space mean
    float3 conic;           // 2D inverse covariance (a, b, c)
    float  alpha;
    float  depth;
    float3 color;           // SH-evaluated RGB at this view
    int    tile_min_x, tile_min_y;
    int    tile_max_x, tile_max_y; // exclusive
    std::uint32_t splat_index;
};

struct DeviceMat4 {
    float m[16]{};
};

struct Mat3 {
    float m[3][3]{};
};

DeviceMat4 to_device_mat4(const mat4& src) {
    DeviceMat4 dst{};
    for (int i = 0; i < 16; ++i) dst.m[i] = src.m[static_cast<std::size_t>(i)];
    return dst;
}

__device__ float clamp01_device(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ float sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float4 mul_device(const DeviceMat4& m, float4 v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w,
    };
}

__device__ float3 normalize3_device(float3 v) {
    const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 0.0f) return { 0.0f, 0.0f, 0.0f };
    return { v.x / len, v.y / len, v.z / len };
}

__device__ float2 normalize2_device(float2 v) {
    const float len = sqrtf(v.x * v.x + v.y * v.y);
    if (len <= 0.0f) return { 1.0f, 0.0f };
    return { v.x / len, v.y / len };
}

__device__ Mat3 rotation_from_quat_device(float4 q_in) {
    const float len = sqrtf(q_in.x * q_in.x + q_in.y * q_in.y +
                            q_in.z * q_in.z + q_in.w * q_in.w);
    const float inv_len = len > 0.0f ? 1.0f / len : 1.0f;
    const float x = len > 0.0f ? q_in.x * inv_len : 0.0f;
    const float y = len > 0.0f ? q_in.y * inv_len : 0.0f;
    const float z = len > 0.0f ? q_in.z * inv_len : 0.0f;
    const float w = len > 0.0f ? q_in.w * inv_len : 1.0f;

    Mat3 r{};
    r.m[0][0] = 1.0f - 2.0f * (y * y + z * z);
    r.m[0][1] = 2.0f * (x * y - z * w);
    r.m[0][2] = 2.0f * (x * z + y * w);
    r.m[1][0] = 2.0f * (x * y + z * w);
    r.m[1][1] = 1.0f - 2.0f * (x * x + z * z);
    r.m[1][2] = 2.0f * (y * z - x * w);
    r.m[2][0] = 2.0f * (x * z - y * w);
    r.m[2][1] = 2.0f * (y * z + x * w);
    r.m[2][2] = 1.0f - 2.0f * (x * x + y * y);
    return r;
}

__device__ Mat3 covariance_world_device(const Gaussian& g) {
    const Mat3 r = rotation_from_quat_device(g.rotation);
    const float s[3] = {
        expf(g.scale_log.x),
        expf(g.scale_log.y),
        expf(g.scale_log.z),
    };

    Mat3 cov{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            for (int k = 0; k < 3; ++k) {
                cov.m[row][col] += r.m[row][k] * s[k] * s[k] * r.m[col][k];
            }
        }
    }
    return cov;
}

__device__ Mat3 linear_view_matrix_device(const DeviceMat4& view) {
    Mat3 a{};
    a.m[0][0] = view.m[0]; a.m[0][1] = view.m[4]; a.m[0][2] = view.m[8];
    a.m[1][0] = view.m[1]; a.m[1][1] = view.m[5]; a.m[1][2] = view.m[9];
    a.m[2][0] = view.m[2]; a.m[2][1] = view.m[6]; a.m[2][2] = view.m[10];
    return a;
}

__device__ Mat3 mul_aba_t_device(const Mat3& a, const Mat3& b) {
    Mat3 tmp{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) tmp.m[r][c] += a.m[r][k] * b.m[k][c];
        }
    }

    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) out.m[r][c] += tmp.m[r][k] * a.m[c][k];
        }
    }
    return out;
}

__device__ bool projected_covariance_2d_device(const Gaussian& g,
                                               const DeviceMat4& view_matrix,
                                               float4 view,
                                               float fx,
                                               float fy,
                                               float& a,
                                               float& b,
                                               float& c) {
    const float depth = -view.z;
    if (depth <= 1.0e-4f) return false;

    const Mat3 cov_view = mul_aba_t_device(linear_view_matrix_device(view_matrix),
                                           covariance_world_device(g));

    const float inv_depth = 1.0f / depth;
    const float j00 = fx * inv_depth;
    const float j02 = fx * view.x * inv_depth * inv_depth;
    const float j11 = fy * inv_depth;
    const float j12 = fy * view.y * inv_depth * inv_depth;

    a = j00 * j00 * cov_view.m[0][0] +
        2.0f * j00 * j02 * cov_view.m[0][2] +
        j02 * j02 * cov_view.m[2][2];
    b = j00 * j11 * cov_view.m[0][1] +
        j00 * j12 * cov_view.m[0][2] +
        j02 * j11 * cov_view.m[2][1] +
        j02 * j12 * cov_view.m[2][2];
    c = j11 * j11 * cov_view.m[1][1] +
        2.0f * j11 * j12 * cov_view.m[1][2] +
        j12 * j12 * cov_view.m[2][2];

    constexpr float min_variance = 0.25f;
    a += min_variance;
    c += min_variance;
    return isfinite(a) && isfinite(b) && isfinite(c);
}

__device__ bool ellipse_basis_device(float a,
                                     float b,
                                     float c,
                                     float extent_sigma,
                                     float2& basis_u,
                                     float2& basis_v,
                                     float3& conic) {
    const float det = a * c - b * b;
    if (!(det > 1.0e-8f)) return false;

    conic = { c / det, -b / det, a / det };

    const float trace = a + c;
    const float disc = sqrtf(fmaxf(0.0f, (a - c) * (a - c) + 4.0f * b * b));
    const float l0 = fmaxf(1.0e-6f, 0.5f * (trace + disc));
    const float l1 = fmaxf(1.0e-6f, 0.5f * (trace - disc));

    float2 e0{};
    if (fabsf(b) > 1.0e-6f) {
        e0 = normalize2_device({ b, l0 - a });
    } else {
        e0 = (a >= c) ? float2{ 1.0f, 0.0f } : float2{ 0.0f, 1.0f };
    }
    const float2 e1{ -e0.y, e0.x };

    basis_u = { e0.x * extent_sigma * sqrtf(l0), e0.y * extent_sigma * sqrtf(l0) };
    basis_v = { e1.x * extent_sigma * sqrtf(l1), e1.y * extent_sigma * sqrtf(l1) };
    return true;
}

__device__ float3 sh_color_device(const Gaussian& g, float3 view_dir) {
    constexpr float c0 = 0.28209479177387814f;
    constexpr float c1 = 0.4886025119029199f;
    constexpr float c2[] = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f,
    };
    constexpr float c3[] = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f,
    };

    const float x = view_dir.x;
    const float y = view_dir.y;
    const float z = view_dir.z;
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float basis[Gaussian::sh_coeffs] = {
        c0,
        -c1 * y,
        c1 * z,
        -c1 * x,
        c2[0] * x * y,
        c2[1] * y * z,
        c2[2] * (2.0f * zz - xx - yy),
        c2[3] * x * z,
        c2[4] * (xx - yy),
        c3[0] * y * (3.0f * xx - yy),
        c3[1] * x * y * z,
        c3[2] * y * (4.0f * zz - xx - yy),
        c3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy),
        c3[4] * x * (4.0f * zz - xx - yy),
        c3[5] * z * (xx - yy),
        c3[6] * x * (xx - 3.0f * yy),
    };

    const auto* sh = reinterpret_cast<const float3*>(&g.sh);
    float3 color{ 0.5f, 0.5f, 0.5f };
    for (int i = 0; i < Gaussian::sh_coeffs; ++i) {
        color.x += basis[i] * sh[i].x;
        color.y += basis[i] * sh[i].y;
        color.z += basis[i] * sh[i].z;
    }
    return { clamp01_device(color.x), clamp01_device(color.y), clamp01_device(color.z) };
}

__device__ PreprocessedGaussian invalid_preprocessed(std::uint32_t splat_index) {
    PreprocessedGaussian p{};
    p.alpha = 0.0f;
    p.tile_min_x = 1;
    p.tile_min_y = 1;
    p.tile_max_x = 0;
    p.tile_max_y = 0;
    p.splat_index = splat_index;
    return p;
}

__device__ GpuProjectedSplat invalid_projected(std::uint32_t splat_index) {
    GpuProjectedSplat p{};
    p.opacity = 0.0f;
    p.splat_index = splat_index;
    return p;
}

__device__ int clamp_int_device(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ bool preprocessed_touches_tile(const PreprocessedGaussian& p, int tile_x, int tile_y) {
    return p.alpha > 0.0f &&
           tile_x >= p.tile_min_x && tile_x < p.tile_max_x &&
           tile_y >= p.tile_min_y && tile_y < p.tile_max_y;
}

} // namespace

// ---------------------------------------------------------------------
// CUDA kernels (forward declarations / launch glue)
// ---------------------------------------------------------------------

__global__ void preprocess_kernel(int num_gaussians,
                                  const Gaussian* __restrict__ gaussians,
                                  PreprocessedGaussian* __restrict__ out,
                                  GpuProjectedSplat* __restrict__ projected,
                                  RasterizerFrameStats* __restrict__ stats,
                                  DeviceMat4 view_matrix,
                                  DeviceMat4 projection_matrix,
                                  int image_w,
                                  int image_h,
                                  int tile_size,
                                  float extent_sigma,
                                  int max_radius_pixels)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= num_gaussians) return;

    const auto splat_index = static_cast<std::uint32_t>(index);
    const Gaussian& g = gaussians[index];
    PreprocessedGaussian p = invalid_preprocessed(splat_index);
    GpuProjectedSplat q = invalid_projected(splat_index);

    const float4 view = mul_device(
        view_matrix, { g.position.x, g.position.y, g.position.z, 1.0f });
    const float4 clip = mul_device(projection_matrix, view);
    if (clip.w <= 0.0f) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    const float inv_w = 1.0f / clip.w;
    const float ndc_x = clip.x * inv_w;
    const float ndc_y = clip.y * inv_w;
    const float ndc_z = clip.z * inv_w;
    if (ndc_z < 0.0f || ndc_z > 1.0f) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    const float half_w = static_cast<float>(image_w) * 0.5f;
    const float half_h = static_cast<float>(image_h) * 0.5f;
    const float fx = fabsf(projection_matrix.m[0]) * half_w;
    const float fy = fabsf(projection_matrix.m[5]) * half_h;

    float cov_a = 0.0f;
    float cov_b = 0.0f;
    float cov_c = 0.0f;
    if (!projected_covariance_2d_device(g, view_matrix, view, fx, fy,
                                        cov_a, cov_b, cov_c)) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    float2 basis_u{};
    float2 basis_v{};
    float3 conic{};
    if (!ellipse_basis_device(cov_a, cov_b, cov_c, extent_sigma,
                              basis_u, basis_v, conic)) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    const float2 center = {
        (ndc_x * 0.5f + 0.5f) * static_cast<float>(image_w),
        (ndc_y * 0.5f + 0.5f) * static_cast<float>(image_h),
    };
    const float extent_x = fabsf(basis_u.x) + fabsf(basis_v.x);
    const float extent_y = fabsf(basis_u.y) + fabsf(basis_v.y);
    if (max_radius_pixels > 0 &&
        (extent_x > static_cast<float>(max_radius_pixels) ||
         extent_y > static_cast<float>(max_radius_pixels))) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    const float min_x = fmaxf(0.0f, center.x - extent_x);
    const float min_y = fmaxf(0.0f, center.y - extent_y);
    const float max_x = fminf(static_cast<float>(image_w), center.x + extent_x);
    const float max_y = fminf(static_cast<float>(image_h), center.y + extent_y);
    if (!(min_x < max_x && min_y < max_y)) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    const int tiles_x = (image_w + tile_size - 1) / tile_size;
    const int tiles_y = (image_h + tile_size - 1) / tile_size;
    const int tile_min_x = clamp_int_device(static_cast<int>(floorf(min_x / tile_size)), 0, tiles_x);
    const int tile_min_y = clamp_int_device(static_cast<int>(floorf(min_y / tile_size)), 0, tiles_y);
    const int tile_max_x = clamp_int_device(static_cast<int>(ceilf(max_x / tile_size)), 0, tiles_x);
    const int tile_max_y = clamp_int_device(static_cast<int>(ceilf(max_y / tile_size)), 0, tiles_y);
    if (!(tile_min_x < tile_max_x && tile_min_y < tile_max_y)) {
        out[index] = p;
        projected[index] = q;
        return;
    }

    const float3 view_dir = normalize3_device({ -view.x, -view.y, -view.z });
    p.mean_screen = center;
    p.conic = conic;
    p.alpha = clamp01_device(sigmoid_device(g.opacity_logit));
    p.depth = fmaxf(1.0e-4f, -view.z);
    p.color = sh_color_device(g, view_dir);
    p.tile_min_x = tile_min_x;
    p.tile_min_y = tile_min_y;
    p.tile_max_x = tile_max_x;
    p.tile_max_y = tile_max_y;
    p.splat_index = splat_index;
    q.center_px = center;
    q.depth = p.depth;
    q.basis_u_px = basis_u;
    q.basis_v_px = basis_v;
    q.conic = conic;
    q.opacity = p.alpha;
    q.color = p.color;
    q.splat_index = splat_index;
    if (stats) {
        atomicAdd(&stats->projected_splats, 1u);
    }
    out[index] = p;
    projected[index] = q;
}

__global__ void build_fixed_tile_lists_kernel(int num_gaussians,
                                              const PreprocessedGaussian* __restrict__ preprocessed,
                                              const GpuProjectedSplat* __restrict__ projected,
                                              std::uint32_t* __restrict__ sorted_projected_indices,
                                              GpuTileRange* __restrict__ tile_ranges,
                                              RasterizerFrameStats* __restrict__ stats,
                                              int tiles_x,
                                              int tiles_y,
                                              int max_splats_per_tile) {
    const int tile = static_cast<int>(blockIdx.x);
    const int tile_count = tiles_x * tiles_y;
    if (tile >= tile_count || threadIdx.x != 0) return;

    const int tile_x = tile % tiles_x;
    const int tile_y = tile / tiles_x;
    const std::uint32_t offset =
        static_cast<std::uint32_t>(tile) * static_cast<std::uint32_t>(max_splats_per_tile);
    std::uint32_t count = 0;

    for (int i = 0; i < num_gaussians; ++i) {
        const PreprocessedGaussian p = preprocessed[i];
        if (!preprocessed_touches_tile(p, tile_x, tile_y)) {
            continue;
        }
        if (count >= static_cast<std::uint32_t>(max_splats_per_tile)) {
            if (stats) {
                atomicAdd(&stats->tile_splat_overflow, 1u);
            }
            continue;
        }

        std::uint32_t insert_at = count;
        const float depth = p.depth;
        while (insert_at > 0) {
            const std::uint32_t prev_index = sorted_projected_indices[offset + insert_at - 1u];
            const float prev_depth = projected[prev_index].depth;
            if (prev_depth >= depth) break;
            sorted_projected_indices[offset + insert_at] = prev_index;
            --insert_at;
        }
        sorted_projected_indices[offset + insert_at] = static_cast<std::uint32_t>(i);
        ++count;
    }

    tile_ranges[tile] = { offset, count };
    if (stats) {
        if (count > 0u) {
            atomicAdd(&stats->nonempty_tiles, 1u);
        }
        atomicAdd(&stats->tile_splat_entries, count);
        atomicMax(&stats->max_tile_splats, count);
    }
}


__global__ void count_compact_tile_entries_kernel(int num_gaussians,
                                                  const PreprocessedGaussian* __restrict__ preprocessed,
                                                  std::uint32_t* __restrict__ tile_counts,
                                                  int tiles_x,
                                                  int tiles_y) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= num_gaussians) return;

    const PreprocessedGaussian p = preprocessed[index];
    if (p.alpha <= 0.0f) return;

    const int min_x = clamp_int_device(p.tile_min_x, 0, tiles_x);
    const int max_x = clamp_int_device(p.tile_max_x, 0, tiles_x);
    const int min_y = clamp_int_device(p.tile_min_y, 0, tiles_y);
    const int max_y = clamp_int_device(p.tile_max_y, 0, tiles_y);
    for (int tile_y = min_y; tile_y < max_y; ++tile_y) {
        for (int tile_x = min_x; tile_x < max_x; ++tile_x) {
            const std::uint32_t tile =
                static_cast<std::uint32_t>(tile_y * tiles_x + tile_x);
            atomicAdd(&tile_counts[tile], 1u);
        }
    }
}

__global__ void scatter_compact_tile_entries_kernel(int num_gaussians,
                                                    const PreprocessedGaussian* __restrict__ preprocessed,
                                                    std::uint32_t* __restrict__ sorted_projected_indices,
                                                    const std::uint32_t* __restrict__ tile_offsets,
                                                    std::uint32_t* __restrict__ tile_write_counts,
                                                    int tiles_x,
                                                    int tiles_y) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= num_gaussians) return;

    const PreprocessedGaussian p = preprocessed[index];
    if (p.alpha <= 0.0f) return;

    const int min_x = clamp_int_device(p.tile_min_x, 0, tiles_x);
    const int max_x = clamp_int_device(p.tile_max_x, 0, tiles_x);
    const int min_y = clamp_int_device(p.tile_min_y, 0, tiles_y);
    const int max_y = clamp_int_device(p.tile_max_y, 0, tiles_y);
    for (int tile_y = min_y; tile_y < max_y; ++tile_y) {
        for (int tile_x = min_x; tile_x < max_x; ++tile_x) {
            const std::uint32_t tile =
                static_cast<std::uint32_t>(tile_y * tiles_x + tile_x);
            const std::uint32_t local = atomicAdd(&tile_write_counts[tile], 1u);
            sorted_projected_indices[tile_offsets[tile] + local] = static_cast<std::uint32_t>(index);
        }
    }
}

__global__ void finalize_compact_tile_ranges_kernel(std::uint32_t tile_count,
                                                    const GpuProjectedSplat* __restrict__ projected,
                                                    std::uint32_t* __restrict__ sorted_projected_indices,
                                                    const std::uint32_t* __restrict__ tile_offsets,
                                                    GpuTileRange* __restrict__ tile_ranges,
                                                    RasterizerFrameStats* __restrict__ stats) {
    const std::uint32_t tile = blockIdx.x;
    if (tile >= tile_count || threadIdx.x != 0) return;

    const std::uint32_t begin = tile_offsets[tile];
    const std::uint32_t end = tile_offsets[tile + 1u];
    const std::uint32_t count = end - begin;

    for (std::uint32_t i = begin + 1u; i < end; ++i) {
        const std::uint32_t key = sorted_projected_indices[i];
        const float key_depth = projected[key].depth;
        std::uint32_t j = i;
        while (j > begin) {
            const std::uint32_t prev = sorted_projected_indices[j - 1u];
            const float prev_depth = projected[prev].depth;
            if (prev_depth > key_depth ||
                (prev_depth == key_depth && prev <= key)) {
                break;
            }
            sorted_projected_indices[j] = prev;
            --j;
        }
        sorted_projected_indices[j] = key;
    }

    tile_ranges[tile] = { begin, count };
    if (stats) {
        if (count > 0u) {
            atomicAdd(&stats->nonempty_tiles, 1u);
        }
        atomicAdd(&stats->tile_splat_entries, count);
        atomicMax(&stats->max_tile_splats, count);
    }
}

__device__ unsigned char to_unorm8_device(float v) {
    const float clamped = fminf(fmaxf(v, 0.0f), 1.0f);
    return static_cast<unsigned char>(clamped * 255.0f + 0.5f);
}

__global__ void clear_rgba32f_kernel(std::size_t pixel_count,
                                     float4 color,
                                     float4* __restrict__ dst) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;
    dst[index] = color;
}

__global__ void clear_surface_rgba8_kernel(std::uint32_t width,
                                           std::uint32_t height,
                                           float4 color,
                                           cudaSurfaceObject_t surface) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uchar4 packed{
        to_unorm8_device(color.x),
        to_unorm8_device(color.y),
        to_unorm8_device(color.z),
        to_unorm8_device(color.w),
    };
    surf2Dwrite(packed, surface, x * sizeof(uchar4), y);
}

__global__ void pack_rgba8_kernel(std::size_t pixel_count,
                                  const float4* __restrict__ src,
                                  unsigned char* __restrict__ dst) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;

    const float4 c = src[index];
    dst[index * 4u + 0u] = to_unorm8_device(c.x);
    dst[index * 4u + 1u] = to_unorm8_device(c.y);
    dst[index * 4u + 2u] = to_unorm8_device(c.z);
    dst[index * 4u + 3u] = to_unorm8_device(c.w);
}

__global__ void pack_rgba16f_kernel(std::size_t pixel_count,
                                    const float4* __restrict__ src,
                                    unsigned short* __restrict__ dst) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;

    const float4 c = src[index];
    dst[index * 4u + 0u] = __half_as_ushort(__float2half_rn(c.x));
    dst[index * 4u + 1u] = __half_as_ushort(__float2half_rn(c.y));
    dst[index * 4u + 2u] = __half_as_ushort(__float2half_rn(c.z));
    dst[index * 4u + 3u] = __half_as_ushort(__float2half_rn(c.w));
}

__global__ void unpack_rgba8_kernel(std::size_t pixel_count,
                                    const unsigned char* __restrict__ src,
                                    float4* __restrict__ dst) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;

    dst[index] = {
        static_cast<float>(src[index * 4u + 0u]) / 255.0f,
        static_cast<float>(src[index * 4u + 1u]) / 255.0f,
        static_cast<float>(src[index * 4u + 2u]) / 255.0f,
        static_cast<float>(src[index * 4u + 3u]) / 255.0f,
    };
}

__global__ void unpack_rgba16f_kernel(std::size_t pixel_count,
                                      const unsigned short* __restrict__ src,
                                      float4* __restrict__ dst) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;

    dst[index] = {
        __half2float(__ushort_as_half(src[index * 4u + 0u])),
        __half2float(__ushort_as_half(src[index * 4u + 1u])),
        __half2float(__ushort_as_half(src[index * 4u + 2u])),
        __half2float(__ushort_as_half(src[index * 4u + 3u])),
    };
}

// ---------------------------------------------------------------------
// RasterizerImpl — opaque device state for the public Rasterizer class
// ---------------------------------------------------------------------

class RasterizerImpl {
public:
    RasterizerImpl() {
        VKGSPLAT_CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~RasterizerImpl() {
        if (gaussians_dev_) cudaFree(gaussians_dev_);
        if (preprocessed_dev_) cudaFree(preprocessed_dev_);
        if (projected_dev_) cudaFree(projected_dev_);
        if (sorted_indices_dev_) cudaFree(sorted_indices_dev_);
        if (tile_ranges_dev_) cudaFree(tile_ranges_dev_);
        if (tile_counts_dev_) cudaFree(tile_counts_dev_);
        if (tile_offsets_dev_) cudaFree(tile_offsets_dev_);
        if (tile_write_counts_dev_) cudaFree(tile_write_counts_dev_);
        if (compact_scan_temp_dev_) cudaFree(compact_scan_temp_dev_);
        if (stats_dev_) cudaFree(stats_dev_);
        if (output_dev_) cudaFree(output_dev_);
        if (output_rgba8_dev_) cudaFree(output_rgba8_dev_);
        if (output_rgba16_dev_) cudaFree(output_rgba16_dev_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void bind_to_uuid(const std::array<unsigned char, 16>& uuid) {
        int count = 0;
        VKGSPLAT_CUDA_CHECK(cudaGetDeviceCount(&count));
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp p{};
            VKGSPLAT_CUDA_CHECK(cudaGetDeviceProperties(&p, i));
            if (std::memcmp(p.uuid.bytes, uuid.data(), 16) == 0) {
                VKGSPLAT_CUDA_CHECK(cudaSetDevice(i));
                std::fprintf(stderr, "[vkgsplat][cuda] bound to device %d (%s)\n", i, p.name);
                return;
            }
        }
        throw std::runtime_error("cuda::Rasterizer: no CUDA device matches Vulkan UUID");
    }

    void upload(const Scene& scene) {
        if (gaussians_dev_) {
            cudaFree(gaussians_dev_);
            gaussians_dev_ = nullptr;
        }
        gaussian_count_ = scene.size();
        if (gaussian_count_ == 0) {
            if (preprocessed_dev_) {
                cudaFree(preprocessed_dev_);
                preprocessed_dev_ = nullptr;
            }
            return;
        }

        const std::size_t bytes = sizeof(Gaussian) * gaussian_count_;
        VKGSPLAT_CUDA_CHECK(cudaMalloc(&gaussians_dev_, bytes));
        VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(gaussians_dev_,
                                           scene.gaussians().data(),
                                           bytes,
                                           cudaMemcpyHostToDevice,
                                           stream_));

        if (preprocessed_dev_) cudaFree(preprocessed_dev_);
        VKGSPLAT_CUDA_CHECK(cudaMalloc(&preprocessed_dev_,
                                      sizeof(PreprocessedGaussian) * gaussian_count_));
    }

    FrameId render(const Camera& camera,
                   const RenderParams& params,
                   const RenderTarget& target) {
        ++frame_;

        if (target.kind != RenderTargetKind::HOST_BUFFER &&
            target.kind != RenderTargetKind::INTEROP_IMAGE) {
            throw std::runtime_error("cuda::Rasterizer: unsupported render target kind");
        }
        if (!target.user_handle) {
            throw std::runtime_error("cuda::Rasterizer: render target requires user_handle");
        }

        ImageDesc desc = target.desc;
        if (desc.width == 0) desc.width = camera.width();
        if (desc.height == 0) desc.height = camera.height();
        if (desc.width == 0 || desc.height == 0) {
            throw std::runtime_error("cuda::Rasterizer: target dimensions must be nonzero");
        }

        if (gaussian_count_ == 0) {
            last_frame_stats_ = {};
            clear_empty_target(desc, params, target);
            return frame_;
        }

        if (tunables_.tile_size <= 0) {
            throw std::runtime_error("cuda::Rasterizer: tile_size must be positive");
        }
        const std::uint64_t tile_pixels =
            static_cast<std::uint64_t>(tunables_.tile_size) *
            static_cast<std::uint64_t>(tunables_.tile_size);
        if (tile_pixels > 1024u) {
            throw std::runtime_error("cuda::Rasterizer: tile_size squared exceeds CUDA block size");
        }
        if (!tunables_.use_compact_tile_lists && tunables_.max_splats_per_tile <= 0) {
            throw std::runtime_error("cuda::Rasterizer: max_splats_per_tile must be positive");
        }

        constexpr float splat_extent_sigma = 3.0f;
        const auto tile_size = static_cast<std::uint32_t>(tunables_.tile_size);
        const TileGrid grid = make_tile_grid(desc, tile_size);
        const std::size_t tile_count = static_cast<std::size_t>(grid.tiles_x) * grid.tiles_y;
        if (tile_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("cuda::Rasterizer: tile count exceeds uint32 offsets");
        }
        if (tile_count + 1u > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("cuda::Rasterizer: compact scan dimensions exceed int range");
        }
        if (gaussian_count_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            desc.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            desc.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            grid.tiles_x > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            grid.tiles_y > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("cuda::Rasterizer: launch dimensions exceed int range");
        }
        std::size_t initial_index_capacity = 1u;
        if (!tunables_.use_compact_tile_lists) {
            const auto max_splats_per_tile =
                static_cast<std::uint32_t>(tunables_.max_splats_per_tile);
            initial_index_capacity = tile_count * static_cast<std::size_t>(max_splats_per_tile);
            if (initial_index_capacity >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error(
                    "cuda::Rasterizer: fixed tile-list capacity exceeds uint32 offsets");
            }
        }
        ensure_device_capacity(gaussian_count_,
                               initial_index_capacity,
                               tile_count,
                               static_cast<std::size_t>(desc.width) * desc.height);
        last_frame_stats_ = {};
        VKGSPLAT_CUDA_CHECK(cudaMemsetAsync(stats_dev_, 0, sizeof(RasterizerFrameStats), stream_));

        const int threads = 256;
        const int blocks  = (static_cast<int>(gaussian_count_) + threads - 1) / threads;
        const DeviceMat4 view_dev = to_device_mat4(camera.view());
        const DeviceMat4 projection_dev = to_device_mat4(camera.projection());

        preprocess_kernel<<<blocks, threads, 0, stream_>>>(
            static_cast<int>(gaussian_count_),
            reinterpret_cast<const Gaussian*>(gaussians_dev_),
            preprocessed_dev_,
            projected_dev_,
            stats_dev_,
            view_dev,
            projection_dev,
            static_cast<int>(desc.width),
            static_cast<int>(desc.height),
            tunables_.tile_size,
            splat_extent_sigma,
            tunables_.max_radii_pixels);
        VKGSPLAT_CUDA_CHECK(cudaGetLastError());

        if (tunables_.use_compact_tile_lists) {
            ensure_compact_tile_aux_capacity(tile_count);
            VKGSPLAT_CUDA_CHECK(cudaMemsetAsync(tile_counts_dev_,
                                                0,
                                                sizeof(std::uint32_t) * (tile_count + 1u),
                                                stream_));
            count_compact_tile_entries_kernel<<<blocks, threads, 0, stream_>>>(
                static_cast<int>(gaussian_count_),
                preprocessed_dev_,
                tile_counts_dev_,
                static_cast<int>(grid.tiles_x),
                static_cast<int>(grid.tiles_y));
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());

            const auto tile_count_u32 = static_cast<std::uint32_t>(tile_count);
            const int scan_items = static_cast<int>(tile_count + 1u);
            ensure_compact_scan_temp_capacity(scan_items);
            VKGSPLAT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                compact_scan_temp_dev_,
                compact_scan_temp_capacity_bytes_,
                tile_counts_dev_,
                tile_offsets_dev_,
                scan_items,
                stream_));

            std::uint32_t compact_index_count = 0;
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(&compact_index_count,
                                                tile_offsets_dev_ + tile_count,
                                                sizeof(compact_index_count),
                                                cudaMemcpyDeviceToHost,
                                                stream_));
            VKGSPLAT_CUDA_CHECK(cudaStreamSynchronize(stream_));
            ensure_sorted_index_capacity(compact_index_count > 0u ? compact_index_count : 1u);
            VKGSPLAT_CUDA_CHECK(cudaMemsetAsync(tile_write_counts_dev_,
                                                0,
                                                sizeof(std::uint32_t) * tile_count,
                                                stream_));
            if (compact_index_count > 0u) {
                scatter_compact_tile_entries_kernel<<<blocks, threads, 0, stream_>>>(
                    static_cast<int>(gaussian_count_),
                    preprocessed_dev_,
                    sorted_indices_dev_,
                    tile_offsets_dev_,
                    tile_write_counts_dev_,
                    static_cast<int>(grid.tiles_x),
                    static_cast<int>(grid.tiles_y));
                VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            }
            finalize_compact_tile_ranges_kernel<<<static_cast<unsigned int>(tile_count), 1, 0, stream_>>>(
                tile_count_u32,
                projected_dev_,
                sorted_indices_dev_,
                tile_offsets_dev_,
                tile_ranges_dev_,
                stats_dev_);
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
        } else {
            build_fixed_tile_lists_kernel<<<static_cast<unsigned int>(tile_count), 1, 0, stream_>>>(
                static_cast<int>(gaussian_count_),
                preprocessed_dev_,
                projected_dev_,
                sorted_indices_dev_,
                tile_ranges_dev_,
                stats_dev_,
                static_cast<int>(grid.tiles_x),
                static_cast<int>(grid.tiles_y),
                tunables_.max_splats_per_tile);
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
        }

        const TileRendererLaunch tile_launch{
            desc.width,
            desc.height,
            tile_size,
            grid.tiles_x,
            grid.tiles_y,
            { params.background.x, params.background.y, params.background.z, 0.0f },
            1.0e-3f,
            params.clear_to_background ? TILE_RENDERER_CLEAR_OUTPUT : 0u,
        };
        if (target.kind == RenderTargetKind::HOST_BUFFER) {
            if (!params.clear_to_background) {
                prime_output_from_target(desc, target);
            }
            launch_tile_renderer(tile_launch, projected_dev_, sorted_indices_dev_,
                                 tile_ranges_dev_, output_dev_, stream_);
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            copy_output_to_target(desc, target);
        } else {
            launch_surface_output(desc, tile_launch, target);
        }
        VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(&last_frame_stats_,
                                           stats_dev_,
                                           sizeof(RasterizerFrameStats),
                                           cudaMemcpyDeviceToHost,
                                           stream_));

        return frame_;
    }

    void wait(FrameId /*frame*/) {
        VKGSPLAT_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (last_frame_stats_.tile_splat_overflow > 0 && !reported_tile_overflow_) {
            std::fprintf(stderr,
                         "[vkgsplat][cuda] fixed tile-list dropped %u tile/splat entries; "
                         "increase max_splats_per_tile or replace M0 binning with count/scan/scatter\n",
                         last_frame_stats_.tile_splat_overflow);
            reported_tile_overflow_ = true;
        }
    }

    [[nodiscard]] RasterizerFrameStats last_stats() const { return last_frame_stats_; }

    void set_tunables(const RasterizerTunables& tunables) { tunables_ = tunables; }

    [[nodiscard]] RasterizerTunables tunables() const { return tunables_; }

private:
    void ensure_device_capacity(std::size_t projected_count,
                                std::size_t sorted_index_count,
                                std::size_t tile_count,
                                std::size_t pixel_count) {
        if (projected_count > projected_capacity_) {
            if (projected_dev_) cudaFree(projected_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&projected_dev_,
                                          sizeof(GpuProjectedSplat) * projected_count));
            projected_capacity_ = projected_count;
        }
        ensure_sorted_index_capacity(sorted_index_count);
        if (tile_count > tile_range_capacity_) {
            if (tile_ranges_dev_) cudaFree(tile_ranges_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&tile_ranges_dev_,
                                          sizeof(GpuTileRange) * tile_count));
            tile_range_capacity_ = tile_count;
        }
        if (!stats_dev_) {
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&stats_dev_, sizeof(RasterizerFrameStats)));
        }
        ensure_output_capacity(pixel_count);
    }

    void ensure_sorted_index_capacity(std::size_t sorted_index_count) {
        if (sorted_index_count > sorted_index_capacity_) {
            if (sorted_indices_dev_) cudaFree(sorted_indices_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&sorted_indices_dev_,
                                          sizeof(std::uint32_t) * sorted_index_count));
            sorted_index_capacity_ = sorted_index_count;
        }
    }

    void ensure_compact_tile_aux_capacity(std::size_t tile_count) {
        if (tile_count > compact_tile_capacity_) {
            if (tile_counts_dev_) cudaFree(tile_counts_dev_);
            if (tile_write_counts_dev_) cudaFree(tile_write_counts_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&tile_counts_dev_,
                                          sizeof(std::uint32_t) * (tile_count + 1u)));
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&tile_write_counts_dev_,
                                          sizeof(std::uint32_t) * tile_count));
            compact_tile_capacity_ = tile_count;
        }
        const std::size_t offset_count = tile_count + 1u;
        if (offset_count > compact_tile_offset_capacity_) {
            if (tile_offsets_dev_) cudaFree(tile_offsets_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&tile_offsets_dev_,
                                          sizeof(std::uint32_t) * offset_count));
            compact_tile_offset_capacity_ = offset_count;
        }
    }

    void ensure_compact_scan_temp_capacity(int scan_items) {
        std::size_t required_bytes = 0;
        VKGSPLAT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            required_bytes,
            tile_counts_dev_,
            tile_offsets_dev_,
            scan_items,
            stream_));
        if (required_bytes > compact_scan_temp_capacity_bytes_) {
            if (compact_scan_temp_dev_) cudaFree(compact_scan_temp_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&compact_scan_temp_dev_, required_bytes));
            compact_scan_temp_capacity_bytes_ = required_bytes;
        }
    }

    void ensure_output_capacity(std::size_t pixel_count) {
        if (pixel_count > output_capacity_) {
            if (output_dev_) cudaFree(output_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&output_dev_, sizeof(float4) * pixel_count));
            output_capacity_ = pixel_count;
        }
    }

    void ensure_rgba8_capacity(std::size_t pixel_count) {
        const std::size_t bytes = pixel_count * 4u;
        if (bytes > output_rgba8_capacity_bytes_) {
            if (output_rgba8_dev_) cudaFree(output_rgba8_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&output_rgba8_dev_, bytes));
            output_rgba8_capacity_bytes_ = bytes;
        }
    }

    void ensure_rgba16_capacity(std::size_t pixel_count) {
        const std::size_t bytes = pixel_count * 4u * sizeof(std::uint16_t);
        if (bytes > output_rgba16_capacity_bytes_) {
            if (output_rgba16_dev_) cudaFree(output_rgba16_dev_);
            VKGSPLAT_CUDA_CHECK(cudaMalloc(&output_rgba16_dev_, bytes));
            output_rgba16_capacity_bytes_ = bytes;
        }
    }

    void prime_output_from_target(const ImageDesc& desc, const RenderTarget& target) {
        const std::size_t pixel_count = static_cast<std::size_t>(desc.width) * desc.height;
        ensure_output_capacity(pixel_count);

        switch (target.desc.format) {
        case PixelFormat::R32G32B32A32_SFLOAT:
        case PixelFormat::UNDEFINED:
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(output_dev_,
                                                target.user_handle,
                                                sizeof(float4) * pixel_count,
                                                cudaMemcpyHostToDevice,
                                                stream_));
            break;
        case PixelFormat::R8G8B8A8_UNORM:
        case PixelFormat::R8G8B8A8_SRGB: {
            ensure_rgba8_capacity(pixel_count);
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(output_rgba8_dev_,
                                                target.user_handle,
                                                pixel_count * 4u,
                                                cudaMemcpyHostToDevice,
                                                stream_));
            constexpr int threads = 256;
            const int blocks = static_cast<int>((pixel_count + threads - 1u) / threads);
            unpack_rgba8_kernel<<<blocks, threads, 0, stream_>>>(
                pixel_count, static_cast<const unsigned char*>(output_rgba8_dev_), output_dev_);
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            break;
        }
        case PixelFormat::R16G16B16A16_SFLOAT: {
            ensure_rgba16_capacity(pixel_count);
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(output_rgba16_dev_,
                                                target.user_handle,
                                                pixel_count * 4u * sizeof(std::uint16_t),
                                                cudaMemcpyHostToDevice,
                                                stream_));
            constexpr int threads = 256;
            const int blocks = static_cast<int>((pixel_count + threads - 1u) / threads);
            unpack_rgba16f_kernel<<<blocks, threads, 0, stream_>>>(
                pixel_count, static_cast<const unsigned short*>(output_rgba16_dev_), output_dev_);
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::runtime_error("cuda::Rasterizer: HOST_BUFFER target format is not supported");
        }
    }

    void clear_empty_target(const ImageDesc& desc,
                            const RenderParams& params,
                            const RenderTarget& target) {
        if (!params.clear_to_background) {
            return;
        }

        const std::size_t pixel_count = static_cast<std::size_t>(desc.width) * desc.height;
        const float4 clear{
            params.background.x,
            params.background.y,
            params.background.z,
            0.0f,
        };

        if (target.kind == RenderTargetKind::HOST_BUFFER) {
            ensure_output_capacity(pixel_count);
            constexpr int threads = 256;
            const int blocks = static_cast<int>((pixel_count + threads - 1u) / threads);
            clear_rgba32f_kernel<<<blocks, threads, 0, stream_>>>(
                pixel_count, clear, output_dev_);
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            copy_output_to_target(desc, target);
            return;
        }

        if (desc.format != PixelFormat::R8G8B8A8_UNORM &&
            desc.format != PixelFormat::R8G8B8A8_SRGB) {
            throw std::runtime_error(
                "cuda::Rasterizer: INTEROP_IMAGE currently requires RGBA8/SDR format");
        }
        if (desc.mip_levels != 1 || desc.array_layers != 1) {
            throw std::runtime_error(
                "cuda::Rasterizer: INTEROP_IMAGE currently supports only 2D level-0 layer-0 output");
        }

        const dim3 block(16, 16, 1);
        const dim3 grid((desc.width + block.x - 1u) / block.x,
                        (desc.height + block.y - 1u) / block.y,
                        1);
        const auto surface = static_cast<cudaSurfaceObject_t>(
            reinterpret_cast<std::uintptr_t>(target.user_handle));
        clear_surface_rgba8_kernel<<<grid, block, 0, stream_>>>(
            desc.width, desc.height, clear, surface);
        VKGSPLAT_CUDA_CHECK(cudaGetLastError());
    }

    void copy_output_to_target(const ImageDesc& desc, const RenderTarget& target) {
        const std::size_t pixel_count = static_cast<std::size_t>(desc.width) * desc.height;
        switch (target.desc.format) {
        case PixelFormat::R32G32B32A32_SFLOAT:
        case PixelFormat::UNDEFINED:
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(target.user_handle,
                                                output_dev_,
                                                sizeof(float4) * pixel_count,
                                                cudaMemcpyDeviceToHost,
                                                stream_));
            break;
        case PixelFormat::R8G8B8A8_UNORM:
        case PixelFormat::R8G8B8A8_SRGB: {
            ensure_rgba8_capacity(pixel_count);
            constexpr int threads = 256;
            const int blocks = static_cast<int>((pixel_count + threads - 1u) / threads);
            pack_rgba8_kernel<<<blocks, threads, 0, stream_>>>(
                pixel_count, output_dev_, static_cast<unsigned char*>(output_rgba8_dev_));
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(target.user_handle,
                                                output_rgba8_dev_,
                                                pixel_count * 4u,
                                                cudaMemcpyDeviceToHost,
                                                stream_));
            break;
        }
        case PixelFormat::R16G16B16A16_SFLOAT: {
            ensure_rgba16_capacity(pixel_count);
            constexpr int threads = 256;
            const int blocks = static_cast<int>((pixel_count + threads - 1u) / threads);
            pack_rgba16f_kernel<<<blocks, threads, 0, stream_>>>(
                pixel_count, output_dev_, static_cast<unsigned short*>(output_rgba16_dev_));
            VKGSPLAT_CUDA_CHECK(cudaGetLastError());
            VKGSPLAT_CUDA_CHECK(cudaMemcpyAsync(target.user_handle,
                                                output_rgba16_dev_,
                                                pixel_count * 4u * sizeof(std::uint16_t),
                                                cudaMemcpyDeviceToHost,
                                                stream_));
            break;
        }
        default:
            throw std::runtime_error("cuda::Rasterizer: HOST_BUFFER target format is not supported");
        }
    }

    void launch_surface_output(const ImageDesc& desc,
                               const TileRendererLaunch& tile_launch,
                               const RenderTarget& target) {
        if (desc.format != PixelFormat::R8G8B8A8_UNORM &&
            desc.format != PixelFormat::R8G8B8A8_SRGB) {
            throw std::runtime_error(
                "cuda::Rasterizer: INTEROP_IMAGE currently requires RGBA8/SDR format");
        }
        if (desc.mip_levels != 1 || desc.array_layers != 1) {
            throw std::runtime_error(
                "cuda::Rasterizer: INTEROP_IMAGE currently supports only 2D level-0 layer-0 output");
        }
        launch_tile_renderer_surface_rgba8(tile_launch,
                                           projected_dev_,
                                           sorted_indices_dev_,
                                           tile_ranges_dev_,
                                           target.user_handle,
                                           stream_);
        VKGSPLAT_CUDA_CHECK(cudaGetLastError());
    }

    cudaStream_t stream_ = nullptr;

    void*                 gaussians_dev_     = nullptr; // Gaussian[gaussian_count_]
    PreprocessedGaussian* preprocessed_dev_  = nullptr;
    std::size_t           gaussian_count_    = 0;

    GpuProjectedSplat*    projected_dev_       = nullptr;
    std::uint32_t*        sorted_indices_dev_  = nullptr;
    GpuTileRange*         tile_ranges_dev_     = nullptr;
    std::uint32_t*        tile_counts_dev_     = nullptr;
    std::uint32_t*        tile_offsets_dev_    = nullptr;
    std::uint32_t*        tile_write_counts_dev_ = nullptr;
    void*                 compact_scan_temp_dev_ = nullptr;
    RasterizerFrameStats*  stats_dev_           = nullptr;
    float4*               output_dev_          = nullptr;
    void*                 output_rgba8_dev_    = nullptr;
    void*                 output_rgba16_dev_   = nullptr;
    std::size_t           projected_capacity_  = 0;
    std::size_t           sorted_index_capacity_ = 0;
    std::size_t           tile_range_capacity_ = 0;
    std::size_t           compact_tile_capacity_ = 0;
    std::size_t           compact_tile_offset_capacity_ = 0;
    std::size_t           compact_scan_temp_capacity_bytes_ = 0;
    std::size_t           output_capacity_     = 0;
    std::size_t           output_rgba8_capacity_bytes_ = 0;
    std::size_t           output_rgba16_capacity_bytes_ = 0;
    RasterizerFrameStats   last_frame_stats_{};
    bool                  reported_tile_overflow_ = false;

    RasterizerTunables tunables_{};
    FrameId            frame_ = 0;
};

// ---------------------------------------------------------------------
// Public Rasterizer (thin pimpl over RasterizerImpl)
// ---------------------------------------------------------------------

Rasterizer::Rasterizer()
    : impl_(std::make_unique<RasterizerImpl>()) {}

Rasterizer::~Rasterizer() = default;

void Rasterizer::upload(const Scene& scene) { impl_->upload(scene); }

FrameId Rasterizer::render(const Camera& camera,
                           const RenderParams& params,
                           const RenderTarget& target) {
    return impl_->render(camera, params, target);
}

void Rasterizer::wait(FrameId frame) { impl_->wait(frame); }

RasterizerFrameStats Rasterizer::last_stats() const { return impl_->last_stats(); }

void Rasterizer::set_tunables(const RasterizerTunables& tunables) {
    impl_->set_tunables(tunables);
}

RasterizerTunables Rasterizer::tunables() const { return impl_->tunables(); }

void Rasterizer::bind_to_device_uuid(const std::array<unsigned char, 16>& uuid) {
    impl_->bind_to_uuid(uuid);
}

} // namespace vkgsplat::cuda

// ---------------------------------------------------------------------
// Backend factory
// ---------------------------------------------------------------------

namespace vkgsplat {

std::unique_ptr<Renderer> make_cuda_renderer() {
    return std::make_unique<cuda::Rasterizer>();
}

} // namespace vkgsplat
