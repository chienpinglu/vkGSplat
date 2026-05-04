// SPDX-License-Identifier: Apache-2.0

#include "vksplat/cpu_reference_renderer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vksplat {
namespace {

struct Mat3 {
    float m[3][3]{};
};

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

float4 mul(const mat4& m, float4 v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w,
    };
}

float3 mul_linear(const mat4& m, float3 v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z,
    };
}

float3 normalize(float3 v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 0.0f) return { 0.0f, 0.0f, 0.0f };
    return { v.x / len, v.y / len, v.z / len };
}

float2 normalize2(float2 v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y);
    if (len <= 0.0f) return { 1.0f, 0.0f };
    return { v.x / len, v.y / len };
}

Mat3 rotation_from_quat(float4 q_in) {
    const float4 qv = [&] {
        const float len = std::sqrt(q_in.x * q_in.x + q_in.y * q_in.y +
                                    q_in.z * q_in.z + q_in.w * q_in.w);
        if (len <= 0.0f) return float4{ 0.0f, 0.0f, 0.0f, 1.0f };
        return float4{ q_in.x / len, q_in.y / len, q_in.z / len, q_in.w / len };
    }();

    const float x = qv.x;
    const float y = qv.y;
    const float z = qv.z;
    const float w = qv.w;

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

Mat3 covariance_world(const Gaussian& g) {
    const Mat3 r = rotation_from_quat(g.rotation);
    const float s[3] = {
        std::exp(g.scale_log.x),
        std::exp(g.scale_log.y),
        std::exp(g.scale_log.z),
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

Mat3 linear_view_matrix(const mat4& view) {
    Mat3 a{};
    a.m[0][0] = view.m[0]; a.m[0][1] = view.m[4]; a.m[0][2] = view.m[8];
    a.m[1][0] = view.m[1]; a.m[1][1] = view.m[5]; a.m[1][2] = view.m[9];
    a.m[2][0] = view.m[2]; a.m[2][1] = view.m[6]; a.m[2][2] = view.m[10];
    return a;
}

Mat3 mul_aba_t(const Mat3& a, const Mat3& b) {
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

float3 sh_color(const Gaussian& g, float3 view_dir) {
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

    float3 color = {
        0.5f + c0 * g.sh[0].x,
        0.5f + c0 * g.sh[0].y,
        0.5f + c0 * g.sh[0].z,
    };

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

    color = { 0.5f, 0.5f, 0.5f };
    for (int i = 0; i < Gaussian::sh_coeffs; ++i) {
        color.x += basis[i] * g.sh[static_cast<std::size_t>(i)].x;
        color.y += basis[i] * g.sh[static_cast<std::size_t>(i)].y;
        color.z += basis[i] * g.sh[static_cast<std::size_t>(i)].z;
    }

    return { clamp01(color.x), clamp01(color.y), clamp01(color.z) };
}

bool projected_covariance_2d(const Gaussian& g,
                             const mat4& view_matrix,
                             float4 view,
                             float fx,
                             float fy,
                             float& a,
                             float& b,
                             float& c) {
    const float z = view.z;
    const float depth = -z;
    if (depth <= 1e-4f) return false;

    const Mat3 cov_view = mul_aba_t(linear_view_matrix(view_matrix), covariance_world(g));

    const float j00 = fx / depth;
    const float j02 = fx * view.x / (depth * depth);
    const float j11 = fy / depth;
    const float j12 = fy * view.y / (depth * depth);

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
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(c);
}

bool ellipse_basis(float a,
                   float b,
                   float c,
                   float extent_sigma,
                   float2& basis_u,
                   float2& basis_v,
                   float& conic_a,
                   float& conic_b,
                   float& conic_c) {
    const float det = a * c - b * b;
    if (!(det > 1e-8f)) return false;

    conic_a = c / det;
    conic_b = -b / det;
    conic_c = a / det;

    const float trace = a + c;
    const float disc = std::sqrt(std::max(0.0f, (a - c) * (a - c) + 4.0f * b * b));
    const float l0 = std::max(1e-6f, 0.5f * (trace + disc));
    const float l1 = std::max(1e-6f, 0.5f * (trace - disc));

    float2 e0{};
    if (std::abs(b) > 1e-6f) {
        e0 = normalize2(float2{ b, l0 - a });
    } else {
        e0 = (a >= c) ? float2{ 1.0f, 0.0f } : float2{ 0.0f, 1.0f };
    }
    const float2 e1{ -e0.y, e0.x };

    basis_u = { e0.x * extent_sigma * std::sqrt(l0), e0.y * extent_sigma * std::sqrt(l0) };
    basis_v = { e1.x * extent_sigma * std::sqrt(l1), e1.y * extent_sigma * std::sqrt(l1) };
    return true;
}

float2 abs_pair(float2 a, float2 b) {
    return { std::abs(a.x) + std::abs(b.x), std::abs(a.y) + std::abs(b.y) };
}

} // namespace

std::vector<ProjectedSplat> project_3dgs_cpu_reference(
    const Scene& scene,
    const Camera& camera,
    const ImageDesc& desc,
    const CpuReferenceRenderOptions& options) {
    std::vector<ProjectedSplat> out;
    out.reserve(scene.size());

    const float half_w = static_cast<float>(desc.width) * 0.5f;
    const float half_h = static_cast<float>(desc.height) * 0.5f;
    const float fx = std::abs(camera.projection().m[0]) * half_w;
    const float fy = std::abs(camera.projection().m[5]) * half_h;

    for (std::size_t i = 0; i < scene.gaussians().size(); ++i) {
        const Gaussian& g = scene.gaussians()[i];
        const float4 view = mul(camera.view(), { g.position.x, g.position.y, g.position.z, 1.0f });
        const float4 clip = mul(camera.projection(), view);
        if (clip.w <= 0.0f) continue;

        const float inv_w = 1.0f / clip.w;
        const float ndc_x = clip.x * inv_w;
        const float ndc_y = clip.y * inv_w;
        const float ndc_z = clip.z * inv_w;
        if (ndc_z < 0.0f || ndc_z > 1.0f) continue;

        const float view_depth = std::max(1e-4f, -view.z);
        float cov_a = 0.0f;
        float cov_b = 0.0f;
        float cov_c = 0.0f;
        if (!projected_covariance_2d(g, camera.view(), view, fx, fy, cov_a, cov_b, cov_c)) continue;

        ProjectedSplat p;
        p.index = static_cast<std::uint32_t>(i);
        p.center = {
            (ndc_x * 0.5f + 0.5f) * static_cast<float>(desc.width),
            (ndc_y * 0.5f + 0.5f) * static_cast<float>(desc.height),
        };
        p.depth = view_depth;

        float3 view_dir = normalize({ -view.x, -view.y, -view.z });
        p.color = sh_color(g, view_dir);
        p.opacity = clamp01(sigmoid(g.opacity_logit));

        if (!ellipse_basis(cov_a, cov_b, cov_c, options.splat_extent_sigma,
                           p.basis_u, p.basis_v, p.conic_a, p.conic_b, p.conic_c)) {
            continue;
        }
        const float2 extent = abs_pair(p.basis_u, p.basis_v);
        p.bounds = {
            p.index,
            p.center.x - extent.x,
            p.center.y - extent.y,
            p.center.x + extent.x,
            p.center.y + extent.y,
        };
        out.push_back(p);
    }

    return out;
}

CpuReferenceRenderResult render_3dgs_cpu_reference(
    const Scene& scene,
    const Camera& camera,
    const RenderParams& params,
    const ImageDesc& target_desc,
    const CpuReferenceRenderOptions& options) {
    if (target_desc.width == 0 || target_desc.height == 0) {
        throw std::runtime_error("render_3dgs_cpu_reference: target dimensions must be nonzero");
    }

    CpuReferenceRenderResult result;
    result.desc = target_desc;
    result.pixels.assign(static_cast<std::size_t>(target_desc.width) * target_desc.height,
                         { params.background.x, params.background.y, params.background.z, 1.0f });

    const TileGrid grid = make_tile_grid(target_desc, options.tile_size);
    result.projected = project_3dgs_cpu_reference(scene, camera, target_desc, options);
    result.bounds.reserve(result.projected.size());
    for (const auto& p : result.projected) result.bounds.push_back(p.bounds);
    result.bins = build_tile_bins(grid, result.bounds);

    std::vector<const ProjectedSplat*> by_index(scene.size(), nullptr);
    for (const auto& p : result.projected) by_index[p.index] = &p;

    for (const TileBin& bin : result.bins) {
        std::vector<const ProjectedSplat*> tile_splats;
        tile_splats.reserve(bin.splat_indices.size());
        for (std::uint32_t idx : bin.splat_indices) {
            if (idx < by_index.size() && by_index[idx]) tile_splats.push_back(by_index[idx]);
        }

        std::stable_sort(tile_splats.begin(), tile_splats.end(), [](const auto* a, const auto* b) {
            return a->depth > b->depth; // back-to-front
        });

        const std::uint32_t x0 = bin.tile_x * grid.tile_size;
        const std::uint32_t y0 = bin.tile_y * grid.tile_size;
        const std::uint32_t x1 = std::min(x0 + grid.tile_size, grid.width);
        const std::uint32_t y1 = std::min(y0 + grid.tile_size, grid.height);

        for (const ProjectedSplat* splat : tile_splats) {
            for (std::uint32_t y = y0; y < y1; ++y) {
                for (std::uint32_t x = x0; x < x1; ++x) {
                    const float px = static_cast<float>(x) + 0.5f;
                    const float py = static_cast<float>(y) + 0.5f;
                    const float dx = px - splat->center.x;
                    const float dy = py - splat->center.y;
                    const float q = splat->conic_a * dx * dx +
                                    2.0f * splat->conic_b * dx * dy +
                                    splat->conic_c * dy * dy;
                    if (q > options.splat_extent_sigma * options.splat_extent_sigma) continue;

                    const float alpha = splat->opacity * std::exp(-0.5f * q);
                    if (alpha <= 1e-4f) continue;

                    float4& dst = result.pixels[static_cast<std::size_t>(y) * target_desc.width + x];
                    dst.x = splat->color.x * alpha + dst.x * (1.0f - alpha);
                    dst.y = splat->color.y * alpha + dst.y * (1.0f - alpha);
                    dst.z = splat->color.z * alpha + dst.z * (1.0f - alpha);
                    dst.w = alpha + dst.w * (1.0f - alpha);
                }
            }
        }
    }

    return result;
}

} // namespace vksplat
