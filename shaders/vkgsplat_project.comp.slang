// SPDX-License-Identifier: Apache-2.0
//
// M4 scaffold: GPU-driven projection/culling pass. This establishes the
// buffer contract used before sorting and mesh-shader expansion.

#include "shaderio.h"

struct GpuGaussian
{
  float3 position;
  float3 scale_log;
  float4 rotation;
  float opacity_logit;
  float3 sh[16];
};

struct GpuProjectedSplat
{
  float2 center_px;
  float depth;
  float2 basis_u_px;
  float2 basis_v_px;
  float3 conic;
  float opacity;
  float3 color;
  uint splat_index;
};

struct GpuIndirectParams
{
  uint splat_count;
  uint projected_count;
  uint tile_count;
  uint mesh_group_count_x;
};

struct ProjectPush
{
  uint width;
  uint height;
  uint tile_size;
  float splat_extent_sigma;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuGaussian> gaussians;
[[vk::binding(1, 0)]] RWStructuredBuffer<GpuProjectedSplat> projected;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sort_keys;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> sort_indices;
[[vk::binding(4, 0)]] RWStructuredBuffer<GpuIndirectParams> indirect_params;
[[vk::push_constant]] ConstantBuffer<ProjectPush> push;

uint encodeDepthKey(float depth)
{
  uint bits = asuint(-depth);
  return bits ^ ((int(bits) >> 31) | 0x80000000u);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
  const uint index = tid.x;
  if(index >= indirect_params[0].splat_count)
  {
    return;
  }

  // Placeholder projection for the M4 scaffold: the CPU reference is
  // the truth model. The production shader will mirror covariance
  // projection and SH evaluation from src/core/cpu_reference_renderer.cpp.
  const GpuGaussian g = gaussians[index];
  if(g.opacity_logit < -20.0)
  {
    return;
  }

  uint out_index;
  InterlockedAdd(indirect_params[0].projected_count, 1, out_index);

  GpuProjectedSplat p;
  p.center_px = float2(push.width, push.height) * 0.5;
  p.depth = 1.0 + float(index) * 0.000001;
  p.basis_u_px = float2(push.splat_extent_sigma, 0.0);
  p.basis_v_px = float2(0.0, push.splat_extent_sigma);
  p.conic = float3(1.0, 0.0, 1.0);
  p.opacity = 1.0 / (1.0 + exp(-g.opacity_logit));
  p.color = saturate(0.5 + 0.28209479177 * g.sh[0]);
  p.splat_index = index;
  projected[out_index] = p;
  sort_keys[out_index] = encodeDepthKey(p.depth);
  sort_indices[out_index] = out_index;

  if(out_index % 32 == 0)
  {
    InterlockedAdd(indirect_params[0].mesh_group_count_x, 1);
  }
}
