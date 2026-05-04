// SPDX-License-Identifier: Apache-2.0
//
// Minimal M3 mesh-shader prototype for 3DGS splats. One workgroup
// expands a batch of screen-space splats into quads. The production
// version will consume the shared projected-splat/tile contract and
// evaluate covariance/SH on-GPU.

struct ProjectedSplat
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

struct VertexOut
{
  [[vk::location(0)]] float2 local_pos : TEXCOORD0;
  [[vk::location(1)]] float4 position : SV_Position;
};

struct PrimOut
{
  [[vk::location(2)]] nointerpolation float4 color_opacity : COLOR0;
};

[[vk::binding(0, 0)]] StructuredBuffer<ProjectedSplat> splats;
[[vk::binding(1, 0)]] StructuredBuffer<uint> sorted_projected_indices;
[[vk::push_constant]] ConstantBuffer<uint> splat_count;

[numthreads(32, 1, 1)]
[outputtopology("triangle")]
[shader("mesh")]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 dispatchThreadID : SV_DispatchThreadID,
          OutputIndices<uint3, 64> tris,
          OutputVertices<VertexOut, 128> verts,
          OutputPrimitives<PrimOut, 64> prims)
{
  const uint local = groupThreadID.x;
  const uint index = dispatchThreadID.x;
  const uint remaining = index < splat_count ? splat_count - index + local : 0;
  const uint quad_count = min(32, remaining);

  if(local == 0)
  {
    SetMeshOutputCounts(quad_count * 4, quad_count * 2);
  }
  if(index >= splat_count)
  {
    return;
  }

  const uint projected_index = sorted_projected_indices[index];
  const ProjectedSplat s = splats[projected_index];
  const float2 local_pos[4] = {
    float2(-1.0, -1.0),
    float2( 1.0, -1.0),
    float2( 1.0,  1.0),
    float2(-1.0,  1.0),
  };

  tris[local * 2 + 0] = uint3(0, 2, 1) + local * 4;
  tris[local * 2 + 1] = uint3(2, 0, 3) + local * 4;
  prims[local * 2 + 0].color_opacity = float4(s.color, s.opacity);
  prims[local * 2 + 1].color_opacity = float4(s.color, s.opacity);

  [unroll]
  for(uint i = 0; i < 4; ++i)
  {
    const float2 p = local_pos[i];
    // M4 prototype assumes pre-normalized screen coordinates will be
    // supplied by the projection pass once wired into the app.
    const float2 ndc = s.center_px + p.x * s.basis_u_px + p.y * s.basis_v_px;
    verts[local * 4 + i].local_pos = p;
    verts[local * 4 + i].position = float4(ndc, 0.5, 1.0);
  }
}
