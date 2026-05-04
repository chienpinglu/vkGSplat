// SPDX-License-Identifier: Apache-2.0

struct VertexIn
{
  [[vk::location(0)]] float2 local_pos : TEXCOORD0;
};

struct PrimIn
{
  [[vk::location(2)]] nointerpolation float4 color_opacity : COLOR0;
};

[shader("fragment")]
float4 main(VertexIn vert, nointerpolation PrimIn prim) : SV_Target0
{
  const float q = dot(vert.local_pos, vert.local_pos);
  if(q > 1.0)
  {
    discard;
  }
  const float alpha = prim.color_opacity.a * exp(-4.0 * q);
  return float4(prim.color_opacity.rgb * alpha, alpha);
}
