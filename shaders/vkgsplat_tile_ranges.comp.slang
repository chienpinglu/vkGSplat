// SPDX-License-Identifier: Apache-2.0
//
// M4 scaffold for tile range construction after projected splats have
// been sorted/compacted. The exact parallel prefix implementation will
// land with the Vulkan radix-sort integration.

struct GpuTileRange
{
  uint offset;
  uint count;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GpuTileRange> tile_ranges;

[numthreads(256, 1, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
  // Initialize ranges deterministically. Real construction will fill
  // offsets/counts from sorted tile keys.
  tile_ranges[tid.x].offset = 0;
  tile_ranges[tid.x].count = 0;
}
