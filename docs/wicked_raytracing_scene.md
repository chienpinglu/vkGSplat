# Wicked Engine Ray-Traced Scene Target

This document defines the first Wicked Engine scene target for vkGSplat capture and reconstruction tests.

## Chosen Scene

Use Wicked Engine's Cornell box asset first:

- Scene asset: `third_party/WickedEngine/Content/models/cornellbox.obj`
- Material asset: `third_party/WickedEngine/Content/models/cornellbox.mtl`
- Render path: `wi::RenderPath3D_PathTracing`
- Initial resolution: 256x256 or 512x512
- Initial sample count: 1 sample per frame for noisy seed data, then 16-64 samples for reference comparisons

The Cornell box is the right first target because it is small, closed, diffuse-heavy, and easy to reason about. It exercises visibility, shadows, indirect bounce light, color bleeding, depth, and denoising without burying failures under complex materials.

## Follow-Up Scene Variants

After the static Cornell box works, add one moving object:

- `third_party/WickedEngine/Content/models/bunny.obj` for a geometry-heavy but familiar object,
- `third_party/WickedEngine/Content/models/DamagedHelmet.glb` for PBR material stress,
- `third_party/WickedEngine/Content/models/dragon.obj` only after capture/export performance is acceptable.

The two-frame variant should move either the camera or the inserted object by a small known transform. That gives vkGSplat a controlled test for depth reprojection, screen-space velocity, primitive/object identity, disocclusion, and temporal accumulation.

## Wicked Path Tracing Surfaces

`wi::RenderPath3D_PathTracing` already allocates the surfaces we need:

| Surface | Wicked format / role | vkGSplat use |
| --- | --- | --- |
| `traceResult` | `R32G32B32A32_FLOAT` path-traced accumulation target | Linear noisy seed color and high-sample reference color. |
| `traceDepth` | `R32_FLOAT` path-traced depth | Reprojection, history rejection, disocclusion masks. |
| `traceStencil` | `R8_UINT` path-traced stencil | Visibility/debug signal. |
| `rtPrimitiveID` | Primitive ID UAV/SRV | Object/primitive identity for correspondence and history rejection. |
| `rtLinearDepth` | Linear-depth pyramid derived from `traceDepth` | Low-resolution reconstruction and bilateral/edge-aware filtering. |
| `depthBuffer_Main` | Depth-stencil safety target | Compatibility with existing render path and compositing. |
| `denoiserAlbedo` | Optional OIDN albedo auxiliary | Reference auxiliary for classical/neural denoising comparisons. |
| `denoiserNormal` | Optional OIDN normal auxiliary | Edge-aware denoising and reconstruction supervision. |
| `denoiserResult` | Optional OIDN output | External denoiser baseline, not part of the core vkGSplat path. |

The useful capture point is immediately after `wi::renderer::RayTraceScene(...)` in `RenderPath3D_PathTracing::Render()`, before tonemapping, bloom, FXAA, chromatic aberration, or GUI composition.

## Wicked Calls To Trace

The path tracer flow is:

1. `RenderPath3D_PathTracing::Update()`
   - Resets accumulation when camera, transforms, or materials change.
   - Increments `sam`, the current sample index.
   - Requests acceleration structure updates on sample 0.

2. `RenderPath3D_PathTracing::Render()`
   - Binds current and previous camera state with `wi::renderer::BindCameraCB(...)`.
   - Calls `wi::renderer::UpdateRaytracingAccelerationStructures(...)` when requested.
   - Calls `wi::renderer::RayTraceScene(...)`.
   - Converts `traceDepth` into `rtLinearDepth`.
   - Later copies/tonemaps `traceResult` or `denoiserResult` into `rtMain`.

3. `RenderPath3D_PathTracing::Compose()`
   - Draws the postprocessed result to the final output.
   - This is too late for vkGSplat's reconstruction inputs because the signal has already passed through display/post effects.

## Capture Contract

For each captured frame, export:

- frame index,
- sample index `sam`,
- camera view/projection matrices,
- previous camera view/projection matrices,
- object transforms for moved objects,
- `traceResult` as linear float color,
- `traceDepth` as float depth,
- `rtLinearDepth` mip 0,
- `rtPrimitiveID`,
- optional normal/albedo if available,
- SPIR-V shader modules captured through the Vulkan backend.

For the first two-frame test, export:

- frame A: static Cornell box at sample 0,
- frame B: same scene with a small camera/object motion at sample 0,
- optional high-sample reference for each frame after accumulation.

## vkGSplat Acceptance Tests

The first vkGSplat-side test should not require building Wicked in CI. It should consume a tiny exported fixture with the same fields:

1. Load two frame records.
2. Reproject frame A into frame B.
3. Reject disoccluded pixels using depth and primitive ID.
4. Accumulate stable pixels.
5. Verify that moved/disoccluded pixels do not borrow stale history.

Once this fixture passes, repeat with real Wicked exports.

## Notes

Wicked's path tracing can run as compute ray tracing even without hardware ray tracing acceleration. That is useful for portability, but vkGSplat's target remains CUDA: the C++ path validates the contract first, then the renderer/reprojection/denoising kernels move to CUDA.
