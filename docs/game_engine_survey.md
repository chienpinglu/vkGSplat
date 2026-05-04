# Open-Source Vulkan Game Engine Survey

vkSplat needs realistic Vulkan programs to capture frame data from: color, depth, motion vectors, camera matrices, materials, object IDs, and eventually shader/scene metadata. The goal is not to depend on a game engine, but to use one as a representative Vulkan workload while we build the 3DGS reconstruction stack.

## Shortlist

| Engine / framework | Repo | Language | License | Vulkan status | Fit for vkSplat |
| --- | --- | --- | --- | --- | --- |
| Wicked Engine | <https://github.com/turanszkij/WickedEngine> | C++ | MIT | Vulkan backend plus DX12 and other platforms | Best first target: renderer-focused, modern, smaller than Godot, good for capture hooks. |
| The Forge | <https://github.com/ConfettiFX/The-Forge> | C/C++ | Apache-2.0 | Mature multi-API rendering framework including Vulkan | Best backend architecture reference; less of a full game engine. |
| Godot 4 | <https://github.com/godotengine/godot> | C++ | MIT | Production Vulkan renderer | Best eventual real-world integration target, but large and harder to instrument first. |
| Acid | <https://github.com/EQMG/Acid> | C++17 | MIT | Vulkan-only renderer | Smaller Vulkan-only game engine; useful second-tier study target. |
| NcEngine | <https://github.com/NcStudios/NcEngine> | C++23 | Open source | Vulkan renderer | Modern and clean, but early-stage. |
| Lugdunum | <https://github.com/Lugdunum3D/Lugdunum> | C++ | Open source | Vulkan backend | Interesting but lower priority. |
| Kaiju | <https://github.com/KaijuEngine/kaiju> | Go | Open source | Vulkan renderer/editor | Interesting editor/runtime, but language mismatch for vkSplat. |

## Recommendation

Start with **Wicked Engine**.

Reasons:

- It is a real 3D engine, not only a sample framework.
- Its license is permissive.
- It is renderer-heavy and easier to inspect than Godot.
- It has modern rendering features that naturally expose buffers we need: depth, velocity/motion vectors, G-buffer-style data, camera constants, and render graph state.
- It is a good candidate for early Vulkan-frame capture experiments before moving to Godot-scale integration.

Use **The Forge** as an architecture reference for clean multi-backend rendering abstractions.

Use **Godot** later when we want a large production engine target with editor workflows, imported assets, animation, scripting, physics, and real user scenes.

## vkSplat Capture Priorities

The first engine integration should extract:

1. Color frame.
2. Depth frame.
3. Camera intrinsics/projection and view matrices.
4. Previous-frame camera matrices.
5. Screen-space motion vectors.
6. Object/material IDs when available.
7. Optional normal/albedo/roughness buffers.
8. Shader entry points and SPIR-V blobs when available.

For ray-traced synthetic training data, COLMAP should not be required because the engine already knows the camera poses. COLMAP remains useful for importing real photo/video captures.

## Wicked Engine First Pass

Local checkout:

- Path: `third_party/WickedEngine`
- Upstream: <https://github.com/turanszkij/WickedEngine>
- Commit inspected: `8e0c260e`

Wicked Engine is a strong first Vulkan workload for vkSplat because it is compact enough to instrument, but still exposes realistic renderer state. The Vulkan backend reports SPIR-V as its shader format and builds `VkShaderModule` objects in `GraphicsDevice_Vulkan::CreateShader`, which is the natural hook for collecting shader bytecode and feeding the SPIR-V-to-C / SPIR-V-to-CUDA translators.

The first render-path capture surface should be `wiRenderPath3D`. It already owns most of the temporal and screen-space data needed for reconstruction:

| Wicked Engine surface | Why vkSplat needs it |
| --- | --- |
| `rtMain` | Main color frame. |
| `depthBuffer_Copy` | Current resolved/copied depth for reprojection and visibility. |
| `depthBuffer_Copy1` | Previous depth; Wicked swaps this with the current depth each frame. |
| `rtLinearDepth` | Linear depth, easier for low-resolution reconstruction passes. |
| `rtVelocity` | Screen-space motion map; created when TAA, motion blur, SSR/SSGI, ray tracing, FSR2, or reprojection paths need it. |
| `rtPrimitiveID` | Primitive/object identity signal for correspondence and visibility analysis. |
| `visibilityResources.texture_normals` | Optional normal signal for reconstruction losses and splat initialization. |
| `visibilityResources.texture_roughness` | Optional material signal for separating appearance from geometry. |
| `reprojectedDepth` | Existing temporal reprojection helper, useful for disocclusion tests. |
| `camera_previous` | Previous-frame camera state, needed to validate generated motion vectors. |

The first Vulkan backend hooks to inspect are:

1. `GraphicsDevice_Vulkan::CreateShader`: capture SPIR-V bytes, shader stage, entry point, and reflection metadata.
2. `GraphicsDevice_Vulkan::RenderPassBegin`: identify frame boundaries, attachments, layouts, and render target formats.
3. `GraphicsDevice_Vulkan::GetBackBuffer`: locate swapchain images when capturing final output.
4. `wiRenderPath3D` post-main-scene/post-visibility phases: capture color, depth, velocity, primitive ID, and optional G-buffer resources before heavy post-processing changes the signal.

Near-term plan for Wicked Engine:

1. Build a read-only capture shim that logs shader modules, render pass attachments, and `wiRenderPath3D` texture descriptors without changing rendering.
2. Add a Cornell box path-tracing capture path, described in `docs/wicked_raytracing_scene.md`, that exports color, depth, velocity or derived motion, primitive ID, and camera matrices for two adjacent frames.
3. Feed those two frames into vkSplat's CPU tiled renderer/reconstruction tests to validate depth reprojection and motion-map consistency.
4. Once the capture contract is stable, translate selected Wicked SPIR-V shaders through vkSplat's restricted SPIR-V translators and compare CPU/CUDA-friendly outputs against the original Vulkan frame data.
