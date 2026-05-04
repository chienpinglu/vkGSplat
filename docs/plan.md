# vkSplat Plan

This is the working implementation plan. The goal is still to render with CUDA. The current strategy is to make the portable C++ backend correct and testable first, then port the stable renderer, temporal reconstruction, and denoising contracts to CUDA.

## Current Position

vkSplat does not need a DLSS clone as a core dependency, and 3DGS is paused as
the default renderer for now.

The default path should prove that Vulkan/SPIR-V programs can feed low-sample
ray-tracing seed frames, temporal reprojection, denoising, and eventually CUDA
kernels. 3DGS remains in-tree behind `-DVKSPLAT_ENABLE_3DGS=ON`, but it should
not block the main stack. The required near-term reconstruction work is:

- temporal reprojection between adjacent frames,
- history rejection for disocclusion and visibility changes,
- temporal anti-aliasing for stable low-resolution renders,
- classical denoising for low-sample ray-traced seed frames,
- native Apple GPU/Metal validation on this Mac as a portability checkpoint,
- CUDA implementation after the C++ contract and capture data are stable,
- optional neural reconstruction only after the CUDA-oriented temporal path is working.

## Milestones

### M1: Portable C++ Reference Contracts

Status: active; 3DGS renderer paused behind `VKSPLAT_ENABLE_3DGS`.

- Keep CPU reference contracts for camera, scene metadata, tile work partitioning, SPIR-V analysis, ray-tracing seed frames, reprojection, and denoising.
- Keep CUDA out of the critical path until the backend contract is stable, but keep every data layout and algorithm tile-friendly for CUDA.
- Use CTest coverage as the acceptance gate.

Paused optional 3DGS work:

- The C++ tiled 3DGS renderer, CUDA 3DGS tile renderer, mesh-shader 3DGS hooks, viewer, and 3DGS asset tests are disabled by default.
- Re-enable them with `-DVKSPLAT_ENABLE_3DGS=ON` if we need to revisit Gaussian-splat rendering.

### M2: Vulkan Program Capture Contract

Status: in progress.

- Use Wicked Engine as the first realistic Vulkan workload.
- Capture color, depth, linear depth, velocity, primitive ID, camera matrices, and previous camera state.
- Capture SPIR-V shader modules at creation time.
- Export two adjacent frames for reprojection and motion-vector validation.

### M3: SPIR-V Lowering

Status: in progress.

- Continue the restricted SPIR-V parser/analyzer.
- Lower selected compute/ray tracing shader shapes to C++ first for testability.
- Keep CUDA translation as the production target once the lowered behavior is validated.
- Add tests from real Wicked Engine shader modules once capture is wired.

### M4: Temporal Reprojection and TAA

Status: in progress.

- Add a C++ motion/reprojection module.
- Reproject previous color using camera matrices, depth, and velocity.
- Reject history using depth thresholds, primitive ID, and disocclusion masks.
- Add a two-frame CTest fixture that validates stable pixels, rejected disocclusions, and motion-vector consistency.

Current implementation:

- `include/vksplat/reprojection.h` defines the CPU reference frame, motion-map, options, and result contract.
- `src/core/reprojection.cpp` derives a current-to-previous screen-space motion map from current NDC depth plus current/previous camera matrices, then reprojects previous-frame color with that map.
- `tests/test_reprojection.cpp` validates camera-derived motion, stable history reuse, edge disocclusion, primitive-ID rejection, and depth rejection.

### M5: Classical Denoising

Status: CPU baseline implemented; multi-frame moments and normal-guided filtering remain follow-up refinements.

- Add an SVGF-style baseline for noisy ray-traced seed frames.
- Track moments/variance over time.
- Use depth/normal/primitive discontinuities for edge-aware filtering.
- Keep the filter tile-friendly so it can later move to CUDA/SYCL/Triton.

Current implementation:

- `include/vksplat/denoise.h` defines the CPU reference denoise frame, options, and result contract.
- `src/core/denoise.cpp` implements temporal accumulation from the M4 `ReprojectionResult`, luminance variance estimation, and a small edge-aware spatial filter.
- `tests/test_denoise.cpp` validates history use, noisy-pixel smoothing, variance tracking, and rejection across depth/primitive discontinuities.

### M6: Ray Tracing Seed Path

Status: first CPU fixture implemented.

- Use the Wicked Engine Cornell box path tracing target in `docs/wicked_raytracing_scene.md`.
- Support a small Vulkan ray tracing API-shaped test.
- Produce low-resolution, low-sample color/depth/visibility frames.
- Feed the temporal and denoising path with realistic noisy inputs.

Current implementation:

- `include/vksplat/raytrace_seed.h` defines a Vulkan-ray-tracing-shaped seed-frame contract: top-level acceleration structure, ray-generation shader, miss shader, closest-hit shader, color, depth, NDC depth, primitive ID, and camera matrices.
- `src/core/raytrace_seed.cpp` implements a small deterministic CPU triangle tracer with low-sample radiance noise.
- `tests/test_raytrace_seed.cpp` generates two noisy 1-spp frames, derives the M4 camera motion map from NDC depth and matrices, reprojects stable hit history, rejects miss pixels, and feeds the result into the M5 denoiser.

### M6.5: Native Apple GPU Backend

Status: first Metal compute pass implemented.

- Add `VKSPLAT_ENABLE_METAL`, enabled by default on Apple platforms.
- Treat the M4 GPU as a local native compute backend for portability testing, not as a replacement for CUDA.
- Port the stable reconstruction contracts to Metal first, starting with denoising, then reprojection and ray-tracing seed kernels.
- Compare Metal results against the CPU reference with epsilon checks before any backend-specific optimization.

Current implementation:

- `include/vksplat/metal/denoise.h` exposes Metal availability, device-name, and SVGF-style denoise entry points.
- `src/metal/denoise.mm` compiles two Metal compute kernels at runtime: temporal accumulation and edge-aware spatial filtering.
- `tests/test_metal_denoise.cpp` compares Metal output against the CPU denoiser and skips with code `77` if no Metal device is visible.

### M7: CUDA Renderer

Status: target backend after the C++ path is stable.

- Port the stable ray-tracing seed/reprojection/denoise contracts to CUDA first, using the Metal backend as an additional portability check.
- Port temporal reprojection, TAA, and denoising kernels to CUDA.
- Keep SYCL/Triton/other accelerator backends as later portability work behind the same host-visible API.
- Do not let CUDA-only interop shape the C++ correctness model.

### M8: Neural Reconstruction Research

Status: optional research branch.

- Consider a small DLSS-like model only if classical temporal reconstruction is insufficient.
- Inputs would be low-resolution color, depth, velocity, jitter, exposure, and history.
- Training data should come from high-sample/high-resolution references generated by vkSplat or captured from Wicked Engine.
- This is not a blocker for the core Vulkan-to-CUDA stack.

## Immediate Next Step

Extend M6 from the synthetic triangle fixture toward the Wicked Cornell-box capture contract, port reprojection to Metal, then port the stable seed/reprojection/denoise path to CUDA in M7.
