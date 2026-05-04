# vkGSplat — research proposal (one-page)

## Thesis

**Graphics APIs and graphics hardware should be decoupled. Synthetic data
generation (SDG) for robotics is the workload where the decoupling pays
off first.**

Vulkan is a clean, vendor-neutral API. Implement it as a *software
renderer expressed in compute kernels* — rasterization, ray tracing,
shadow mapping, and 3D Gaussian splatting all lowered to CUDA-class
compute — and run it on AI accelerators (H100, B200, TPU, Trainium).
Existing graphics applications keep their Vulkan code path unchanged;
the implementation underneath stops depending on rasterizers, ROPs, or
RT cores.

## Why this is not a contrarian bet

The graphics-hardware industry has been moving exactly this direction
for five years. **DLSS** is the proof. Each successive generation has
shrunk what the rasterizer and RT cores produce — first to a
sub-native-resolution image (DLSS 1), then to a temporally jittered
sub-image with motion vectors (DLSS 2), then to every-other-frame with
the rest interpolated on tensor cores (DLSS 3). Ray-traced lighting
followed the same path: 1 sample-per-pixel path tracing, denoised by
SVGF/NRD, importance-resampled by ReSTIR. **The fixed-function units no
longer produce the final image — they produce a sparse, noisy seed
that AI reconstructs on tensor cores.** vkGSplat is the architectural
endpoint of that trajectory: replace the seed generator with a
software renderer on the same tensor-rich silicon, and graphics
hardware drops out of the loop.

For SDG specifically — throughput-bound, latency-tolerant, consumed by
a learned model rather than a human eye — this trade is straightforward.

## Why now

1. **Silicon supply.** H100 / B200 / TPU FLOPS dwarf RTX-class FLOPS in
   data-center deployments. SDG running on AI compute uses chips
   companies already own.
2. **Compute-native rendering is mature.** Software
   rasterization (Laine & Karras 2011), GPU ray traversal (Aila & Laine
   2009), Mitsuba 3 / Dr.Jit, and 3DGS (Kerbl et al. 2023) have all
   demonstrated competitive rendering on pure compute.
3. **Robotics SDG demand is exploding.** Isaac Sim, BlenderProc,
   Kubric, Cosmos, ProcTHOR are all rendering-bottlenecked.
4. **The Vulkan ecosystem is huge.** Unreal RHI, Unity, Godot,
   Filament, RenderDoc, NSight all speak Vulkan. Reusing the API gives
   the project the entire graphics tooling ecosystem for free.

## Architecture (3 layers)

1. **Vulkan front-end** — loadable ICD; conformance-targeted subset
   of Vulkan 1.3 + ray-tracing + external-memory; apps need no changes.
2. **Compute IR** — SPIR-V → LLVM/MLIR; command buffers as
   record-and-replay; backend-agnostic.
3. **Compute backend** — CUDA reference; Triton / SYCL portable;
   eventually XLA HLO (TPU), NeuronCore (Trainium).

## v1 plan: 3DGS first

Ship a working 3D Gaussian Splatting backend before tackling raster /
RT lowering. 3DGS is already compute-native — no fixed-function pipeline
required — and it lets the system be useful immediately for the natural
SDG flow: capture real environments with a phone, splat them, render
training data inside them with controlled randomization.

A non-standard `VK_VKGSPLAT_gaussian_splatting` extension lets apps
submit Gaussians directly. Classical raster and RT lowering follow as
v2 / v3.

## What this is *not*

- Not a real-time gaming renderer.
- Not a Vulkan-conformant implementation in v1.
- Not a competitor to Omniverse on Omniverse's own ground.
- Not a new graphics API. Vulkan is the API.

## Evaluation plan (in three numbers)

- **Throughput** — images / sec on Mip-NeRF 360, ProcTHOR, Isaac Lab
  scenes; H100 vs L40 vs TPU v5e.
- **Cost & energy** — $ / image and J / image vs Isaac Sim and
  BlenderProc.
- **Downstream task** — train an object-detection or manipulation
  policy on vkGSplat data vs Isaac Sim data, controlling for total
  training compute. Hypothesis: per-dollar parity or better.

## Risks (honest)

- Software raster / RT on compute pays a constant factor vs
  fixed-function units. Bet: SDG quality budgets absorb it.
- Vulkan's ICD interface has rough edges for unusual backends.
  Mitigation: target a minimum viable subset.
- Retargeting to non-NVIDIA AI chips (TPU, Trainium) is non-trivial.
  Mitigation: CUDA reference first, portability later.

## What I want next

1. Refereed feedback on the position paper (`paper/vkGSplat.tex`).
2. A small prototype of the 3DGS + minimal-Vulkan front-end on H100,
   targeting one Mip-NeRF 360 scene end-to-end.
3. Collaborators who care about the SDG-on-AI-compute thesis.

— CP Lu, chienpng.lu@gmail.com
