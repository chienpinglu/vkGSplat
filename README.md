# vkSplat

**A compute-first Vulkan path for synthetic data generation on AI accelerators.**

vkSplat is a software implementation of Vulkan whose backend is a renderer
expressed as a tile-based software path. The production target is CUDA; the
current portable host C++ backend is the correctness/reference path we use to
stabilize the contract before moving kernels onto CUDA. SYCL / Triton / TPU /
Trainium remain later portability targets. Existing applications keep their Vulkan code path
unchanged; the implementation underneath stops depending on rasterizers, ROPs,
or RT cores. The target workload is **synthetic data generation for
robotics** — throughput-bound, latency-tolerant, consumed by a learned model
rather than a human eye.

The position the project defends is laid out in `paper/vkSplat.tex` and
summarized in `paper/PROPOSAL.md`. **Read those first.** The short version:
DLSS-style rendering has already moved most of the real-time pipeline off
fixed-function graphics units onto tensor cores. vkSplat is the architectural
endpoint of that trajectory — a renderer where seed *and* reconstruction both
live on tensor-rich compute.

The current implementation plan is tracked in `docs/plan.md`. The important
near-term decision is that 3DGS is paused as the default path. We keep the
3DGS renderer in-tree behind `-DVKSPLAT_ENABLE_3DGS=ON`, but the default build
now focuses on Vulkan/SPIR-V capture contracts, ray-tracing seed frames,
temporal reprojection, denoising, native Metal validation on Apple GPUs, and
CUDA lowering. A neural DLSS-like reconstruction model remains an optional
research branch after those contracts are stable.

## Active path: Vulkan ray-tracing seeds first

The active path is a low-sample ray-tracing seed renderer shaped like
`VK_KHR_ray_tracing_pipeline`, followed by temporal reprojection, history
rejection, denoising, and eventually CUDA kernels. This keeps the project
aligned with the goal of taking Vulkan programs and rendering them on CUDA
without getting blocked on 3DGS visual quality.

3DGS remains useful research context and an optional backend, but it is no
longer part of the default milestone gate.

## Repository layout

```
vkSplat/
├── paper/                  Position paper (LaTeX) and one-page proposal — START HERE
├── include/vksplat/        Public headers
├── src/
│   ├── core/               Scene, camera, IO, math
│   ├── cuda/               Experimental compute kernels
│   ├── vulkan/             Vulkan ICD front-end: instance, device, command buffers
│   └── interop/            Compute <-> Vulkan external memory + semaphores
├── shaders/                SPIR-V passes that survive into the compute backend
├── apps/viewer/            Debug / reference viewer
├── tests/
├── docs/                   Architecture notes
├── scripts/                Setup and tooling
├── third_party/vulkan/     Vulkan SDK / loader / validation layers (gitignored)
└── assets/                 Test scenes (.ply / .splat)
```

A `tinyrender/` scratch sandbox lives next to this repo for prototype spikes.

## Roadmap

1. **Paper.** Land the position argument (`paper/vkSplat.tex`) and circulate.
2. **Vulkan capture contract.** Capture SPIR-V, frame resources, camera state,
   depth, primitive identity, and adjacent-frame metadata from real Vulkan
   programs.
3. **Ray-tracing seed path.** Start with API-shaped CPU fixtures, then connect
   real Wicked Engine/Cornell-box capture and lower the stable contract to CUDA.
4. **Native Apple GPU backend.** Use Metal on Apple Silicon as a local compute
   backend for reconstruction kernels and portability checks.
5. **Interop & capture.** External-memory and timeline-semaphore interop;
   PNG / EXR capture; deterministic seeds for SDG reproducibility.
6. **Temporal reconstruction.** Motion-vector reprojection, depth/ID-based
   history rejection, TAA, and classical denoising for low-sample seed frames.
7. **Raster lowering.** Tile-based software rasterizer in the spirit of
   Laine & Karras 2011; enough to render typical Isaac Lab / ProcTHOR scenes.
8. **RT lowering.** Software BVH traversal (Aila & Laine 2009 pattern) for
   shadow rays and global illumination; couple with denoising.
9. **Portability.** Second backend in Triton or SYCL; experimental TPU /
   Trainium paths.
10. **Evaluation.** Throughput, $/image, J/image vs Isaac Sim / BlenderProc;
   downstream task accuracy on a small sim-to-real benchmark.

## Build prerequisites

- CMake ≥ 3.24, C++20 (clang ≥ 15, gcc ≥ 12, MSVC 19.34+)
- CUDA Toolkit ≥ 12.0 (optional, for the experimental CUDA backend)
- Vulkan SDK ≥ 1.3 (optional for CPU-only builds)
- Xcode Command Line Tools with Metal.framework (optional, enabled by default
  on Apple platforms)
- Python ≥ 3.10 for tooling

## Quick start

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DVKSPLAT_ENABLE_VULKAN=OFF
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

On Apple Silicon, `VKSPLAT_ENABLE_METAL` defaults to `ON`. The Metal tests need
access to the native GPU; in sandboxed runners they may skip even though they
pass from a normal terminal.

The paused 3DGS path can be rebuilt explicitly with
`-DVKSPLAT_ENABLE_3DGS=ON`; that also enables the 3DGS viewer and
3DGS-specific tests.

## Non-goals

- Real-time interactive rendering for games or VR.
- Vulkan conformance in v1 (target is a defined subset, not the full spec).
- Outperforming RTX hardware on its own ground.

## License

TBD.

## Citation

```
@misc{lu2026vksplat,
  title  = {vkSplat: A Compute-First Vulkan Path for Robotics Synthetic Data Generation on AI Accelerators},
  author = {Lu, Chien-Ping},
  year   = {2026},
  note   = {Working draft, \url{https://github.com/chienpinglu/vkSplat}}
}
```
