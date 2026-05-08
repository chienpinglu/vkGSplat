# vkGSplat

**A compute-first Vulkan path for synthetic data generation on AI accelerators.**

vkGSplat is a software implementation of Vulkan whose backend is a renderer
expressed as a tile-based software path. The production target is CUDA; the
current portable host C++ backend is the correctness/reference path we use to
stabilize the contract before moving kernels onto CUDA. SYCL / Triton / TPU /
Trainium remain later portability targets. Existing applications keep their Vulkan code path
unchanged; the implementation underneath stops depending on rasterizers, ROPs,
or RT cores. The target workload is **synthetic data generation for
robotics** — throughput-bound, latency-tolerant, consumed by a learned model
rather than a human eye.

The position the project defends is laid out in `papers/vkGSplat.tex` and
summarized in `papers/PROPOSAL.md`. **Read those first.** The short version:
DLSS-style rendering has already moved most of the real-time pipeline off
fixed-function graphics units onto tensor cores. vkGSplat is the architectural
endpoint of that trajectory — a renderer where seed *and* reconstruction both
live on tensor-rich compute.

The current implementation plan is tracked in `docs/plan.md`. The default build
focuses on Vulkan/SPIR-V capture contracts, ray-tracing seed frames, temporal
reprojection, denoising, native Metal validation on Apple GPUs, and CUDA
lowering. The 3DGS renderer remains behind `-DVKGSPLAT_ENABLE_3DGS=ON`; it is
now also the CUDA renderer bring-up scaffold for projected-splat preprocessing,
device tile lists, and tile blending. A neural DLSS-like reconstruction model
remains an optional research branch after those contracts are stable.

## Active path: Vulkan ray-tracing seeds first

The active path is a low-sample ray-tracing seed renderer shaped like
`VK_KHR_ray_tracing_pipeline`, followed by temporal reprojection, history
rejection, denoising, and eventually CUDA kernels. This keeps the project
aligned with the goal of taking Vulkan programs and rendering them on CUDA
without getting blocked on 3DGS visual quality.

3DGS remains optional in the default build, but the CUDA 3DGS path is useful as
the first end-to-end software rasterizer scaffold: it exercises the shared
`GpuProjectedSplat` / `GpuTileRange` ABI that temporal reconstruction,
supersampling, and frame-generation kernels should reuse.

## Repository layout

```
vkGSplat/
├── papers/                  Position paper (LaTeX) and one-page proposal — START HERE
├── include/vkgsplat/        Public headers
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

1. **Paper.** Land the position argument (`papers/vkGSplat.tex`) and circulate.
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
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DVKGSPLAT_ENABLE_VULKAN=OFF
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

On Apple Silicon, `VKGSPLAT_ENABLE_METAL` defaults to `ON`. The Metal tests need
access to the native GPU; in sandboxed runners they may skip even though they
pass from a normal terminal.

The paused 3DGS path can be rebuilt explicitly with
`-DVKGSPLAT_ENABLE_3DGS=ON`; that also enables the 3DGS viewer and
3DGS-specific tests.

## Current test plan

The local always-on gate is the CPU/Metal-safe suite:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DVKGSPLAT_ENABLE_VULKAN=OFF
cmake --build build-cpu --parallel 4
ctest --test-dir build-cpu --output-on-failure
```

On this Mac, the latest local run passed all 11 default tests. CUDA is not
validated locally because the machine does not have `nvcc`; CMake stops with
`Failed to find nvcc`.

On an NVIDIA CUDA workstation, use the RTX 5090 bring-up plan:
`docs/rtx5090_workstation_test_plan.md`. The CUDA gate currently includes:

- `test_cuda_tile_renderer`: validates the tile blend kernel on a tiny
  projected-splat fixture, including direct RGBA8 CUDA surface writes.
- `test_cuda_rasterizer_smoke`: exercises the public `make_renderer("cuda")`
  path through upload, CUDA preprocess/projection, deterministic fixed-capacity
  device tile lists/ranges, tile blending, host-buffer readback, and
  `INTEROP_IMAGE` CUDA-surface output.
- `test_cuda_gaussian_reconstruction`: validates the tensorized reconstruction
  kernels for nvdiffrast/seed-buffer ingestion, device-side sample counts,
  tile bin/compact/resolve, gated weighted resolve, state update, and feature
  projection.

The native Vulkan hardware gate remains separate: run the Wicked/NVIDIA smoke
test on a Linux or Windows NVIDIA Vulkan stack, then run the CUDA gates, then
run CUDA+Vulkan interop once both sides independently pass.

## Wicked Engine on NVIDIA Vulkan

The real Wicked acceptance path requires a Linux or Windows machine with an
NVIDIA Vulkan driver that exposes mesh shaders and ray tracing. On Linux, after
checking out Wicked Engine into `third_party/WickedEngine`, run:

```bash
scripts/run_wicked_nvidia_smoke.sh
```

To register it with CTest while allowing non-NVIDIA machines to skip on Linux:

```bash
cmake -S . -B build-nvidia-tests \
  -DVKGSPLAT_ENABLE_WICKED_NVIDIA_TESTS=ON \
  -DVKGSPLAT_ENABLE_VULKAN=OFF \
  -DVKGSPLAT_ENABLE_CUDA=OFF
ctest --test-dir build-nvidia-tests -R test_wicked_nvidia_vulkan_smoke --output-on-failure
```

On Windows, the same CTest option uses
`scripts/run_wicked_nvidia_smoke.ps1`; direct use is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_wicked_nvidia_smoke.ps1
```

The test passes only when Wicked selects an NVIDIA adapter, reports SPIR-V,
mesh-shader support, ray-tracing support, loads the Cornell asset, and reaches
`capture.ready=yes`. On this Mac it is expected to skip because MoltenVK does
not expose the required NVIDIA Vulkan path.

For a new RTX 5090 workstation bring-up, follow
`docs/rtx5090_workstation_test_plan.md`.

## Non-goals

- Real-time interactive rendering for games or VR.
- Vulkan conformance in v1 (target is a defined subset, not the full spec).
- Outperforming RTX hardware on its own ground.

## License

TBD.

## Citation

```
@misc{lu2026vkgsplat,
  title  = {vkGSplat: A Compute-First Vulkan Path for Robotics Synthetic Data Generation on AI Accelerators},
  author = {Lu, Chien-Ping},
  year   = {2026},
  note   = {Working draft, \url{https://github.com/chienpinglu/vkGSplat}}
}
```
