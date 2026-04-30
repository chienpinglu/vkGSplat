# vkSplat

**A compute-first Vulkan path for synthetic data generation on AI accelerators.**

vkSplat is a software implementation of Vulkan whose backend is a renderer
expressed entirely in compute kernels (CUDA today; SYCL / Triton / TPU /
Trainium tomorrow). Existing applications keep their Vulkan code path
unchanged; the implementation underneath stops depending on rasterizers, ROPs,
or RT cores. The target workload is **synthetic data generation for
robotics** — throughput-bound, latency-tolerant, consumed by a learned model
rather than a human eye.

The position the project defends is laid out in `paper/main.tex` and
summarized in `paper/PROPOSAL.md`. **Read those first.** The short version:
DLSS-style rendering has already moved most of the real-time pipeline off
fixed-function graphics units onto tensor cores. vkSplat is the architectural
endpoint of that trajectory — a renderer where seed *and* reconstruction both
live on tensor-rich compute.

## v1 plan: 3D Gaussian Splatting first

3DGS is compute-native by construction (no rasterizer, no ROPs, no RT cores
needed). Shipping a working 3DGS backend behind a thin Vulkan front-end gives
the project a useful end-to-end system before tackling the harder problem of
lowering classical raster, ray tracing, and shadow mapping. Capture a real
environment with a phone, splat it, render training data inside it with
controlled randomization — that is the v1 SDG flow.

## Repository layout

```
vkSplat/
├── paper/                  Position paper (LaTeX) and one-page proposal — START HERE
├── include/vksplat/        Public headers
├── src/
│   ├── core/               Scene, camera, IO, math
│   ├── cuda/               Compute kernels: 3DGS rasterizer, sort, blend
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

1. **Paper.** Land the position argument (`paper/main.tex`) and circulate.
2. **3DGS backend.** Port a forward Gaussian rasterizer to CUDA with an
   offline-friendly variant (deterministic ordering, deeper sample budgets).
3. **Minimal Vulkan front-end.** Loadable ICD covering the subset needed for
   `VK_VKSPLAT_gaussian_splatting` and headless rendering.
4. **Interop & capture.** External-memory and timeline-semaphore interop;
   PNG / EXR capture; deterministic seeds for SDG reproducibility.
5. **Raster lowering.** Tile-based software rasterizer in the spirit of
   Laine & Karras 2011; enough to render typical Isaac Lab / ProcTHOR scenes.
6. **RT lowering.** Software BVH traversal (Aila & Laine 2009 pattern) for
   shadow rays and global illumination; couple with denoising.
7. **Portability.** Second backend in Triton or SYCL; experimental TPU /
   Trainium paths.
8. **Evaluation.** Throughput, $/image, J/image vs Isaac Sim / BlenderProc;
   downstream task accuracy on a small sim-to-real benchmark.

## Build prerequisites

- CMake ≥ 3.24, C++20 (clang ≥ 15, gcc ≥ 12, MSVC 19.34+)
- CUDA Toolkit ≥ 12.0 (for the reference backend)
- Vulkan SDK ≥ 1.3 — see `third_party/vulkan/README.md`
- Python ≥ 3.10 for tooling

## Quick start (placeholder — code does not exist yet)

```bash
./scripts/setup_vulkan.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/apps/viewer/vksplat_viewer assets/<scene>.ply
```

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
