# RTX 5090 Workstation Starter Test Plan

This plan is for a new developer validating vkGSplat on a workstation with a
GeForce RTX 5090-class GPU. Windows is a first-class target for this bring-up;
Linux commands are included where they differ. The goal is not to benchmark
final performance yet. The goal is to prove that the workstation can run the
real NVIDIA Vulkan path, the CPU reference tests, and the CUDA build path that
will become the production backend.

## Assumptions

- OS: Windows 11 Pro/Workstation or Ubuntu 24.04/22.04 LTS.
- GPU: GeForce RTX 5090 or equivalent Blackwell GeForce workstation card.
- Driver: recent NVIDIA proprietary driver with Vulkan support.
- CUDA: CUDA Toolkit 12.8 or newer.
- Vulkan SDK/tools: `vulkaninfo`, `glslc`, validation layers.
- Build tools:
  - Windows: Visual Studio 2022 with C++ workload, CMake, Git, PowerShell,
    Python 3, Windows SDK.
  - Linux: `git`, `cmake`, `ninja-build` or Make, `clang`/`gcc`, Python 3.

NVIDIA lists GeForce RTX 5090 as Blackwell with 32 GB GDDR7 memory. NVIDIA's
CUDA GPU table lists GeForce RTX 5090 under compute capability 12.0, and NVIDIA
states CUDA Toolkit 12.8 is the first toolkit with full Blackwell support. For
this repo, the CUDA CMake default includes SM 12.0 when the CUDA compiler is
12.8 or newer.

References:

- https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- https://developer.nvidia.com/cuda/gpus
- https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/

## Phase 0: Hardware And Driver Sanity

Windows PowerShell:

```powershell
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
vulkaninfo --summary
nvcc --version
cmake --version
```

Linux shell:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
vulkaninfo --summary
nvcc --version
cmake --version
```

Pass criteria:

- `nvidia-smi` shows the RTX 5090.
- Driver version is visible and no persistence/Xid errors appear.
- `vulkaninfo --summary` lists an NVIDIA physical device.
- The workstation is not accidentally using llvmpipe, Mesa software Vulkan, or
  an integrated GPU as the primary Vulkan device.

Record:

- OS version: `lsb_release -a`
- Kernel: `uname -a`
- GPU/driver: `nvidia-smi`
- Vulkan summary: `vulkaninfo --summary`

On Windows, replace the OS/kernel lines with:

```powershell
systeminfo | Select-String "OS Name","OS Version","System Type"
```

## Phase 1: Repository And Submodules

Windows PowerShell:

```powershell
git clone https://github.com/chienpinglu/vkGSplat.git
cd vkGSplat
git status --short
New-Item -ItemType Directory -Force third_party
```

Linux shell:

```bash
git clone https://github.com/chienpinglu/vkGSplat.git
cd vkGSplat
git status --short
mkdir -p third_party
```

Pull or place the local third-party checkouts expected by the project:

Wicked Engine should live at `third_party/WickedEngine`.

Pass criteria:

- `third_party/WickedEngine/WickedEngine` exists.
- `third_party/WickedEngine/Samples/vkSplatCapture/CMakeLists.txt` exists.
- `third_party/WickedEngine/Content/models/cornellbox.obj` exists.

## Phase 2: CPU Reference Build

Build the portable reference path first. This isolates repo correctness from GPU
driver issues.

Windows PowerShell:

```powershell
cmake -S . -B build-cpu-rtx5090-win `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DVKGSPLAT_ENABLE_VULKAN=OFF `
  -DVKGSPLAT_ENABLE_CUDA=OFF `
  -DVKGSPLAT_ENABLE_3DGS=OFF `
  -DVKGSPLAT_ENABLE_TORCH=OFF `
  -DVKGSPLAT_ENABLE_METAL=OFF
cmake --build build-cpu-rtx5090-win --config Release --parallel
ctest --test-dir build-cpu-rtx5090-win -C Release --output-on-failure
```

Linux shell:

```bash
cmake -S . -B build-cpu-rtx5090 \
  -DCMAKE_BUILD_TYPE=Release \
  -DVKGSPLAT_ENABLE_VULKAN=OFF \
  -DVKGSPLAT_ENABLE_CUDA=OFF \
  -DVKGSPLAT_ENABLE_3DGS=OFF \
  -DVKGSPLAT_ENABLE_TORCH=OFF \
  -DVKGSPLAT_ENABLE_METAL=OFF
cmake --build build-cpu-rtx5090 --parallel
ctest --test-dir build-cpu-rtx5090 --output-on-failure
```

Pass criteria:

- All CPU tests pass.
- In particular, `test_raytrace_seed`, `test_reprojection`, `test_denoise`,
  `test_spirv_translate`, and `test_sensor_model` pass.

## Phase 3: Vulkan Frontend Smoke

Build vkGSplat with Vulkan enabled, still without CUDA.

Windows PowerShell:

```powershell
cmake -S . -B build-vulkan-rtx5090-win `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DVKGSPLAT_ENABLE_VULKAN=ON `
  -DVKGSPLAT_ENABLE_CUDA=OFF `
  -DVKGSPLAT_ENABLE_3DGS=OFF `
  -DVKGSPLAT_ENABLE_TORCH=OFF
cmake --build build-vulkan-rtx5090-win --config Release --parallel
ctest --test-dir build-vulkan-rtx5090-win -C Release --output-on-failure
```

Linux shell:

```bash
cmake -S . -B build-vulkan-rtx5090 \
  -DCMAKE_BUILD_TYPE=Release \
  -DVKGSPLAT_ENABLE_VULKAN=ON \
  -DVKGSPLAT_ENABLE_CUDA=OFF \
  -DVKGSPLAT_ENABLE_3DGS=OFF \
  -DVKGSPLAT_ENABLE_TORCH=OFF
cmake --build build-vulkan-rtx5090 --parallel
ctest --test-dir build-vulkan-rtx5090 --output-on-failure
```

Pass criteria:

- Vulkan targets configure without falling back to CPU-only.
- `test_vulkan_m7_offscreen` either passes or skips explicitly with code 77.
- No ambiguous loader/ICD errors appear.

If Vulkan selects the wrong device, set the ICD explicitly or remove conflicting
software ICDs from the test shell.

## Phase 4: Real Wicked/NVIDIA Gate

Run the hardware-gated Wicked Cornell smoke test:

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_wicked_nvidia_smoke.ps1
```

Linux shell:

```bash
scripts/run_wicked_nvidia_smoke.sh
```

Or through CTest:

Windows PowerShell:

```powershell
cmake -S . -B build-wicked-nvidia-rtx5090-win `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DVKGSPLAT_ENABLE_WICKED_NVIDIA_TESTS=ON `
  -DVKGSPLAT_ENABLE_VULKAN=OFF `
  -DVKGSPLAT_ENABLE_CUDA=OFF `
  -DVKGSPLAT_ENABLE_3DGS=OFF `
  -DVKGSPLAT_ENABLE_TORCH=OFF `
  -DVKGSPLAT_ENABLE_METAL=OFF
ctest --test-dir build-wicked-nvidia-rtx5090-win `
  -C Release `
  -R test_wicked_nvidia_vulkan_smoke `
  --output-on-failure
```

Linux shell:

```bash
cmake -S . -B build-wicked-nvidia-rtx5090 \
  -DCMAKE_BUILD_TYPE=Release \
  -DVKGSPLAT_ENABLE_WICKED_NVIDIA_TESTS=ON \
  -DVKGSPLAT_ENABLE_VULKAN=OFF \
  -DVKGSPLAT_ENABLE_CUDA=OFF \
  -DVKGSPLAT_ENABLE_3DGS=OFF \
  -DVKGSPLAT_ENABLE_TORCH=OFF \
  -DVKGSPLAT_ENABLE_METAL=OFF
ctest --test-dir build-wicked-nvidia-rtx5090 \
  -R test_wicked_nvidia_vulkan_smoke \
  --output-on-failure
```

Pass criteria:

- Wicked selects an NVIDIA adapter.
- `shader_format=spirv`
- `capability.mesh_shader=yes`
- `capability.raytracing=yes`
- `scene.loaded=yes`
- `render_path=RenderPath3D_PathTracing`
- `capture.ready=yes`
- `capture.mode=raytracing-ready`

Expected log:

```text
vkSplatCapture: initialized=yes
vkSplatCapture: adapter=<NVIDIA ...>
vkSplatCapture: shader_format=spirv
vkSplatCapture: capability.mesh_shader=yes
vkSplatCapture: capability.raytracing=yes
vkSplatCapture: scene.loaded=yes
vkSplatCapture: capture.ready=yes
vkSplatCapture: capture.mode=raytracing-ready
```

Failure handling:

- If `nvidia-smi` fails, fix the driver before touching vkGSplat.
- If `vulkaninfo` does not show NVIDIA, fix the Vulkan ICD/loader.
- If Wicked reports SPIR-V but no ray tracing, update the driver/Vulkan SDK and
  verify the selected adapter.
- If the Cornell scene fails to load, check the Wicked checkout location.

## Phase 5: CUDA Build Gate

Build the CUDA path with Blackwell architecture enabled. The project will add
SM 12.0 automatically for CUDA 12.8+, but pass it explicitly during first
bring-up so the log is unambiguous. This phase validates the software-renderer
CUDA path without Vulkan interop: projected-splat preprocessing, deterministic
fixed tile-list construction, opt-in CUB-scanned compact tile lists, tile
blending, and tensorized reconstruction kernels.

The shortest path is the repo script:

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rtx5090_cuda_smoke.ps1
```

Linux shell:

```bash
scripts/run_rtx5090_cuda_smoke.sh
```

The expanded commands are below for debugging or custom build directories.

Windows PowerShell:

```powershell
cmake -S . -B build-cuda-rtx5090-win `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DVKGSPLAT_ENABLE_CUDA=ON `
  -DVKGSPLAT_ENABLE_3DGS=ON `
  -DVKGSPLAT_ENABLE_VULKAN=OFF `
  -DVKGSPLAT_ENABLE_TORCH=OFF `
  -DVKGSPLAT_CUDA_ARCHITECTURES=120
cmake --build build-cuda-rtx5090-win --config Release --parallel
ctest --test-dir build-cuda-rtx5090-win `
  -C Release `
  -R "cuda|raytrace|reprojection|denoise" `
  --output-on-failure
```

Linux shell:

```bash
cmake -S . -B build-cuda-rtx5090 \
  -DCMAKE_BUILD_TYPE=Release \
  -DVKGSPLAT_ENABLE_CUDA=ON \
  -DVKGSPLAT_ENABLE_3DGS=ON \
  -DVKGSPLAT_ENABLE_VULKAN=OFF \
  -DVKGSPLAT_ENABLE_TORCH=OFF \
  -DVKGSPLAT_CUDA_ARCHITECTURES=120
cmake --build build-cuda-rtx5090 --parallel
ctest --test-dir build-cuda-rtx5090 \
  -R "cuda|raytrace|reprojection|denoise" \
  --output-on-failure
```

Pass criteria:

- CMake detects CUDA Toolkit 12.8 or newer.
- `nvcc` accepts `sm_120`.
- `test_cuda_tile_renderer` passes: the CUDA tile blend kernel consumes
  `GpuProjectedSplat`, sorted indices, and `GpuTileRange`; produces the
  expected blended pixel in float-buffer and RGBA8 CUDA-surface outputs; checks
  surface preserve mode against the float-buffer preserve path; and rejects
  invalid launches such as oversized tiles, missing output buffers, and missing
  surface handles.
- `test_cuda_rasterizer_smoke` passes: the public `make_renderer("cuda")`
  path runs `upload -> render -> wait` through CUDA preprocess/projection,
  fixed-capacity deterministic device tile lists, opt-in compact tile lists,
  tile blending, and `HOST_BUFFER` readback for FP32, RGBA8, and FP16 targets.
  It also validates empty-scene clears, preserve mode, frame stats, CUDA
  rasterizer tunable rejection, and `RenderTargetKind::INTEROP_IMAGE`
  CUDA-surface output for normal frames, empty clears, and preserve-mode
  blending.
- `test_cuda_gaussian_reconstruction` passes: the tensorized reconstruction
  path handles nvdiffrast/seed-buffer ingestion, device-side sample counts,
  tile bin/compact/resolve, gated weighted resolve, state update, and feature
  projection.
- CUDA tests either pass or skip with code 77 only when a test is intentionally
  not runnable on the selected machine.
- CPU ray-tracing/reprojection/denoise tests still pass in the CUDA build.

Important M0 limitation:

- The CUDA rasterizer still keeps the fixed-capacity deterministic per-tile
  list (`max_splats_per_tile`) as the M0 fallback. It also exposes an opt-in M1
  count/scan/scatter path through `RasterizerTunables::use_compact_tile_lists`:
  count each projected splat's tile intersections, run a CUB exclusive scan over
  tile counts, scatter duplicated splat IDs into compact tile spans, then order
  each tile before blending. The smoke test should exercise this compact path with
  `max_splats_per_tile=0` to show it is independent of fixed-list capacity.
- The M1 compact path is still correctness-first: it keeps per-tile insertion
  sorting and includes a small host readback for the compact entry count. The
  next renderer implementation milestone is replacing tile-local insertion sort
  with production radix/cooperative sorting and then making compact tile lists
  the default. Do not treat timings from either M0 or M1 as representative of
  production performance yet.

## Phase 6: CUDA + Vulkan Interop Probe

Only run this after Phase 4 and Phase 5 both pass.

Windows PowerShell:

```powershell
cmake -S . -B build-cuda-vulkan-rtx5090-win `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DVKGSPLAT_ENABLE_CUDA=ON `
  -DVKGSPLAT_ENABLE_VULKAN=ON `
  -DVKGSPLAT_ENABLE_3DGS=ON `
  -DVKGSPLAT_ENABLE_TORCH=OFF `
  -DVKGSPLAT_CUDA_ARCHITECTURES=120
cmake --build build-cuda-vulkan-rtx5090-win --config Release --parallel
ctest --test-dir build-cuda-vulkan-rtx5090-win -C Release --output-on-failure
```

Linux shell:

```bash
cmake -S . -B build-cuda-vulkan-rtx5090 \
  -DCMAKE_BUILD_TYPE=Release \
  -DVKGSPLAT_ENABLE_CUDA=ON \
  -DVKGSPLAT_ENABLE_VULKAN=ON \
  -DVKGSPLAT_ENABLE_3DGS=ON \
  -DVKGSPLAT_ENABLE_TORCH=OFF \
  -DVKGSPLAT_CUDA_ARCHITECTURES=120
cmake --build build-cuda-vulkan-rtx5090 --parallel
ctest --test-dir build-cuda-vulkan-rtx5090 --output-on-failure
```

Pass criteria:

- Vulkan and CUDA both select the same NVIDIA GPU.
- External memory/semaphore tests, if present, do not fail with unsupported
  handle-type errors.
- Any unsupported interop path skips explicitly rather than crashing.

## Phase 7: Output Package For Review

Ask the starter to attach these files or paste these command outputs:

- `git rev-parse HEAD`
- `git status --short`
- `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`
- `nvcc --version`
- `vulkaninfo --summary`
- `build-cpu-rtx5090/Testing/Temporary/LastTest.log`
- `build-wicked-nvidia-rtx5090/Testing/Temporary/LastTest.log`
- `build/wicked-nvidia/wicked_nvidia_smoke.log`
- `build-cuda-rtx5090/Testing/Temporary/LastTest.log`

On Windows, use the `*-win` build-directory names from the commands above.

## Starter Success Definition

The starter is done when:

1. CPU reference tests pass.
2. Vulkan build configures and the offscreen Vulkan smoke is pass-or-explicit-skip.
3. Wicked/NVIDIA Cornell smoke passes with `capture.ready=yes`.
4. CUDA build accepts SM 12.0 and runs `test_cuda_tile_renderer`,
   `test_cuda_rasterizer_smoke`, and `test_cuda_gaussian_reconstruction`.
5. The starter records exact driver, CUDA, Vulkan, and git revision data.

The first useful bug report is not "it failed"; it is the phase number, command,
exit code, final 100 log lines, and whether NVIDIA was visible to both
`nvidia-smi` and `vulkaninfo`.
