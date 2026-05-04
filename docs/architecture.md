# vkGSplat Architecture

## One-line summary

A 3D Gaussian Splatting renderer that runs the **rasterization on CUDA** and
the **presentation on Vulkan**, joined by a zero-copy interop bridge.

## Why split compute and display?

Most production 3DGS renderers today bundle compute and display on a single
backend — usually pure CUDA + OpenGL/CUDA interop, or pure compute shaders.
That's fine for a research demo but creates friction for:

- UI layers (ImGui, native windowing, embedded engines).
- Cross-vendor presentation (the *display* path can run anywhere Vulkan does).
- Sharing kernels with a training pipeline (the CUDA backend can stay aligned
  with the differentiable rasterizer used for fitting).
- Profiling: clear seam between "what the kernel costs" and "what the
  presentation costs."

vkGSplat treats this as an architectural commitment, not a workaround.

## Block diagram (textual)

```
                       +---------------------------+
   .ply / .splat  -->  |   Scene / Camera (core)   |  <-- ImGui controls
                       +-------------+-------------+
                                     |
                          +----------+----------+
                          | Frame Orchestrator  |
                          +-+----------------+--+
                            |                |
              CUDA queue    |                |  Vulkan graphics queue
                            v                v
                   +-----------------+   +-----------------+
                   | CUDA Rasterizer |   | Vulkan Frontend |
                   |  - preprocess   |   |  - swapchain    |
                   |  - tile-bin     |   |  - postprocess  |
                   |  - sort         |   |  - ImGui        |
                   |  - blend  -->   +-->+  - present      |
                   +--------+--------+   +--------+--------+
                            |                     ^
                            v                     |
                  +----------------------------------------+
                  |  Interop Layer                         |
                  |   - VkImage <-> CUarray (external mem) |
                  |   - VkSemaphore <-> cudaExtSemaphore   |
                  |   - timeline values for ordering       |
                  +----------------------------------------+
```

## Per-frame timeline

```
t0  CPU: poll input, update camera, advance frame index
t1  Vulkan: acquire swapchain image; signal semaphore S_acq (timeline N)
t2  CUDA: wait on S_acq=N; rasterize Gaussians into the imported VkImage
t3  CUDA: signal semaphore S_render (timeline N)
t4  Vulkan: wait on S_render=N; run postprocess + ImGui
t5  Vulkan: submit + present
```

The same timeline semaphores are reused across frames — only the value advances.

## Module ownership

| Module                   | Owner backend | Notes                                           |
|--------------------------|---------------|-------------------------------------------------|
| `core/scene`             | host C++      | .ply / .splat IO, Gaussian buffer layout.       |
| `core/camera`            | host C++      | View/proj matrices, controller.                 |
| `cuda/rasterizer`        | CUDA          | Preprocess, tile-bin, sort, blend kernels.      |
| `vulkan/instance,device` | Vulkan        | Extensions: external memory/semaphore, timeline.|
| `vulkan/swapchain`       | Vulkan        | Surface + image views + sync.                   |
| `interop/external_memory`| both          | Allocate VkImage with external handle, import to CUDA. |
| `interop/timeline_semaphore` | both     | Single semaphore object visible to both APIs.   |

## Required Vulkan extensions

- `VK_KHR_external_memory` (+ `_fd` Linux / `_win32` Windows)
- `VK_KHR_external_semaphore` (+ `_fd` / `_win32`)
- `VK_KHR_timeline_semaphore` (or core in Vulkan 1.2+)
- `VK_KHR_swapchain`
- (Optional) `VK_EXT_debug_utils` for validation.

## Design constraints / non-goals

- **Not** a fully-portable compute backend. CUDA is the compute path; a
  Vulkan-compute alternative is future work, not v0.
- **Not** a training framework. We render trained Gaussians; we don't fit them
  (yet). A backward pass is on the roadmap.
- macOS is a partial target — only the Vulkan side runs there (via MoltenVK).
  The full pipeline targets Linux and Windows with NVIDIA GPUs.
