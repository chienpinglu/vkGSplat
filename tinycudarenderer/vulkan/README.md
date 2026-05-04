# tinycudarenderer / vulkan

The smallest interesting demonstration of the vkSplat thesis: a
Vulkan-shaped application whose underlying driver is implemented as
CUDA compute kernels, with no fixed-function graphics hardware in the
loop.

## Why this is not just "another CUDA renderer"

The application code (`main.cpp`) **never includes anything
CUDA-related**. Its only dependency on the renderer is `tinyvk.h`,
which looks and feels like the real `<vulkan/vulkan.h>`. The same call
sequence — instance creation, physical-device enumeration, device
creation, image creation, command buffer recording, queue submit,
device wait — is what you would write against any Vulkan
implementation.

The driver (`tinyvk_driver.cpp`) is plain C++ that includes
`../cuda/cuda_renderer.h`, which exposes the CUDA-side renderer
through a host-callable shim. When the application records
`tvkCmdDrawMeshVKSPLAT` and calls `tvkQueueSubmit`, the driver walks
the recorded commands and translates each one into a CUDA call into
that shim.

This is exactly the structure the position paper at
`../../paper/vkSplat.tex` proposes for the production vkSplat
implementation, in 200 lines instead of 20,000.

## What is "minimal Vulkan API support"?

The `Tvk*` API surface in `tinyvk.h` covers:

- **Instance / physical device / device / queue.** The full Vulkan
  initialisation sequence, with `TvkApplicationInfo`,
  `tvkCreateInstance`, `tvkEnumeratePhysicalDevices`,
  `tvkGetPhysicalDeviceProperties`, `tvkCreateDevice`,
  `tvkGetDeviceQueue`. One physical device, one queue — but the
  shape is preserved.
- **Image and buffer creation/destruction**, including
  `tvkGetImageData` for headless readback.
- **Command buffer recording** — `tvkAllocateCommandBuffers`,
  `tvkBeginCommandBuffer`, `tvkEndCommandBuffer`, plus
  `tvkCmdBeginRendering` / `tvkCmdEndRendering` (the dynamic-rendering
  variant — no render-pass objects).
- **Synchronisation primitives** — `tvkQueueSubmit`,
  `tvkDeviceWaitIdle`, opaque `TvkFence` (currently a no-op stub).
- **One vendor extension**: `TVK_VKSPLAT_mesh`. `tvkCreateMeshVKSPLAT`
  and `tvkCmdDrawMeshVKSPLAT` mirror the
  `VK_VKSPLAT_gaussian_splatting` extension declared by the parent
  vkSplat project (`include/vksplat/extensions/`), but for triangle
  meshes — exactly the compute-primitive draw path the paper
  advocates.

What is **not** here:
- Real render passes, framebuffers, or attachments — the dynamic
  rendering path is sufficient.
- Shader objects, pipeline layouts, descriptor sets — the mesh
  extension carries everything the kernel needs in
  `TvkMeshDrawInfoVKSPLAT`.
- Memory-allocation objects (`vkAllocateMemory`) — images and buffers
  own their device backing directly.
- Swapchain and presentation — the demo is headless, like the SDG
  flow vkSplat targets.

## Files

| File                  | Role                                                                |
|-----------------------|---------------------------------------------------------------------|
| `tinyvk.h`            | The Vulkan-shaped C API. Self-contained, no Vulkan SDK required.    |
| `tinyvk_driver.cpp`   | Driver implementation. Plain C++; links against the CUDA renderer.  |
| `main.cpp`            | Application. Pure `tvk_*` calls; never includes any CUDA header.    |
| `CMakeLists.txt`      | Builds `tinycudarenderer_vulkan` against `tinycudarenderer_cuda_lib`.|

## Reading order

1. `tinyvk.h` — get the API surface.
2. `main.cpp` — see how an "application" uses it. Ignore the matrix
   helpers near the top; the interesting block is from
   `tvkCreateInstance` to `tvkDestroyInstance`.
3. `tinyvk_driver.cpp` — see how each entry point is implemented as
   either bookkeeping or a translation into the CUDA renderer.
4. `../cuda/cuda_renderer.h` — the seam the driver calls into. The
   actual kernels live in `../cuda/render.cu`, unchanged from the
   standalone CUDA executable at `../cuda/main.cu`.

## Build & run

Built as part of the parent vkSplat tree. See `../README.md`. The
output is `build/tinycudarenderer/vulkan/tinycudarenderer_vulkan`,
runnable like the other two front-ends:

```bash
./build/tinycudarenderer/vulkan/tinycudarenderer_vulkan \
    /path/to/tinyrenderer/obj/diablo3_pose/diablo3_pose.obj \
    /path/to/tinyrenderer/obj/floor.obj
```
