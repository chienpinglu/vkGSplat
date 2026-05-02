# tinycudarenderer

A teaching-grade companion to **vkSplat**: the same render job expressed
at three abstraction levels, each in its own subtree.

| Subtree    | Executable                  | Backend                            | Notes                              |
|------------|-----------------------------|------------------------------------|------------------------------------|
| (root)     | `tinycudarenderer`          | CPU rasterizer                     | Verbatim ssloy *tinyrenderer*.     |
| `cuda/`    | `tinycudarenderer_cuda`     | Direct CUDA rasterizer             | Per-face kernels.                  |
| `vulkan/`  | `tinycudarenderer_vulkan`   | Vulkan-shaped API → CUDA driver    | The vkSplat thesis in microcosm.   |

All three render the same scene to `framebuffer.tga`, so the outputs
can be diffed pixel-for-pixel.

## Why this exists

The position paper at `../paper/main.tex` argues that a Vulkan-style
graphics API can be implemented as a software renderer expressed in
compute kernels — i.e., that the API surface and the silicon should be
decoupled. `vulkan/` is a 200-line proof of that idea on a problem
small enough to read in one sitting:

- `vulkan/tinyvk.h` — a Vulkan-shaped C API (`Tvk*` types,
  `tvkCreateInstance`, `tvkAllocateCommandBuffers`,
  `tvkCmdDrawMeshVKSPLAT`, `tvkQueueSubmit`, …).
- `vulkan/tinyvk_driver.cpp` — implements every entry point. The
  command buffer is a `std::vector<Command>`; on submit, each command
  is translated into a CUDA call. **The CUDA host code IS the
  driver.**
- `vulkan/main.cpp` — a Vulkan-flavoured application that knows
  nothing about CUDA. It loads an OBJ via the standard host loader,
  hands it to the driver through `tvkCreateMeshVKSPLAT`, records a
  command buffer, submits, and reads back the image.

If you replace `tinyvk.h` with the real `<vulkan/vulkan.h>` and the
`Tvk*` types with the real `Vk*` types, the application loop is the
same. That is the entire vkSplat pitch.

## Build

From the parent vkSplat repo:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces, alongside the main vkSplat targets:

```
build/tinycudarenderer/tinycudarenderer
build/tinycudarenderer/cuda/tinycudarenderer_cuda
build/tinycudarenderer/vulkan/tinycudarenderer_vulkan
```

You can disable the sandbox with
`-DVKSPLAT_BUILD_TINYCUDARENDERER=OFF` if you do not have a CUDA
toolchain.

If you would rather build it standalone (no parent vkSplat, no
`find_package(Vulkan)`), the `tinycudarenderer/CMakeLists.txt` works
on its own:

```bash
cmake -S tinycudarenderer -B build/tinycudarenderer-only
cmake --build build/tinycudarenderer-only -j
```

## Run

The OBJ assets are not committed; point at the upstream
ssloy/tinyrenderer working tree:

```bash
OBJ=/path/to/tinyrenderer/obj
./build/tinycudarenderer/tinycudarenderer            "$OBJ/diablo3_pose/diablo3_pose.obj" "$OBJ/floor.obj"
./build/tinycudarenderer/cuda/tinycudarenderer_cuda  "$OBJ/diablo3_pose/diablo3_pose.obj" "$OBJ/floor.obj"
./build/tinycudarenderer/vulkan/tinycudarenderer_vulkan \
                                                     "$OBJ/diablo3_pose/diablo3_pose.obj" "$OBJ/floor.obj"
```

Each writes `framebuffer.tga` in the current directory. The CPU and
CUDA outputs differ only in floating-point precision; the Vulkan
output is bit-identical to the CUDA output (same kernels run, just
reached through the Vulkan-shaped front-end).

## Provenance

The CPU files at the root of this tree are reproductions of
[ssloy/tinyrenderer](https://github.com/ssloy/tinyrenderer) at commit
706b2dfecff65daeb93de568ee2c2bd87f277860 and later. They are
distributed under the same liberal license — see `LICENSE.txt`.

The CUDA port in `cuda/` is the in-tree `cutinyrender` project
maintained alongside tinyrenderer, refactored here so the renderer is
also reachable as a static library
(`tinycudarenderer_cuda_lib`) so the Vulkan driver in `vulkan/` can
call into it.

## Layout

```
tinycudarenderer/
├── README.md              this file
├── LICENSE.txt            ssloy tinyrenderer license
├── CMakeLists.txt         drives all three executables
│
├── main.cpp               CPU baseline
├── our_gl.{h,cpp}
├── model.{h,cpp}
├── tgaimage.{h,cpp}
├── geometry.h
│
├── cuda/                  direct CUDA port + reusable lib
│   ├── CMakeLists.txt
│   ├── geometry.cuh
│   ├── render.{cuh,cu}
│   ├── cuda_renderer.{h,cu}    host-callable shim (lib seam)
│   ├── host_loader.{h,cpp}
│   ├── model.{h,cpp}
│   ├── tgaimage.{h,cpp}
│   └── main.cu
│
└── vulkan/                Vulkan-on-CUDA proof of concept
    ├── CMakeLists.txt
    ├── README.md
    ├── tinyvk.h           Vulkan-shaped C API (Tvk*/tvk*)
    ├── tinyvk_driver.cpp  routes tvk_* into the CUDA renderer
    └── main.cpp           Vulkan-shaped application
```
