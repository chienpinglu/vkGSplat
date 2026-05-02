# shaders/

GLSL passes that survive into the Vulkan presentation path. The 3DGS
rasterization itself is **not** here — it lives in CUDA
(`src/cuda/rasterizer.cu`). What lives in this folder is:

| File           | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `present.vert` | Fullscreen-triangle vertex shader (no VB needed).                        |
| `present.frag` | Samples the CUDA-written render target and blits it to the swapchain.   |

The position paper (Section 3.4) plans for DLSS-style neural
reconstruction to slot into the present-fragment stage so that the CUDA
seed can stay low-resolution and noisy. v1 ships the passthrough only.

## Compilation

Compile to SPIR-V with `glslc` (ships with the Vulkan SDK):

```bash
glslc -O present.vert -o present.vert.spv
glslc -O present.frag -o present.frag.spv
```

CMake integration (added when the viewer wires shaders in) uses
`find_program(GLSLC glslc)` and a custom command per `.vert/.frag`.
