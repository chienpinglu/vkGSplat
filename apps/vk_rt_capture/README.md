# vk_rt_capture

A minimal standalone Vulkan + ray-tracing capability probe used as a Phase 4
substitute for the author-private `WickedEngine/Samples/vkSplatCapture/` sample.

## Purpose

`docs/rtx5090_workstation_test_plan.md` Phase 4 requires an external real
Vulkan application to confirm that the workstation can run:

- a Vulkan instance on an NVIDIA adapter,
- SPIR-V shader modules,
- the mesh-shader extension (`VK_EXT_mesh_shader`),
- the Vulkan ray-tracing pipeline extension (`VK_KHR_ray_tracing_pipeline`),
- a non-trivial scene (Cornell-box-shaped acceleration structure),
- a path-tracing-style render path.

The original plan uses Wicked Engine with a private `vkSplatCapture` sample
that we do not have access to. This program is a smaller real replacement:
it queries the same Vulkan capabilities, builds a real BLAS/TLAS for a tiny
Cornell-box mesh, and prints the same `vkSplatCapture:` log lines the Phase 4
PowerShell script expects.

## What it does NOT do

- It does not enter a render loop.
- It does not actually trace rays (no ray-gen shader is dispatched).
- It does not consume Wicked Engine's internal scene format.

This keeps it small while still being a real Vulkan application, sufficient
for Phase 4's "workstation can host a real Vulkan RT app" gate. When the
author publishes the real Wicked `vkSplatCapture` sample, this module can be
disabled with `-DVKGSPLAT_ENABLE_VK_RT_CAPTURE=OFF`.

## Build

Configured automatically when `VKGSPLAT_ENABLE_VK_RT_CAPTURE=ON` and Vulkan is
enabled. Produces `vk_rt_capture(.exe)` next to the viewer binary.
