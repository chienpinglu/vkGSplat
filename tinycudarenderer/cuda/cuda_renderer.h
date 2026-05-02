// SPDX-License-Identifier: Apache-2.0
//
// tinycudarenderer / cuda — host-callable shim around the CUDA renderer.
// This is the seam the Vulkan driver in ../vulkan/ links against.
//
// Functions in this header are pure C++ (no CUDA-specific syntax) so they
// can be called from translation units compiled by the host compiler.
// The implementation in cuda_renderer.cu uses cudaMalloc/cudaMemcpy and
// dispatches the kernels declared in render.cuh.

#pragma once

#include "render.cuh"
#include "host_loader.h"

#include <cstddef>
#include <cstdint>

namespace tinycuda {

// ---------------------------------------------------------------------------
// Allocation primitives. Opaque pointers; callers should not dereference.
// ---------------------------------------------------------------------------
uint8_t* alloc_framebuffer(int width, int height);
float*   alloc_zbuffer(int width, int height);
void     free_device(void* ptr);

// Fill the framebuffer with a constant BGR color (TGA storage convention).
void clear_framebuffer(uint8_t* d_framebuffer, int width, int height,
                       uint8_t b, uint8_t g, uint8_t r);

// Reset the depth buffer to a far value.
void clear_zbuffer(float* d_zbuffer, int width, int height, float far_value);

void download_framebuffer(const uint8_t* d_framebuffer,
                          uint8_t* host_dst, std::size_t bytes);

// ---------------------------------------------------------------------------
// Model upload / teardown.
// ---------------------------------------------------------------------------
DeviceModel upload_model_to_device(const HostModel& hm);
void        free_device_model(DeviceModel& dm);

// ---------------------------------------------------------------------------
// Per-frame render entry point. Thin wrapper around launch_render_kernel
// that also synchronises the device on return.
// ---------------------------------------------------------------------------
void render_model(const DeviceModel& dm, const RenderParams& params,
                  uint8_t* d_framebuffer, float* d_zbuffer);

// ---------------------------------------------------------------------------
// Camera / matrix helpers — duplicated here so that callers do not have to
// reimplement them. Shared between cuda/main.cu and vulkan/tinyvk_driver.cpp.
// ---------------------------------------------------------------------------
mat<4,4> build_lookat(const vec3& eye, const vec3& center, const vec3& up);
mat<4,4> build_perspective(float f);
mat<4,4> build_viewport(int x, int y, int w, int h);

// ---------------------------------------------------------------------------
// TGA writer that does not pull in tgaimage.h. Suitable for command-line
// readback of the framebuffer.
// ---------------------------------------------------------------------------
bool write_tga(const char* path, const uint8_t* bgr, int width, int height);

} // namespace tinycuda
