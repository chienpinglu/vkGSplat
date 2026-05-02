// SPDX-License-Identifier: Apache-2.0
//
// tinycudarenderer (CUDA executable). Same wall-clock job as the CPU
// baseline at ../main.cpp, but every fragment of work runs on the GPU.
//
// All renderer guts live in cuda_renderer.{h,cu} so this file is just
// argv parsing + scene setup. The same shim is reused by ../vulkan/
// where the renderer becomes a Vulkan-style driver backend.

#include "cuda_renderer.h"
#include "host_loader.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " obj/model.obj [more.obj ...]\n";
        return 1;
    }

    constexpr int width  = 800;
    constexpr int height = 800;
    const vec3 light  { 1, 1, 1};
    const vec3 eye    {-1, 0, 2};
    const vec3 center { 0, 0, 0};
    const vec3 up     { 0, 1, 0};

    RenderParams params;
    params.ModelView   = tinycuda::build_lookat(eye, center, up);
    params.Perspective = tinycuda::build_perspective(norm(eye - center));
    params.Viewport    = tinycuda::build_viewport(width / 16, height / 16,
                                                  width * 7 / 8, height * 7 / 8);
    params.light_dir   = normalized(params.ModelView *
                                    vec4{light.x, light.y, light.z, 0.f});
    params.width  = width;
    params.height = height;

    uint8_t* d_fb = tinycuda::alloc_framebuffer(width, height);
    float*   d_zb = tinycuda::alloc_zbuffer(width, height);

    // Background BGR (matches the CPU baseline's {177, 195, 209, 255} RGBA).
    tinycuda::clear_framebuffer(d_fb, width, height, 209, 195, 177);
    tinycuda::clear_zbuffer(d_zb, width, height, -1000.f);

    for (int m = 1; m < argc; ++m) {
        HostModel hm = load_model(argv[m]);
        DeviceModel dm = tinycuda::upload_model_to_device(hm);
        tinycuda::render_model(dm, params, d_fb, d_zb);
        tinycuda::free_device_model(dm);
    }

    const std::size_t fb_bytes = static_cast<std::size_t>(width) * height * 3;
    std::vector<uint8_t> host_fb(fb_bytes);
    tinycuda::download_framebuffer(d_fb, host_fb.data(), fb_bytes);

    if (!tinycuda::write_tga("framebuffer.tga", host_fb.data(), width, height)) {
        return 1;
    }
    std::cerr << "[tinycudarenderer/cuda] wrote framebuffer.tga\n";

    tinycuda::free_device(d_fb);
    tinycuda::free_device(d_zb);
    return 0;
}
