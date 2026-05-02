// SPDX-License-Identifier: Apache-2.0
//
// Host-callable shim that the Vulkan driver in ../vulkan/ and the standalone
// cuda/main.cu both link against. See cuda_renderer.h.

#include "cuda_renderer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define TINYCUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::abort(); \
    } \
} while(0)

namespace tinycuda {

namespace {

DeviceTexture upload_texture(const HostTexture& ht) {
    DeviceTexture dt;
    dt.w   = ht.w;
    dt.h   = ht.h;
    dt.bpp = ht.bpp;
    dt.data = nullptr;
    if (ht.data.empty()) return dt;
    TINYCUDA_CHECK(cudaMalloc(&dt.data, ht.data.size()));
    TINYCUDA_CHECK(cudaMemcpy(dt.data, ht.data.data(), ht.data.size(),
                              cudaMemcpyHostToDevice));
    return dt;
}

} // namespace

uint8_t* alloc_framebuffer(int width, int height) {
    uint8_t* p = nullptr;
    TINYCUDA_CHECK(cudaMalloc(&p, static_cast<std::size_t>(width) * height * 3));
    return p;
}

float* alloc_zbuffer(int width, int height) {
    float* p = nullptr;
    TINYCUDA_CHECK(cudaMalloc(&p, static_cast<std::size_t>(width) * height * sizeof(float)));
    return p;
}

void free_device(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void clear_framebuffer(uint8_t* d_framebuffer, int width, int height,
                       uint8_t b, uint8_t g, uint8_t r) {
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<uint8_t> bg(pixels * 3);
    for (std::size_t i = 0; i < pixels; ++i) {
        bg[i * 3 + 0] = b;
        bg[i * 3 + 1] = g;
        bg[i * 3 + 2] = r;
    }
    TINYCUDA_CHECK(cudaMemcpy(d_framebuffer, bg.data(), bg.size(),
                              cudaMemcpyHostToDevice));
}

void clear_zbuffer(float* d_zbuffer, int width, int height, float far_value) {
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> zinit(pixels, far_value);
    TINYCUDA_CHECK(cudaMemcpy(d_zbuffer, zinit.data(), pixels * sizeof(float),
                              cudaMemcpyHostToDevice));
}

void download_framebuffer(const uint8_t* d_framebuffer,
                          uint8_t* host_dst, std::size_t bytes) {
    TINYCUDA_CHECK(cudaMemcpy(host_dst, d_framebuffer, bytes,
                              cudaMemcpyDeviceToHost));
}

DeviceModel upload_model_to_device(const HostModel& hm) {
    DeviceModel dm;
    dm.nfaces = hm.nfaces;
    const int total = hm.nfaces * 3;

    TINYCUDA_CHECK(cudaMalloc(&dm.verts, hm.verts.size() * sizeof(float)));
    TINYCUDA_CHECK(cudaMemcpy(dm.verts, hm.verts.data(),
                              hm.verts.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

    TINYCUDA_CHECK(cudaMalloc(&dm.norms, hm.norms.size() * sizeof(float)));
    TINYCUDA_CHECK(cudaMemcpy(dm.norms, hm.norms.data(),
                              hm.norms.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

    TINYCUDA_CHECK(cudaMalloc(&dm.tex, hm.tex.size() * sizeof(float)));
    TINYCUDA_CHECK(cudaMemcpy(dm.tex, hm.tex.data(),
                              hm.tex.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

    TINYCUDA_CHECK(cudaMalloc(&dm.facet_vrt, total * sizeof(int)));
    TINYCUDA_CHECK(cudaMemcpy(dm.facet_vrt, hm.indices.data(),
                              total * sizeof(int), cudaMemcpyHostToDevice));

    TINYCUDA_CHECK(cudaMalloc(&dm.facet_nrm, total * sizeof(int)));
    TINYCUDA_CHECK(cudaMemcpy(dm.facet_nrm, hm.indices.data(),
                              total * sizeof(int), cudaMemcpyHostToDevice));

    TINYCUDA_CHECK(cudaMalloc(&dm.facet_tex, total * sizeof(int)));
    TINYCUDA_CHECK(cudaMemcpy(dm.facet_tex, hm.indices.data(),
                              total * sizeof(int), cudaMemcpyHostToDevice));

    dm.diffuse   = upload_texture(hm.diffuse);
    dm.normalmap = upload_texture(hm.normalmap);
    dm.specular  = upload_texture(hm.specular);

    return dm;
}

void free_device_model(DeviceModel& dm) {
    cudaFree(dm.verts);     dm.verts = nullptr;
    cudaFree(dm.norms);     dm.norms = nullptr;
    cudaFree(dm.tex);       dm.tex = nullptr;
    cudaFree(dm.facet_vrt); dm.facet_vrt = nullptr;
    cudaFree(dm.facet_nrm); dm.facet_nrm = nullptr;
    cudaFree(dm.facet_tex); dm.facet_tex = nullptr;
    if (dm.diffuse.data)   { cudaFree(dm.diffuse.data);   dm.diffuse.data   = nullptr; }
    if (dm.normalmap.data) { cudaFree(dm.normalmap.data); dm.normalmap.data = nullptr; }
    if (dm.specular.data)  { cudaFree(dm.specular.data);  dm.specular.data  = nullptr; }
    dm.nfaces = 0;
}

void render_model(const DeviceModel& dm, const RenderParams& params,
                  uint8_t* d_framebuffer, float* d_zbuffer) {
    launch_render_kernel(dm, params, d_framebuffer, d_zbuffer);
    TINYCUDA_CHECK(cudaDeviceSynchronize());
}

mat<4,4> build_lookat(const vec3& eye, const vec3& center, const vec3& up) {
    vec3 n = normalized(eye - center);
    vec3 l = normalized(cross(up, n));
    vec3 m = normalized(cross(n, l));
    mat<4,4> R = {{{l.x,l.y,l.z,0}, {m.x,m.y,m.z,0}, {n.x,n.y,n.z,0}, {0,0,0,1}}};
    mat<4,4> T = {{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
    return R * T;
}

mat<4,4> build_perspective(float f) {
    return {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,-1.f/f,1}}};
}

mat<4,4> build_viewport(int x, int y, int w, int h) {
    return {{{w/2.f, 0, 0, x+w/2.f}, {0, h/2.f, 0, y+h/2.f}, {0,0,1,0}, {0,0,0,1}}};
}

bool write_tga(const char* path, const uint8_t* bgr, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::fprintf(stderr, "cannot open %s for writing\n", path);
        return false;
    }
    uint8_t header[18] = {};
    header[2]  = 2;
    header[12] = static_cast<uint8_t>(width  & 0xFF);
    header[13] = static_cast<uint8_t>((width  >> 8) & 0xFF);
    header[14] = static_cast<uint8_t>(height & 0xFF);
    header[15] = static_cast<uint8_t>((height >> 8) & 0xFF);
    header[16] = 24;
    header[17] = 0;
    out.write(reinterpret_cast<const char*>(header), 18);
    out.write(reinterpret_cast<const char*>(bgr),
              static_cast<std::streamsize>(width) * height * 3);
    uint8_t footer[26] = {};
    const char* sig = "TRUEVISION-XFILE.";
    std::memcpy(footer + 8, sig, 18);
    out.write(reinterpret_cast<const char*>(footer), 26);
    return out.good();
}

} // namespace tinycuda
