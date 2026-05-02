#pragma once
#include "geometry.cuh"
#include <cstdint>

struct DeviceTexture {
    uint8_t *data;
    int w, h, bpp;
};

struct DeviceModel {
    float *verts;       // vec4 packed as 4 floats, nverts*4
    float *norms;       // vec4 packed as 4 floats, nnorms*4
    float *tex;         // vec2 packed as 2 floats, ntex*2
    int   *facet_vrt;   // nfaces*3
    int   *facet_nrm;   // nfaces*3
    int   *facet_tex;   // nfaces*3
    int    nfaces;
    DeviceTexture diffuse;
    DeviceTexture normalmap;
    DeviceTexture specular;
};

struct RenderParams {
    mat<4,4> ModelView;
    mat<4,4> Perspective;
    mat<4,4> Viewport;
    vec4 light_dir;     // light direction in eye coordinates
    int width;
    int height;
};

#ifdef __CUDACC__

__device__ inline float atomicMaxFloat(float *addr, float value) {
    int *addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) return __int_as_float(assumed);
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ inline vec4 tex2d(const DeviceTexture &tex, float u, float v) {
    int x = static_cast<int>(u * tex.w);
    int y = static_cast<int>(v * tex.h);
    if (x < 0) x = 0; if (x >= tex.w) x = tex.w - 1;
    if (y < 0) y = 0; if (y >= tex.h) y = tex.h - 1;
    const uint8_t *p = tex.data + (x + y * tex.w) * tex.bpp;
    vec4 c;
    if (tex.bpp >= 3) { c[0] = p[0]; c[1] = p[1]; c[2] = p[2]; }
    else              { c[0] = c[1] = c[2] = p[0]; }
    if (tex.bpp == 4) c[3] = p[3];
    else              c[3] = 255.f;
    return c;
}

#endif

void launch_render_kernel(const DeviceModel &model, const RenderParams &params,
                          uint8_t *d_framebuffer, float *d_zbuffer);
