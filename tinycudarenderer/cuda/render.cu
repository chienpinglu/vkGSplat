#include "render.cuh"
#include <cstdio>
#include <cfloat>

__device__ vec4 load_vert(const DeviceModel &m, int face, int vert) {
    int idx = m.facet_vrt[face * 3 + vert] * 4;
    return {m.verts[idx], m.verts[idx+1], m.verts[idx+2], m.verts[idx+3]};
}

__device__ vec4 load_normal(const DeviceModel &m, int face, int vert) {
    int idx = m.facet_nrm[face * 3 + vert] * 4;
    return {m.norms[idx], m.norms[idx+1], m.norms[idx+2], m.norms[idx+3]};
}

__device__ vec2 load_uv(const DeviceModel &m, int face, int vert) {
    int idx = m.facet_tex[face * 3 + vert] * 2;
    return {m.tex[idx], m.tex[idx+1]};
}

__device__ vec4 sample_normal_map(const DeviceModel &m, vec2 uv) {
    vec4 c = tex2d(m.normalmap, uv.x, uv.y);
    return normalized(vec4{c[2], c[1], c[0], 0.f} * 2.f / 255.f - vec4{1,1,1,0});
}

__device__ void phong_fragment(const DeviceModel &m, const vec4 &l,
                               const vec2 varying_uv[3], const vec4 varying_nrm[3],
                               const vec4 tri[3], const vec3 &bar,
                               bool &discard, uint8_t out_bgra[4]) {
    mat<2,4> E  = {tri[1] - tri[0], tri[2] - tri[0]};
    mat<2,2> U  = {varying_uv[1] - varying_uv[0], varying_uv[2] - varying_uv[0]};
    mat<2,4> T  = U.invert() * E;
    mat<4,4> D  = {normalized(T[0]),
                   normalized(T[1]),
                   normalized(varying_nrm[0]*bar[0] + varying_nrm[1]*bar[1] + varying_nrm[2]*bar[2]),
                   {0,0,0,1}};
    vec2 uv = varying_uv[0] * bar[0] + varying_uv[1] * bar[1] + varying_uv[2] * bar[2];
    vec4 n  = normalized(D.transpose() * sample_normal_map(m, uv));
    vec4 r  = normalized(n * (n * l) * 2.f - l);

    float ambient  = 0.4f;
    float diffuse  = 1.f * fmaxf(0.f, n * l);

    vec4 spec_sample = tex2d(m.specular, uv.x, uv.y);
    float specular = (0.5f + 2.f * spec_sample[0] / 255.f) * powf(fmaxf(r.z, 0.f), 35.f);

    vec4 diffuse_color = tex2d(m.diffuse, uv.x, uv.y);
    float intensity = ambient + diffuse + specular;
    out_bgra[0] = static_cast<uint8_t>(fminf(255.f, diffuse_color[0] * intensity));
    out_bgra[1] = static_cast<uint8_t>(fminf(255.f, diffuse_color[1] * intensity));
    out_bgra[2] = static_cast<uint8_t>(fminf(255.f, diffuse_color[2] * intensity));
    out_bgra[3] = 255;
    discard = false;
}

__global__ void render_kernel(DeviceModel model, RenderParams params,
                              uint8_t *framebuffer, float *zbuffer) {
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= model.nfaces) return;

    mat<4,4> MV  = params.ModelView;
    mat<4,4> P   = params.Perspective;
    mat<4,4> VP  = params.Viewport;
    vec4 light   = params.light_dir;
    int width    = params.width;
    int height   = params.height;

    vec2 varying_uv[3];
    vec4 varying_nrm[3];
    vec4 tri[3];
    vec4 clip[3];

    mat<4,4> MV_IT = MV.invert_transpose();
    for (int v = 0; v < 3; v++) {
        varying_uv[v]  = load_uv(model, face, v);
        varying_nrm[v] = MV_IT * load_normal(model, face, v);
        vec4 gl_Position = MV * load_vert(model, face, v);
        tri[v] = gl_Position;
        clip[v] = P * gl_Position;
    }

    vec4 ndc[3] = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };
    vec2 screen[3] = { (VP*ndc[0]).xy(), (VP*ndc[1]).xy(), (VP*ndc[2]).xy() };

    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.f},
                      {screen[1].x, screen[1].y, 1.f},
                      {screen[2].x, screen[2].y, 1.f} }};
    if (ABC.det() < 1.f) return;

    float bbminx = fminf(fminf(screen[0].x, screen[1].x), screen[2].x);
    float bbmaxx = fmaxf(fmaxf(screen[0].x, screen[1].x), screen[2].x);
    float bbminy = fminf(fminf(screen[0].y, screen[1].y), screen[2].y);
    float bbmaxy = fmaxf(fmaxf(screen[0].y, screen[1].y), screen[2].y);

    int xmin = max(static_cast<int>(bbminx), 0);
    int xmax = min(static_cast<int>(bbmaxx), width - 1);
    int ymin = max(static_cast<int>(bbminy), 0);
    int ymax = min(static_cast<int>(bbmaxy), height - 1);

    mat<3,3> ABC_IT = ABC.invert_transpose();

    for (int x = xmin; x <= xmax; x++) {
        for (int y = ymin; y <= ymax; y++) {
            vec3 bc_screen = ABC_IT * vec3{static_cast<float>(x), static_cast<float>(y), 1.f};
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

            vec3 bc_clip = {bc_screen.x / clip[0].w, bc_screen.y / clip[1].w, bc_screen.z / clip[2].w};
            bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);

            float z = bc_screen * vec3{ndc[0].z, ndc[1].z, ndc[2].z};

            int pix = x + y * width;
            float old_z = zbuffer[pix];
            if (z <= old_z) continue;

            float prev = atomicMaxFloat(&zbuffer[pix], z);
            if (prev >= z && prev != old_z) continue;

            bool discard_frag;
            uint8_t color[4];
            phong_fragment(model, light, varying_uv, varying_nrm, tri, bc_clip,
                           discard_frag, color);
            if (discard_frag) continue;

            int fb_idx = pix * 3;
            framebuffer[fb_idx + 0] = color[0];
            framebuffer[fb_idx + 1] = color[1];
            framebuffer[fb_idx + 2] = color[2];
        }
    }
}

void launch_render_kernel(const DeviceModel &model, const RenderParams &params,
                          uint8_t *d_framebuffer, float *d_zbuffer) {
    int threads = 256;
    int blocks = (model.nfaces + threads - 1) / threads;
    render_kernel<<<blocks, threads>>>(model, params, d_framebuffer, d_zbuffer);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
}
