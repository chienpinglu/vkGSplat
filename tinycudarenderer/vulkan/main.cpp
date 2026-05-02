// SPDX-License-Identifier: Apache-2.0
//
// tinycudarenderer / vulkan — a minimal Vulkan-shaped application that
// renders a textured mesh by calling exclusively into the tinyvk API
// (../vulkan/tinyvk.h). The driver underneath is implemented in CUDA
// (see tinyvk_driver.cpp); this translation unit knows nothing about
// CUDA, cudaMalloc, kernels, or device pointers.
//
// Read alongside ../main.cpp (CPU) and ../cuda/main.cu (raw CUDA) to
// see the same job expressed at three abstraction levels:
//
//   ../main.cpp           : direct CPU rasterizer.
//   ../cuda/main.cu       : direct CUDA rasterizer (raw kernels).
//   ./main.cpp (this file): Vulkan-shaped API call sequence; the
//                            backend is CUDA but the application is
//                            unaware of it. This is the vkSplat thesis
//                            in microcosm.

#include "tinyvk.h"

// We reuse the existing host-only OBJ + texture loader from the CUDA
// project. It is plain C++ (no CUDA dependencies) and lives in
// ../cuda/ only because that is where it currently belongs.
#include "../cuda/host_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Tiny matrix helpers — Vulkan/GLSL column-major float[16].
// Replicated here so that this file does not pull in any of the renderer's
// internal math types.
// ---------------------------------------------------------------------------

struct vec3f { float x, y, z; };

vec3f sub(vec3f a, vec3f b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
vec3f cross(vec3f a, vec3f b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
float dot(vec3f a, vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
vec3f normalize(vec3f v) {
    const float n = std::sqrt(dot(v, v));
    return n > 0.f ? vec3f{ v.x / n, v.y / n, v.z / n } : vec3f{ 0,0,0 };
}

void identity(float m[16]) {
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.f;
}

// Write a 4x4 in column-major order. mat4_set(m, row, col, v) sets
// m[col*4 + row].
void set(float m[16], int row, int col, float v) { m[col * 4 + row] = v; }

// Build the same matrices as ../cuda/main.cu so the rendered image is
// pixel-identical regardless of which front-end we drove the CUDA
// backend through.
void build_lookat(vec3f eye, vec3f center, vec3f up, float out[16]) {
    vec3f n = normalize(sub(eye, center));
    vec3f l = normalize(cross(up, n));
    vec3f m_ = normalize(cross(n, l));

    float R[16]; identity(R);
    set(R, 0, 0, l.x); set(R, 0, 1, l.y); set(R, 0, 2, l.z);
    set(R, 1, 0, m_.x); set(R, 1, 1, m_.y); set(R, 1, 2, m_.z);
    set(R, 2, 0, n.x); set(R, 2, 1, n.y); set(R, 2, 2, n.z);

    float T[16]; identity(T);
    set(T, 0, 3, -center.x);
    set(T, 1, 3, -center.y);
    set(T, 2, 3, -center.z);

    // out = R * T  (row-major math; layout is column-major)
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += R[k * 4 + r] * T[c * 4 + k];
            set(out, r, c, s);
        }
    }
}

void build_perspective(float f, float out[16]) {
    identity(out);
    set(out, 3, 2, -1.f / f);
    set(out, 3, 3, 1.f);
}

void build_viewport(int x, int y, int w, int h, float out[16]) {
    identity(out);
    set(out, 0, 0, w / 2.f); set(out, 0, 3, x + w / 2.f);
    set(out, 1, 1, h / 2.f); set(out, 1, 3, y + h / 2.f);
    set(out, 2, 2, 1.f);
}

// out = lhs * rhs (column-major operands)
void mat4_mul(const float lhs[16], const float rhs[16], float out[16]) {
    float tmp[16]; std::memset(tmp, 0, sizeof(tmp));
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k)
                tmp[c * 4 + r] += lhs[k * 4 + r] * rhs[c * 4 + k];
    std::memcpy(out, tmp, sizeof(tmp));
}

// Write a 24-bpp uncompressed BGR TGA — matches what the CUDA backend
// produces internally so the bytes can be saved as-is.
bool write_tga(const char* path, const std::vector<uint8_t>& bgr,
               int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    uint8_t header[18] = {};
    header[2]  = 2;
    header[12] = static_cast<uint8_t>(width  & 0xFF);
    header[13] = static_cast<uint8_t>((width  >> 8) & 0xFF);
    header[14] = static_cast<uint8_t>(height & 0xFF);
    header[15] = static_cast<uint8_t>((height >> 8) & 0xFF);
    header[16] = 24;
    header[17] = 0;
    out.write(reinterpret_cast<const char*>(header), 18);
    out.write(reinterpret_cast<const char*>(bgr.data()),
              static_cast<std::streamsize>(bgr.size()));
    uint8_t footer[26] = {};
    const char* sig = "TRUEVISION-XFILE.";
    std::memcpy(footer + 8, sig, 18);
    out.write(reinterpret_cast<const char*>(footer), 26);
    return out.good();
}

void check(TvkResult r, const char* where) {
    if (r != TVK_SUCCESS) {
        std::fprintf(stderr, "tinyvk error in %s: %d\n", where, static_cast<int>(r));
        std::exit(2);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s obj/model.obj [more.obj ...]\n", argv[0]);
        return 1;
    }

    constexpr int width  = 800;
    constexpr int height = 800;

    // === Vulkan boilerplate, the small version ============================
    TvkApplicationInfo app{};
    app.pApplicationName = "tinycudarenderer-vulkan";
    app.pEngineName      = "tinyvk";
    app.apiVersion       = 0x00010000;

    TvkInstanceCreateInfo ici{};
    ici.pApplicationInfo = &app;
    TvkInstance instance = nullptr;
    check(tvkCreateInstance(&ici, &instance), "tvkCreateInstance");

    uint32_t pd_count = 0;
    check(tvkEnumeratePhysicalDevices(instance, &pd_count, nullptr),
          "tvkEnumeratePhysicalDevices(count)");
    std::vector<TvkPhysicalDevice> physicals(pd_count);
    check(tvkEnumeratePhysicalDevices(instance, &pd_count, physicals.data()),
          "tvkEnumeratePhysicalDevices(devices)");

    TvkPhysicalDeviceProperties props{};
    tvkGetPhysicalDeviceProperties(physicals[0], &props);
    std::fprintf(stderr, "[tinyvk-app] picked physical device: %s (backend=%s)\n",
                 props.deviceName, props.backendName);

    TvkDeviceCreateInfo dci{};
    dci.physicalDevice = physicals[0];
    TvkDevice device = nullptr;
    check(tvkCreateDevice(physicals[0], &dci, &device), "tvkCreateDevice");

    TvkQueue queue = nullptr;
    tvkGetDeviceQueue(device, &queue);

    // === Color attachment ==================================================
    TvkImageCreateInfo color_ci{};
    color_ci.extent = { static_cast<uint32_t>(width),
                        static_cast<uint32_t>(height) };
    color_ci.format = TVK_FORMAT_R8G8B8_UNORM;
    color_ci.usage  = TVK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      TVK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    TvkImage color = nullptr;
    check(tvkCreateImage(device, &color_ci, &color), "tvkCreateImage(color)");

    // === Camera ============================================================
    float view[16], proj[16], viewport[16];
    const vec3f eye{ -1.f, 0.f, 2.f };
    const vec3f center{ 0.f, 0.f, 0.f };
    const vec3f up{ 0.f, 1.f, 0.f };
    build_lookat(eye, center, up, view);
    build_perspective(std::sqrt((eye.x - center.x) * (eye.x - center.x) +
                                (eye.y - center.y) * (eye.y - center.y) +
                                (eye.z - center.z) * (eye.z - center.z)), proj);
    build_viewport(width / 16, height / 16, width * 7 / 8, height * 7 / 8, viewport);

    // Light vector pre-transformed into eye coordinates so the driver
    // doesn't have to know about the world-space light convention.
    float light_world[4] = { 1.f, 1.f, 1.f, 0.f };
    float light_eye[4]   = { 0.f, 0.f, 0.f, 0.f };
    for (int r = 0; r < 4; ++r) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += view[k * 4 + r] * light_world[k];
        light_eye[r] = s;
    }
    {
        const float n = std::sqrt(light_eye[0]*light_eye[0] +
                                  light_eye[1]*light_eye[1] +
                                  light_eye[2]*light_eye[2] +
                                  light_eye[3]*light_eye[3]);
        if (n > 0.f) for (float& v : light_eye) v /= n;
    }

    // === Per-mesh draw =====================================================
    TvkCommandBuffer cmdbuf = nullptr;
    check(tvkAllocateCommandBuffers(device, &cmdbuf), "tvkAllocateCommandBuffers");
    check(tvkBeginCommandBuffer(cmdbuf), "tvkBeginCommandBuffer");

    // Background BGR (177, 195, 209 RGBA matches the CPU baseline).
    tvkCmdBeginRendering(cmdbuf, color,
                         /*B=*/209.f / 255.f,
                         /*G=*/195.f / 255.f,
                         /*R=*/177.f / 255.f);

    std::vector<TvkMeshVKSPLAT> meshes;
    std::vector<HostModel>      pinned_hosts; // keep the pointers alive till submit
    pinned_hosts.reserve(argc - 1);

    for (int m = 1; m < argc; ++m) {
        pinned_hosts.push_back(load_model(argv[m]));
        const HostModel& hm = pinned_hosts.back();

        TvkMeshGeometryVKSPLAT geom{};
        geom.pVertices   = hm.verts.data();
        geom.pNormals    = hm.norms.data();
        geom.pUVs        = hm.tex.data();
        geom.vertexCount = static_cast<uint32_t>(hm.nfaces) * 3u;

        auto fill = [](TvkTextureDataVKSPLAT& t, const HostTexture& s) {
            t.pPixels       = s.data.empty() ? nullptr : s.data.data();
            t.width         = static_cast<uint32_t>(s.w);
            t.height        = static_cast<uint32_t>(s.h);
            t.bytesPerPixel = static_cast<uint32_t>(s.bpp);
        };

        TvkMeshCreateInfoVKSPLAT mci{};
        mci.geometry = geom;
        fill(mci.diffuse,   hm.diffuse);
        fill(mci.normalMap, hm.normalmap);
        fill(mci.specular,  hm.specular);

        TvkMeshVKSPLAT mesh = nullptr;
        check(tvkCreateMeshVKSPLAT(device, &mci, &mesh), "tvkCreateMeshVKSPLAT");
        meshes.push_back(mesh);

        TvkMeshDrawInfoVKSPLAT dinfo{};
        std::memcpy(dinfo.modelView,   view,     sizeof(view));
        std::memcpy(dinfo.perspective, proj,     sizeof(proj));
        std::memcpy(dinfo.viewport,    viewport, sizeof(viewport));
        std::memcpy(dinfo.lightDirEye, light_eye, sizeof(light_eye));
        dinfo.clearColor[0] = 0.f; dinfo.clearColor[1] = 0.f; dinfo.clearColor[2] = 0.f;

        tvkCmdDrawMeshVKSPLAT(cmdbuf, mesh, &dinfo);
    }

    tvkCmdEndRendering(cmdbuf);
    check(tvkEndCommandBuffer(cmdbuf), "tvkEndCommandBuffer");

    check(tvkQueueSubmit(queue, cmdbuf, /*fence=*/nullptr), "tvkQueueSubmit");
    check(tvkDeviceWaitIdle(device), "tvkDeviceWaitIdle");

    // === Readback ==========================================================
    const std::size_t image_bytes = static_cast<std::size_t>(width) * height * 3;
    std::vector<uint8_t> host_pixels(image_bytes);
    tvkGetImageData(device, color, host_pixels.data(),
                    static_cast<uint32_t>(image_bytes));

    if (!write_tga("framebuffer.tga", host_pixels, width, height)) {
        std::fprintf(stderr, "failed to write framebuffer.tga\n");
        return 1;
    }
    std::fprintf(stderr, "[tinyvk-app] wrote framebuffer.tga (%dx%d)\n", width, height);

    // === Teardown ==========================================================
    for (auto m : meshes) tvkDestroyMeshVKSPLAT(device, m);
    tvkFreeCommandBuffers(device, cmdbuf);
    tvkDestroyImage(device, color);
    tvkDestroyDevice(device);
    tvkDestroyInstance(instance);
    return 0;
}
