// SPDX-License-Identifier: Apache-2.0
//
// tinyvk driver — implements the Vulkan-shaped tinyvk.h entry points by
// routing every command into the CUDA renderer in ../cuda/.
//
// This is the architectural punchline of the vkGSplat thesis in
// miniature: the *application* speaks a Vulkan-style API; the *driver*
// underneath is a software renderer expressed entirely in CUDA compute.
// No fixed-function graphics hardware is touched on the device side;
// no Vulkan SDK is needed on the host side.
//
// The translation unit is plain C++ (host compiler) and links against
// tinycudarenderer_cuda_lib for the device-side kernels.

#include "tinyvk.h"

#include "../cuda/cuda_renderer.h"
#include "../cuda/host_loader.h"
#include "../cuda/render.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <variant>
#include <vector>

// ===========================================================================
// Opaque handle definitions
// ===========================================================================

struct TvkInstance_T {
    char appName[64] = "tinyvk-app";
};

struct TvkPhysicalDevice_T {
    char name[64]    = "CUDA backend";
    char backend[16] = "cuda";
};

struct TvkDevice_T {
    TvkPhysicalDevice physical = nullptr;
};

struct TvkQueue_T {
    TvkDevice device = nullptr;
};

struct TvkImage_T {
    uint32_t width  = 0;
    uint32_t height = 0;
    TvkFormat format = TVK_FORMAT_UNDEFINED;
    uint8_t* d_pixels = nullptr;       // device, packed BGR (TGA convention)
    float*   d_depth  = nullptr;       // device, optional companion depth buffer
};

struct TvkBuffer_T {
    uint64_t size = 0;
    void*    host_ptr = nullptr;       // simple host-shadow buffer (pedagogical)
};

struct TvkMeshVKGSPLAT_T {
    DeviceModel device_model{};        // the CUDA-uploaded mesh + textures
    bool        valid = false;
};

struct TvkFence_T { bool signalled = false; };

// ===========================================================================
// Recorded commands. We model a Vulkan-style command buffer as a vector of
// tagged unions; tvkQueueSubmit walks the list and translates each entry
// into a CUDA call.
// ===========================================================================
namespace {

struct CmdBeginRendering {
    TvkImage image;
    float clearB, clearG, clearR;
};
struct CmdEndRendering {};
struct CmdDrawMesh {
    TvkMeshVKGSPLAT mesh;
    TvkMeshDrawInfoVKGSPLAT info;
};

using Command = std::variant<CmdBeginRendering, CmdEndRendering, CmdDrawMesh>;

} // namespace

struct TvkCommandBuffer_T {
    TvkDevice            device = nullptr;
    bool                 recording = false;
    std::vector<Command> commands;
};

// ===========================================================================
// Globals — a single physical device and a single queue, mirroring
// the smallest possible Vulkan environment.
// ===========================================================================
namespace {

TvkPhysicalDevice_T g_physical_device;
TvkQueue_T          g_queue;

// Convert a column-major float[16] (Vulkan/GLSL convention) into the
// renderer's row-major mat<4,4>.
mat<4,4> mat4_from_columns(const float c[16]) {
    mat<4,4> m;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            m[row][col] = c[col * 4 + row];
        }
    }
    return m;
}

// Translate a TvkTextureDataVKGSPLAT into the HostTexture the CUDA upload
// path expects.
HostTexture to_host_texture(const TvkTextureDataVKGSPLAT& src) {
    HostTexture ht;
    ht.w   = static_cast<int>(src.width);
    ht.h   = static_cast<int>(src.height);
    ht.bpp = static_cast<int>(src.bytesPerPixel);
    if (!src.pPixels || ht.w == 0 || ht.h == 0 || ht.bpp == 0) return ht;
    const std::size_t nbytes =
        static_cast<std::size_t>(ht.w) * ht.h * ht.bpp;
    ht.data.assign(src.pPixels, src.pPixels + nbytes);
    return ht;
}

void execute_begin(const CmdBeginRendering& c) {
    if (!c.image || !c.image->d_pixels) return;
    tinycuda::clear_framebuffer(c.image->d_pixels,
                                static_cast<int>(c.image->width),
                                static_cast<int>(c.image->height),
                                static_cast<uint8_t>(c.clearB * 255.f),
                                static_cast<uint8_t>(c.clearG * 255.f),
                                static_cast<uint8_t>(c.clearR * 255.f));
    if (c.image->d_depth) {
        tinycuda::clear_zbuffer(c.image->d_depth,
                                static_cast<int>(c.image->width),
                                static_cast<int>(c.image->height),
                                -1000.f);
    }
}

void execute_draw(TvkImage attachment, const CmdDrawMesh& c) {
    if (!attachment || !attachment->d_pixels || !attachment->d_depth) return;
    if (!c.mesh || !c.mesh->valid) return;

    RenderParams params;
    params.ModelView   = mat4_from_columns(c.info.modelView);
    params.Perspective = mat4_from_columns(c.info.perspective);
    params.Viewport    = mat4_from_columns(c.info.viewport);
    params.light_dir   = vec4{ c.info.lightDirEye[0],
                               c.info.lightDirEye[1],
                               c.info.lightDirEye[2],
                               c.info.lightDirEye[3] };
    params.width  = static_cast<int>(attachment->width);
    params.height = static_cast<int>(attachment->height);

    tinycuda::render_model(c.mesh->device_model, params,
                           attachment->d_pixels, attachment->d_depth);
}

} // namespace

// ===========================================================================
// Entry-point implementations
// ===========================================================================

TvkResult tvkCreateInstance(const TvkInstanceCreateInfo* pCreateInfo,
                            TvkInstance*                 pInstance) {
    if (!pInstance) return TVK_ERROR_INITIALIZATION_FAILED;
    auto* inst = new (std::nothrow) TvkInstance_T;
    if (!inst) return TVK_ERROR_OUT_OF_HOST_MEMORY;
    if (pCreateInfo && pCreateInfo->pApplicationInfo &&
        pCreateInfo->pApplicationInfo->pApplicationName) {
        std::strncpy(inst->appName,
                     pCreateInfo->pApplicationInfo->pApplicationName,
                     sizeof(inst->appName) - 1);
    }
    *pInstance = inst;
    return TVK_SUCCESS;
}

void tvkDestroyInstance(TvkInstance instance) { delete instance; }

TvkResult tvkEnumeratePhysicalDevices(TvkInstance        /*instance*/,
                                      uint32_t*          pCount,
                                      TvkPhysicalDevice* pDevices) {
    if (!pCount) return TVK_ERROR_INITIALIZATION_FAILED;
    if (!pDevices) {
        *pCount = 1;
        return TVK_SUCCESS;
    }
    if (*pCount < 1) return TVK_ERROR_INITIALIZATION_FAILED;
    pDevices[0] = &g_physical_device;
    *pCount = 1;
    return TVK_SUCCESS;
}

void tvkGetPhysicalDeviceProperties(TvkPhysicalDevice            pd,
                                    TvkPhysicalDeviceProperties* pProps) {
    if (!pd || !pProps) return;
    std::strncpy(pProps->deviceName,  pd->name,    sizeof(pProps->deviceName) - 1);
    std::strncpy(pProps->backendName, pd->backend, sizeof(pProps->backendName) - 1);
    pProps->apiVersion = 0x00010000;
}

TvkResult tvkCreateDevice(TvkPhysicalDevice          physicalDevice,
                          const TvkDeviceCreateInfo* /*pCreateInfo*/,
                          TvkDevice*                 pDevice) {
    if (!physicalDevice || !pDevice) return TVK_ERROR_INITIALIZATION_FAILED;
    auto* d = new (std::nothrow) TvkDevice_T;
    if (!d) return TVK_ERROR_OUT_OF_HOST_MEMORY;
    d->physical = physicalDevice;
    g_queue.device = d;
    *pDevice = d;
    return TVK_SUCCESS;
}

void tvkDestroyDevice(TvkDevice device) {
    if (!device) return;
    if (g_queue.device == device) g_queue.device = nullptr;
    delete device;
}

void tvkGetDeviceQueue(TvkDevice device, TvkQueue* pQueue) {
    if (!device || !pQueue) return;
    g_queue.device = device;
    *pQueue = &g_queue;
}

// --- Resources -------------------------------------------------------------

TvkResult tvkCreateImage(TvkDevice                 device,
                         const TvkImageCreateInfo* pCreateInfo,
                         TvkImage*                 pImage) {
    if (!device || !pCreateInfo || !pImage) return TVK_ERROR_INITIALIZATION_FAILED;
    if (pCreateInfo->format != TVK_FORMAT_R8G8B8_UNORM) {
        return TVK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    auto* img = new (std::nothrow) TvkImage_T;
    if (!img) return TVK_ERROR_OUT_OF_HOST_MEMORY;
    img->width  = pCreateInfo->extent.width;
    img->height = pCreateInfo->extent.height;
    img->format = pCreateInfo->format;
    img->d_pixels = tinycuda::alloc_framebuffer(static_cast<int>(img->width),
                                                static_cast<int>(img->height));
    img->d_depth  = tinycuda::alloc_zbuffer(static_cast<int>(img->width),
                                            static_cast<int>(img->height));
    *pImage = img;
    return TVK_SUCCESS;
}

void tvkDestroyImage(TvkDevice /*device*/, TvkImage image) {
    if (!image) return;
    tinycuda::free_device(image->d_pixels);
    tinycuda::free_device(image->d_depth);
    delete image;
}

void tvkGetImageData(TvkDevice /*device*/, TvkImage image,
                     uint8_t* pHostBuffer, uint32_t bufferSize) {
    if (!image || !pHostBuffer) return;
    const std::size_t needed = static_cast<std::size_t>(image->width) * image->height * 3;
    if (bufferSize < needed) return;
    tinycuda::download_framebuffer(image->d_pixels, pHostBuffer, needed);
}

TvkResult tvkCreateBuffer(TvkDevice                  /*device*/,
                          const TvkBufferCreateInfo* pCreateInfo,
                          TvkBuffer*                 pBuffer) {
    if (!pCreateInfo || !pBuffer) return TVK_ERROR_INITIALIZATION_FAILED;
    auto* buf = new (std::nothrow) TvkBuffer_T;
    if (!buf) return TVK_ERROR_OUT_OF_HOST_MEMORY;
    buf->size = pCreateInfo->size;
    buf->host_ptr = std::malloc(pCreateInfo->size);
    if (!buf->host_ptr) { delete buf; return TVK_ERROR_OUT_OF_HOST_MEMORY; }
    *pBuffer = buf;
    return TVK_SUCCESS;
}

void tvkDestroyBuffer(TvkDevice /*device*/, TvkBuffer buffer) {
    if (!buffer) return;
    std::free(buffer->host_ptr);
    delete buffer;
}

TvkResult tvkCreateMeshVKGSPLAT(TvkDevice                       /*device*/,
                               const TvkMeshCreateInfoVKGSPLAT* pCreateInfo,
                               TvkMeshVKGSPLAT*                 pMesh) {
    if (!pCreateInfo || !pMesh) return TVK_ERROR_INITIALIZATION_FAILED;

    HostModel hm;
    const uint32_t vc = pCreateInfo->geometry.vertexCount;
    if (vc == 0 || vc % 3 != 0) return TVK_ERROR_INITIALIZATION_FAILED;

    hm.nfaces = static_cast<int>(vc / 3);
    hm.verts.assign(pCreateInfo->geometry.pVertices,
                    pCreateInfo->geometry.pVertices + vc * 4);
    hm.norms.assign(pCreateInfo->geometry.pNormals,
                    pCreateInfo->geometry.pNormals  + vc * 4);
    hm.tex.assign(pCreateInfo->geometry.pUVs,
                  pCreateInfo->geometry.pUVs        + vc * 2);
    hm.indices.resize(vc);
    for (uint32_t i = 0; i < vc; ++i) hm.indices[i] = static_cast<int>(i);

    hm.diffuse   = to_host_texture(pCreateInfo->diffuse);
    hm.normalmap = to_host_texture(pCreateInfo->normalMap);
    hm.specular  = to_host_texture(pCreateInfo->specular);

    auto* mesh = new (std::nothrow) TvkMeshVKGSPLAT_T;
    if (!mesh) return TVK_ERROR_OUT_OF_HOST_MEMORY;
    mesh->device_model = tinycuda::upload_model_to_device(hm);
    mesh->valid = true;
    *pMesh = mesh;
    return TVK_SUCCESS;
}

void tvkDestroyMeshVKGSPLAT(TvkDevice /*device*/, TvkMeshVKGSPLAT mesh) {
    if (!mesh) return;
    if (mesh->valid) tinycuda::free_device_model(mesh->device_model);
    delete mesh;
}

// --- Command buffer --------------------------------------------------------

TvkResult tvkAllocateCommandBuffers(TvkDevice         device,
                                    TvkCommandBuffer* pCommandBuffer) {
    if (!device || !pCommandBuffer) return TVK_ERROR_INITIALIZATION_FAILED;
    auto* cb = new (std::nothrow) TvkCommandBuffer_T;
    if (!cb) return TVK_ERROR_OUT_OF_HOST_MEMORY;
    cb->device = device;
    *pCommandBuffer = cb;
    return TVK_SUCCESS;
}

void tvkFreeCommandBuffers(TvkDevice /*device*/, TvkCommandBuffer cb) {
    delete cb;
}

TvkResult tvkBeginCommandBuffer(TvkCommandBuffer cb) {
    if (!cb) return TVK_ERROR_INVALID_HANDLE;
    cb->commands.clear();
    cb->recording = true;
    return TVK_SUCCESS;
}

TvkResult tvkEndCommandBuffer(TvkCommandBuffer cb) {
    if (!cb) return TVK_ERROR_INVALID_HANDLE;
    cb->recording = false;
    return TVK_SUCCESS;
}

void tvkCmdBeginRendering(TvkCommandBuffer cb, TvkImage image,
                          float clearB, float clearG, float clearR) {
    if (!cb || !cb->recording) return;
    cb->commands.emplace_back(CmdBeginRendering{ image, clearB, clearG, clearR });
}

void tvkCmdEndRendering(TvkCommandBuffer cb) {
    if (!cb || !cb->recording) return;
    cb->commands.emplace_back(CmdEndRendering{});
}

void tvkCmdDrawMeshVKGSPLAT(TvkCommandBuffer              cb,
                           TvkMeshVKGSPLAT                mesh,
                           const TvkMeshDrawInfoVKGSPLAT* pDrawInfo) {
    if (!cb || !cb->recording || !pDrawInfo) return;
    cb->commands.emplace_back(CmdDrawMesh{ mesh, *pDrawInfo });
}

// --- Submit / wait ---------------------------------------------------------

TvkResult tvkQueueSubmit(TvkQueue /*queue*/,
                         TvkCommandBuffer cb,
                         TvkFence /*fence*/) {
    if (!cb) return TVK_ERROR_INVALID_HANDLE;
    TvkImage current_attachment = nullptr;
    for (const auto& cmd : cb->commands) {
        std::visit([&](const auto& payload) {
            using T = std::decay_t<decltype(payload)>;
            if constexpr (std::is_same_v<T, CmdBeginRendering>) {
                current_attachment = payload.image;
                execute_begin(payload);
            } else if constexpr (std::is_same_v<T, CmdEndRendering>) {
                current_attachment = nullptr;
            } else if constexpr (std::is_same_v<T, CmdDrawMesh>) {
                execute_draw(current_attachment, payload);
            }
        }, cmd);
    }
    return TVK_SUCCESS;
}

TvkResult tvkDeviceWaitIdle(TvkDevice /*device*/) {
    // execute_*() already cudaDeviceSynchronize via tinycuda::render_model;
    // nothing else to wait on in this minimal driver.
    return TVK_SUCCESS;
}
