// SPDX-License-Identifier: Apache-2.0
//
// tinyvk — a minimal Vulkan-shaped C API whose driver is implemented on
// top of CUDA compute kernels. Pedagogical proof-of-concept for the
// vkSplat thesis: keep the API surface, replace the silicon.
//
// The naming is deliberately Vulkan-flavoured but uses a Tvk*/tvk*
// prefix so it can coexist with the real Vulkan SDK in one process
// without symbol collisions. Application code that knows Vulkan should
// recognise every entry point by analogy:
//
//      Vulkan                              tinyvk
//      VkInstance                          TvkInstance
//      vkCreateInstance                    tvkCreateInstance
//      VkPhysicalDevice                    TvkPhysicalDevice
//      VkDevice                            TvkDevice
//      VkQueue                             TvkQueue
//      VkCommandBuffer                     TvkCommandBuffer
//      VkImage                             TvkImage
//      vkCmdBeginRendering                 tvkCmdBeginRendering
//      vkQueueSubmit                       tvkQueueSubmit
//      vkDeviceWaitIdle                    tvkDeviceWaitIdle
//      VK_*_EXT (vendor extension)         Tvk*VKSPLAT (vendor extension)
//
// The single non-classical addition is tvkCmdDrawMeshVKSPLAT, a
// "draw a mesh as a compute primitive" command that mirrors the
// VK_VKSPLAT_gaussian_splatting extension declared in the parent
// vkSplat project (../include/vksplat/extensions/) but for triangle
// meshes instead of Gaussians.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status codes (subset of VkResult)
// ---------------------------------------------------------------------------
typedef enum TvkResult {
    TVK_SUCCESS                          =  0,
    TVK_ERROR_OUT_OF_HOST_MEMORY         = -1,
    TVK_ERROR_OUT_OF_DEVICE_MEMORY       = -2,
    TVK_ERROR_INITIALIZATION_FAILED      = -3,
    TVK_ERROR_DEVICE_LOST                = -4,
    TVK_ERROR_FORMAT_NOT_SUPPORTED       = -5,
    TVK_ERROR_FEATURE_NOT_PRESENT        = -6,
    TVK_ERROR_INVALID_HANDLE             = -7,
    TVK_ERROR_UNKNOWN                    = -100,
} TvkResult;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------
typedef struct TvkInstance_T*       TvkInstance;
typedef struct TvkPhysicalDevice_T* TvkPhysicalDevice;
typedef struct TvkDevice_T*         TvkDevice;
typedef struct TvkQueue_T*          TvkQueue;
typedef struct TvkCommandBuffer_T*  TvkCommandBuffer;
typedef struct TvkImage_T*          TvkImage;
typedef struct TvkBuffer_T*         TvkBuffer;
typedef struct TvkFence_T*          TvkFence;
typedef struct TvkMeshVKSPLAT_T*    TvkMeshVKSPLAT;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
typedef enum TvkFormat {
    TVK_FORMAT_UNDEFINED       = 0,
    TVK_FORMAT_R8G8B8_UNORM    = 1,
    TVK_FORMAT_R8G8B8A8_UNORM  = 2,
    TVK_FORMAT_D32_SFLOAT      = 3,
} TvkFormat;

typedef enum TvkBufferUsageFlagBits {
    TVK_BUFFER_USAGE_VERTEX_BUFFER_BIT  = 0x01,
    TVK_BUFFER_USAGE_INDEX_BUFFER_BIT   = 0x02,
    TVK_BUFFER_USAGE_TRANSFER_DST_BIT   = 0x04,
    TVK_BUFFER_USAGE_TRANSFER_SRC_BIT   = 0x08,
} TvkBufferUsageFlagBits;
typedef uint32_t TvkBufferUsageFlags;

typedef enum TvkImageUsageFlagBits {
    TVK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x01,
    TVK_IMAGE_USAGE_TRANSFER_SRC_BIT     = 0x02,
    TVK_IMAGE_USAGE_TRANSFER_DST_BIT     = 0x04,
    TVK_IMAGE_USAGE_STORAGE_BIT          = 0x08,
} TvkImageUsageFlagBits;
typedef uint32_t TvkImageUsageFlags;

// ---------------------------------------------------------------------------
// Structs (sType field omitted to keep this minimal; the real VkSplat
// implementation in ../../include/vksplat/ uses the full sType pattern.)
// ---------------------------------------------------------------------------
typedef struct TvkApplicationInfo {
    const char* pApplicationName;
    uint32_t    applicationVersion;
    const char* pEngineName;
    uint32_t    apiVersion;
} TvkApplicationInfo;

typedef struct TvkInstanceCreateInfo {
    const TvkApplicationInfo* pApplicationInfo;
    uint32_t                  enabledExtensionCount;
    const char* const*        ppEnabledExtensionNames;
} TvkInstanceCreateInfo;

typedef struct TvkPhysicalDeviceProperties {
    char     deviceName[64];
    char     backendName[16];     // e.g. "cuda"
    uint32_t apiVersion;
} TvkPhysicalDeviceProperties;

typedef struct TvkDeviceCreateInfo {
    TvkPhysicalDevice  physicalDevice;
    uint32_t           enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} TvkDeviceCreateInfo;

typedef struct TvkExtent2D {
    uint32_t width;
    uint32_t height;
} TvkExtent2D;

typedef struct TvkImageCreateInfo {
    TvkExtent2D        extent;
    TvkFormat          format;
    TvkImageUsageFlags usage;
} TvkImageCreateInfo;

typedef struct TvkBufferCreateInfo {
    uint64_t            size;
    TvkBufferUsageFlags usage;
} TvkBufferCreateInfo;

// ---------------------------------------------------------------------------
// VK_VKSPLAT_mesh extension (the compute-primitive mesh draw path)
// ---------------------------------------------------------------------------
#define TVK_VKSPLAT_MESH_EXTENSION_NAME "TVK_VKSPLAT_mesh"

typedef struct TvkMeshGeometryVKSPLAT {
    // Per-vertex arrays, length = vertexCount each (3 vertices per triangle,
    // already de-indexed).
    const float* pVertices;   // 4 floats per vertex (x, y, z, 1)
    const float* pNormals;    // 4 floats per vertex (nx, ny, nz, 0)
    const float* pUVs;        // 2 floats per vertex (u, v)
    uint32_t     vertexCount; // = triangleCount * 3
} TvkMeshGeometryVKSPLAT;

typedef struct TvkTextureDataVKSPLAT {
    const uint8_t* pPixels;   // tightly-packed BGR or BGRA
    uint32_t       width;
    uint32_t       height;
    uint32_t       bytesPerPixel; // 1 = grayscale, 3 = BGR, 4 = BGRA
} TvkTextureDataVKSPLAT;

typedef struct TvkMeshCreateInfoVKSPLAT {
    TvkMeshGeometryVKSPLAT geometry;
    TvkTextureDataVKSPLAT  diffuse;
    TvkTextureDataVKSPLAT  normalMap;
    TvkTextureDataVKSPLAT  specular;
} TvkMeshCreateInfoVKSPLAT;

typedef struct TvkMeshDrawInfoVKSPLAT {
    // Column-major 4x4 matrices; same convention as Vulkan/GLSL.
    float modelView[16];
    float perspective[16];
    float viewport[16];
    float lightDirEye[4];     // light direction already in eye coordinates
    float clearColor[3];      // BGR clear before the draw (ignored if mid-pass)
} TvkMeshDrawInfoVKSPLAT;

// ---------------------------------------------------------------------------
// Entry points — instance / device / queue
// ---------------------------------------------------------------------------
TvkResult tvkCreateInstance(const TvkInstanceCreateInfo* pCreateInfo,
                            TvkInstance*                  pInstance);
void      tvkDestroyInstance(TvkInstance instance);

TvkResult tvkEnumeratePhysicalDevices(TvkInstance        instance,
                                      uint32_t*          pPhysicalDeviceCount,
                                      TvkPhysicalDevice* pPhysicalDevices);

void      tvkGetPhysicalDeviceProperties(TvkPhysicalDevice            pd,
                                         TvkPhysicalDeviceProperties* pProps);

TvkResult tvkCreateDevice(TvkPhysicalDevice          physicalDevice,
                          const TvkDeviceCreateInfo* pCreateInfo,
                          TvkDevice*                 pDevice);
void      tvkDestroyDevice(TvkDevice device);

void      tvkGetDeviceQueue(TvkDevice device, TvkQueue* pQueue);

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------
TvkResult tvkCreateImage(TvkDevice                  device,
                         const TvkImageCreateInfo*  pCreateInfo,
                         TvkImage*                  pImage);
void      tvkDestroyImage(TvkDevice device, TvkImage image);
void      tvkGetImageData(TvkDevice device, TvkImage image,
                          uint8_t* pHostBuffer, uint32_t bufferSize);

TvkResult tvkCreateBuffer(TvkDevice                   device,
                          const TvkBufferCreateInfo*  pCreateInfo,
                          TvkBuffer*                  pBuffer);
void      tvkDestroyBuffer(TvkDevice device, TvkBuffer buffer);

// vkSplat extension: a mesh is a compute primitive owned by the driver.
TvkResult tvkCreateMeshVKSPLAT(TvkDevice                       device,
                               const TvkMeshCreateInfoVKSPLAT* pCreateInfo,
                               TvkMeshVKSPLAT*                 pMesh);
void      tvkDestroyMeshVKSPLAT(TvkDevice device, TvkMeshVKSPLAT mesh);

// ---------------------------------------------------------------------------
// Command buffer recording
// ---------------------------------------------------------------------------
TvkResult tvkAllocateCommandBuffers(TvkDevice         device,
                                    TvkCommandBuffer* pCommandBuffer);
void      tvkFreeCommandBuffers(TvkDevice         device,
                                TvkCommandBuffer  commandBuffer);

TvkResult tvkBeginCommandBuffer(TvkCommandBuffer commandBuffer);
TvkResult tvkEndCommandBuffer  (TvkCommandBuffer commandBuffer);

void tvkCmdBeginRendering(TvkCommandBuffer commandBuffer,
                          TvkImage         colorAttachment,
                          float            clearB,
                          float            clearG,
                          float            clearR);
void tvkCmdEndRendering  (TvkCommandBuffer commandBuffer);

void tvkCmdDrawMeshVKSPLAT(TvkCommandBuffer              commandBuffer,
                           TvkMeshVKSPLAT                mesh,
                           const TvkMeshDrawInfoVKSPLAT* pDrawInfo);

// ---------------------------------------------------------------------------
// Submission and synchronisation
// ---------------------------------------------------------------------------
TvkResult tvkQueueSubmit  (TvkQueue queue,
                           TvkCommandBuffer commandBuffer,
                           TvkFence fence);
TvkResult tvkDeviceWaitIdle(TvkDevice device);

#ifdef __cplusplus
} // extern "C"
#endif
