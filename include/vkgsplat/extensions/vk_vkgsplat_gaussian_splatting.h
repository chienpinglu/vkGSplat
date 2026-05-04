// SPDX-License-Identifier: Apache-2.0
//
// VK_VKGSPLAT_gaussian_splatting — vendor extension declared by the
// position paper (Section 4, "The 3DGS fast path").
//
// Lets a Vulkan application submit Gaussian primitives directly,
// bypassing the classical pipeline objects entirely. The driver
// dispatches to a compute-only 3DGS rasterizer; on vkGSplat the
// implementation is the kernel set in src/cuda/rasterizer.cu.
//
// This header is the *application-facing* declaration. It is not yet
// registered with Khronos; the VK_VKGSPLAT_ prefix follows the
// VK_<TAG>_ vendor convention pending registration.
#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VK_VKGSPLAT_GAUSSIAN_SPLATTING_EXTENSION_NAME "VK_VKGSPLAT_gaussian_splatting"
#define VK_VKGSPLAT_GAUSSIAN_SPLATTING_SPEC_VERSION   1

// New structure types in the VkStructureType space. The reserved
// vendor enum range used here (1000800000+) is illustrative — replace
// once an enum block is allocated.
#define VK_STRUCTURE_TYPE_GAUSSIAN_SPLAT_BUFFER_INFO_VKGSPLAT  ((VkStructureType)1000800000)
#define VK_STRUCTURE_TYPE_GAUSSIAN_SPLAT_RENDER_INFO_VKGSPLAT  ((VkStructureType)1000800001)
#define VK_STRUCTURE_TYPE_GAUSSIAN_SPLAT_FEATURES_VKGSPLAT     ((VkStructureType)1000800002)

// One Gaussian primitive, packed for efficient device-side load. Layout
// matches the v1 CUDA backend (see include/vkgsplat/types.h::Gaussian).
typedef struct VkGaussianPrimitiveVKGSPLAT {
    float    position[3];
    float    scale_log[3];
    float    rotation[4];      // unit quaternion (xyzw)
    float    opacity_logit;
    uint32_t sh_offset;        // index into the SH coefficient buffer
} VkGaussianPrimitiveVKGSPLAT;

typedef struct VkGaussianSplatBufferInfoVKGSPLAT {
    VkStructureType sType;
    const void*     pNext;
    VkBuffer        primitiveBuffer;        // array of VkGaussianPrimitiveVKGSPLAT
    VkDeviceSize    primitiveBufferOffset;
    uint32_t        primitiveCount;
    VkBuffer        sphericalHarmonicsBuffer; // float3[primitiveCount * sh_coeffs]
    VkDeviceSize    sphericalHarmonicsOffset;
    uint32_t        sphericalHarmonicsDegree; // 0..3 supported in v1
} VkGaussianSplatBufferInfoVKGSPLAT;

typedef struct VkGaussianSplatRenderInfoVKGSPLAT {
    VkStructureType sType;
    const void*     pNext;
    VkImage         outputImage;
    VkExtent2D      outputExtent;
    float           viewMatrix[16];      // column-major
    float           projectionMatrix[16];
    float           background[4];
    uint32_t        seed;                // for deterministic dithering
    VkBool32        deterministicOrder;
} VkGaussianSplatRenderInfoVKGSPLAT;

typedef struct VkPhysicalDeviceGaussianSplatFeaturesVKGSPLAT {
    VkStructureType sType;
    void*           pNext;
    VkBool32        gaussianSplatting;
    VkBool32        deterministicSplatOrdering;
    VkBool32        antiAliasedFilter;        // Mip-Splatting style
} VkPhysicalDeviceGaussianSplatFeaturesVKGSPLAT;

// New entry point. Records a 3DGS dispatch into a command buffer.
typedef void (VKAPI_PTR *PFN_vkCmdDrawGaussianSplatsVKGSPLAT)(
    VkCommandBuffer                          commandBuffer,
    const VkGaussianSplatBufferInfoVKGSPLAT*  pBufferInfo,
    const VkGaussianSplatRenderInfoVKGSPLAT*  pRenderInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkCmdDrawGaussianSplatsVKGSPLAT(
    VkCommandBuffer                          commandBuffer,
    const VkGaussianSplatBufferInfoVKGSPLAT*  pBufferInfo,
    const VkGaussianSplatRenderInfoVKGSPLAT*  pRenderInfo);
#endif

#ifdef __cplusplus
} // extern "C"
#endif
