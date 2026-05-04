// SPDX-License-Identifier: Apache-2.0
//
// vkSplat — umbrella header. Application code can `#include
// <vksplat/vksplat.h>` to pull in the public surface.
//
// The position paper (paper/vkSplat.tex) and architecture notes
// (docs/architecture.md) are the canonical references for the design.
#pragma once

#include "version.h"
#include "types.h"
#include "scene.h"
#include "camera.h"
#include "renderer.h"
#include "denoise.h"
#include "raytrace_seed.h"
#include "reprojection.h"
#include "tile_raster.h"
#include "spirv/module.h"
#include "spirv/translator.h"

#if defined(VKSPLAT_ENABLE_3DGS)
#  include "cpu_reference_renderer.h"
#  include "gpu_pipeline.h"
#endif

#if defined(VKSPLAT_ENABLE_VULKAN)
#  include "vulkan/instance.h"
#  include "vulkan/device.h"
#  include "vulkan/swapchain.h"
#  if defined(VKSPLAT_ENABLE_3DGS)
#    include "vulkan/mesh_shader_3dgs.h"
#    include "extensions/vk_vksplat_gaussian_splatting.h"
#  endif
#endif

#if defined(VKSPLAT_ENABLE_CUDA) && defined(VKSPLAT_ENABLE_3DGS)
#  include "cuda/rasterizer.h"
#  include "interop/external_memory.h"
#  include "interop/timeline_semaphore.h"
#endif

#if defined(VKSPLAT_ENABLE_METAL)
#  include "metal/denoise.h"
#endif
