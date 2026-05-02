// SPDX-License-Identifier: Apache-2.0
//
// vkSplat — umbrella header. Application code can `#include
// <vksplat/vksplat.h>` to pull in the public surface.
//
// The position paper (paper/main.tex) and architecture notes
// (docs/architecture.md) are the canonical references for the design.
#pragma once

#include "version.h"
#include "types.h"
#include "scene.h"
#include "camera.h"
#include "renderer.h"
#include "vulkan/instance.h"
#include "vulkan/device.h"
#include "vulkan/swapchain.h"
#include "cuda/rasterizer.h"
#include "interop/external_memory.h"
#include "interop/timeline_semaphore.h"
#include "extensions/vk_vksplat_gaussian_splatting.h"
