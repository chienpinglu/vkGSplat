// SPDX-License-Identifier: Apache-2.0
// vkGSplat: A compute-first Vulkan path for robotics SDG.
// See papers/vkGSplat.tex (Lu, 2026) for the position argument.
#pragma once

#define VKGSPLAT_VERSION_MAJOR 0
#define VKGSPLAT_VERSION_MINOR 0
#define VKGSPLAT_VERSION_PATCH 1

#define VKGSPLAT_VERSION_STRING "0.0.1"

namespace vkgsplat {

constexpr unsigned version_major = VKGSPLAT_VERSION_MAJOR;
constexpr unsigned version_minor = VKGSPLAT_VERSION_MINOR;
constexpr unsigned version_patch = VKGSPLAT_VERSION_PATCH;

constexpr const char* version_string = VKGSPLAT_VERSION_STRING;

} // namespace vkgsplat
