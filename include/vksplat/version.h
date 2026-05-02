// SPDX-License-Identifier: Apache-2.0
// vkSplat: A compute-first Vulkan path for robotics SDG.
// See paper/main.tex (Lu, 2026) for the position argument.
#pragma once

#define VKSPLAT_VERSION_MAJOR 0
#define VKSPLAT_VERSION_MINOR 0
#define VKSPLAT_VERSION_PATCH 1

#define VKSPLAT_VERSION_STRING "0.0.1"

namespace vksplat {

constexpr unsigned version_major = VKSPLAT_VERSION_MAJOR;
constexpr unsigned version_minor = VKSPLAT_VERSION_MINOR;
constexpr unsigned version_patch = VKSPLAT_VERSION_PATCH;

constexpr const char* version_string = VKSPLAT_VERSION_STRING;

} // namespace vksplat
