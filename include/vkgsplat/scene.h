// SPDX-License-Identifier: Apache-2.0
//
// Scene: a host-side container for a 3D Gaussian splat scene.
//
// The v1 SDG flow assumes scenes are loaded from disk (.ply / .splat),
// optionally augmented with synthetic foreground objects, and then
// uploaded to the compute backend. Editing is out of scope for v1.
#pragma once

#include "types.h"

#include <filesystem>
#include <string>
#include <vector>

namespace vkgsplat {

// Backing storage for a loaded splat scene. Owns a contiguous Gaussian
// buffer; a CUDA backend can mirror this buffer to device memory once,
// then iterate camera positions across many SDG frames.
class Scene {
public:
    Scene() = default;

    // Load a scene from a .ply or .splat file. Throws std::runtime_error
    // on parse failure. Format is detected from the extension.
    static Scene load(const std::filesystem::path& path);

    // Capacity reserved for the Gaussian buffer; resize() is preferred
    // when the count is known up front to avoid reallocation.
    void reserve(std::size_t n);
    void resize(std::size_t n);

    [[nodiscard]] std::size_t size() const noexcept { return gaussians_.size(); }
    [[nodiscard]] bool empty() const noexcept { return gaussians_.empty(); }

    [[nodiscard]] const std::vector<Gaussian>& gaussians() const noexcept { return gaussians_; }
    [[nodiscard]] std::vector<Gaussian>&       gaussians()       noexcept { return gaussians_; }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

private:
    std::vector<Gaussian> gaussians_;
    std::string           name_;
};

// File-format helpers. Implementations live in src/core/scene.cpp.
namespace io {

Scene load_ply(const std::filesystem::path& path);
Scene load_splat(const std::filesystem::path& path);

} // namespace io
} // namespace vkgsplat
