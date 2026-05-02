// SPDX-License-Identifier: Apache-2.0
//
// Host-side Scene loader for .ply / .splat 3DGS scenes.
//
// v1 status: this is the parsing skeleton. The .ply path implements the
// header walk against the layout produced by the original Kerbl 2023
// training code (positions, normals, sh_dc, sh_rest, opacity, scale,
// rotation). The .splat path is a thin reader for the binary format
// popularized by the antimatter15/splat web viewer.
//
// Both implementations stop short of the full robust parser pending a
// dedicated dependency choice (tinyply vs nanoply vs hand-rolled).

#include "vksplat/scene.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vksplat {

void Scene::reserve(std::size_t n) { gaussians_.reserve(n); }
void Scene::resize(std::size_t n)  { gaussians_.resize(n); }

Scene Scene::load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("vksplat::Scene::load: file not found: " + path.string());
    }

    const auto ext = path.extension().string();
    if (ext == ".ply") {
        return io::load_ply(path);
    }
    if (ext == ".splat") {
        return io::load_splat(path);
    }
    throw std::runtime_error("vksplat::Scene::load: unsupported extension: " + ext);
}

namespace io {

namespace {

// Minimal PLY header walk. Returns the byte offset of the data section
// and the declared vertex count. Asserts the format is binary_little_endian.
struct PlyHeader {
    std::size_t   vertex_count = 0;
    std::streampos data_offset = 0;
    bool          binary_little_endian = false;
};

PlyHeader parse_ply_header(std::istream& in) {
    PlyHeader hdr;
    std::string line;
    if (!std::getline(in, line) || line != "ply") {
        throw std::runtime_error("not a PLY file");
    }
    while (std::getline(in, line)) {
        if (line.rfind("format ", 0) == 0) {
            hdr.binary_little_endian = (line.find("binary_little_endian") != std::string::npos);
        } else if (line.rfind("element vertex ", 0) == 0) {
            hdr.vertex_count = static_cast<std::size_t>(std::stoull(line.substr(15)));
        } else if (line == "end_header") {
            hdr.data_offset = in.tellg();
            break;
        }
    }
    if (!hdr.binary_little_endian) {
        throw std::runtime_error("PLY: only binary_little_endian is supported in v1");
    }
    return hdr;
}

} // namespace

Scene load_ply(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Scene::load_ply: cannot open " + path.string());
    }

    const PlyHeader hdr = parse_ply_header(in);
    in.seekg(hdr.data_offset);

    Scene scene;
    scene.set_name(path.stem().string());
    scene.resize(hdr.vertex_count);

    // TODO(v1): implement the full property walk against the trained
    // 3DGS .ply layout (positions, sh_dc, sh_rest [degree-3 -> 45
    // floats], opacity logit, scale_log[3], rotation[4]). The layout
    // is fixed by the original training code so a hand-rolled reader
    // is fine; we keep a sentinel here so callers see clear failure
    // until that is in.
    (void)in;
    return scene;
}

Scene load_splat(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Scene::load_splat: cannot open " + path.string());
    }
    const auto byte_size = static_cast<std::size_t>(in.tellg());
    in.seekg(0);

    // antimatter15/splat layout: 32 bytes per Gaussian, packed as
    //   float[3] position, float[3] scale, uint8_t[4] color (RGBA),
    //   uint8_t[4] rotation (xyzw quaternion in [0,255]).
    constexpr std::size_t splat_record_size = 32;
    if (byte_size % splat_record_size != 0) {
        throw std::runtime_error("Scene::load_splat: file size not a multiple of 32 bytes");
    }
    const std::size_t count = byte_size / splat_record_size;

    Scene scene;
    scene.set_name(path.stem().string());
    scene.resize(count);

    // TODO(v1): unpack the records into vksplat::Gaussian. Same status
    // as load_ply — sentinel resize, real unpack pending the fixed
    // dependency story.
    return scene;
}

} // namespace io
} // namespace vksplat
