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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
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
    std::vector<std::string> vertex_properties;
};

PlyHeader parse_ply_header(std::istream& in) {
    PlyHeader hdr;
    std::string line;
    if (!std::getline(in, line) || line != "ply") {
        throw std::runtime_error("not a PLY file");
    }
    bool in_vertex = false;
    while (std::getline(in, line)) {
        if (line.rfind("format ", 0) == 0) {
            hdr.binary_little_endian = (line.find("binary_little_endian") != std::string::npos);
        } else if (line.rfind("element vertex ", 0) == 0) {
            hdr.vertex_count = static_cast<std::size_t>(std::stoull(line.substr(15)));
            in_vertex = true;
        } else if (line.rfind("element ", 0) == 0) {
            in_vertex = false;
        } else if (in_vertex && line.rfind("property ", 0) == 0) {
            std::istringstream ls(line);
            std::string property;
            std::string type;
            std::string name;
            ls >> property >> type >> name;
            if (type == "list") {
                throw std::runtime_error("PLY: list properties in vertex elements are not supported");
            }
            if (type != "float" && type != "float32") {
                throw std::runtime_error("PLY: only float vertex properties are supported");
            }
            hdr.vertex_properties.push_back(name);
        } else if (line == "end_header") {
            hdr.data_offset = in.tellg();
            break;
        }
    }
    if (!hdr.binary_little_endian) {
        throw std::runtime_error("PLY: only binary_little_endian is supported in v1");
    }
    if (hdr.data_offset == std::streampos{}) {
        throw std::runtime_error("PLY: missing end_header");
    }
    return hdr;
}

int find_property(const std::vector<std::string>& props, const std::string& name) {
    const auto it = std::find(props.begin(), props.end(), name);
    if (it == props.end()) return -1;
    return static_cast<int>(std::distance(props.begin(), it));
}

float read_prop(const std::vector<float>& row, int index, const char* name) {
    if (index < 0) {
        throw std::runtime_error(std::string{"PLY: missing required property "} + name);
    }
    return row[static_cast<std::size_t>(index)];
}

} // namespace

Scene load_ply(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Scene::load_ply: cannot open " + path.string());
    }

    const PlyHeader hdr = parse_ply_header(in);
    in.seekg(hdr.data_offset);

    const int idx_x = find_property(hdr.vertex_properties, "x");
    const int idx_y = find_property(hdr.vertex_properties, "y");
    const int idx_z = find_property(hdr.vertex_properties, "z");
    const int idx_opacity = find_property(hdr.vertex_properties, "opacity");
    const int idx_scale0 = find_property(hdr.vertex_properties, "scale_0");
    const int idx_scale1 = find_property(hdr.vertex_properties, "scale_1");
    const int idx_scale2 = find_property(hdr.vertex_properties, "scale_2");
    const int idx_rot0 = find_property(hdr.vertex_properties, "rot_0");
    const int idx_rot1 = find_property(hdr.vertex_properties, "rot_1");
    const int idx_rot2 = find_property(hdr.vertex_properties, "rot_2");
    const int idx_rot3 = find_property(hdr.vertex_properties, "rot_3");
    const int idx_fdc0 = find_property(hdr.vertex_properties, "f_dc_0");
    const int idx_fdc1 = find_property(hdr.vertex_properties, "f_dc_1");
    const int idx_fdc2 = find_property(hdr.vertex_properties, "f_dc_2");

    Scene scene;
    scene.set_name(path.stem().string());
    scene.resize(hdr.vertex_count);

    std::vector<float> row(hdr.vertex_properties.size());
    for (std::size_t i = 0; i < hdr.vertex_count; ++i) {
        in.read(reinterpret_cast<char*>(row.data()),
                static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!in) {
            throw std::runtime_error("Scene::load_ply: failed while reading vertex data");
        }

        Gaussian g{};
        g.position = {
            read_prop(row, idx_x, "x"),
            read_prop(row, idx_y, "y"),
            read_prop(row, idx_z, "z"),
        };
        g.scale_log = {
            read_prop(row, idx_scale0, "scale_0"),
            read_prop(row, idx_scale1, "scale_1"),
            read_prop(row, idx_scale2, "scale_2"),
        };
        g.rotation = {
            read_prop(row, idx_rot0, "rot_0"),
            read_prop(row, idx_rot1, "rot_1"),
            read_prop(row, idx_rot2, "rot_2"),
            read_prop(row, idx_rot3, "rot_3"),
        };
        g.opacity_logit = read_prop(row, idx_opacity, "opacity");
        g.sh[0] = {
            read_prop(row, idx_fdc0, "f_dc_0"),
            read_prop(row, idx_fdc1, "f_dc_1"),
            read_prop(row, idx_fdc2, "f_dc_2"),
        };

        for (int sh = 1; sh < Gaussian::sh_coeffs; ++sh) {
            const int r = find_property(hdr.vertex_properties, "f_rest_" + std::to_string(sh - 1));
            const int g_idx = find_property(hdr.vertex_properties, "f_rest_" + std::to_string((sh - 1) + 15));
            const int b = find_property(hdr.vertex_properties, "f_rest_" + std::to_string((sh - 1) + 30));
            if (r >= 0 && g_idx >= 0 && b >= 0) {
                g.sh[static_cast<std::size_t>(sh)] = {
                    row[static_cast<std::size_t>(r)],
                    row[static_cast<std::size_t>(g_idx)],
                    row[static_cast<std::size_t>(b)],
                };
            }
        }

        scene.gaussians()[i] = g;
    }

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

    struct SplatRecord {
        float position[3];
        float scale[3];
        std::uint8_t color[4];
        std::uint8_t rotation[4];
    };
    static_assert(sizeof(SplatRecord) == splat_record_size);

    std::vector<SplatRecord> records(count);
    in.read(reinterpret_cast<char*>(records.data()), static_cast<std::streamsize>(byte_size));
    if (!in) {
        throw std::runtime_error("Scene::load_splat: failed to read " + path.string());
    }

    Scene scene;
    scene.set_name(path.stem().string());
    scene.resize(count);

    constexpr float sh_c0 = 0.28209479177387814f;
    constexpr float eps = 1e-6f;

    for (std::size_t i = 0; i < count; ++i) {
        const auto& r = records[i];
        Gaussian g{};
        g.position = { r.position[0], r.position[1], r.position[2] };

        for (float s : r.scale) {
            if (!(s > 0.0f)) {
                throw std::runtime_error("Scene::load_splat: scale values must be positive");
            }
        }
        g.scale_log = { std::log(r.scale[0]), std::log(r.scale[1]), std::log(r.scale[2]) };

        // antimatter15 .splat stores quaternion bytes as xyzw in [0,255].
        g.rotation = {
            (static_cast<float>(r.rotation[0]) - 128.0f) / 128.0f,
            (static_cast<float>(r.rotation[1]) - 128.0f) / 128.0f,
            (static_cast<float>(r.rotation[2]) - 128.0f) / 128.0f,
            (static_cast<float>(r.rotation[3]) - 128.0f) / 128.0f,
        };

        const float alpha = std::clamp(static_cast<float>(r.color[3]) / 255.0f, eps, 1.0f - eps);
        g.opacity_logit = std::log(alpha / (1.0f - alpha));

        g.sh[0] = {
            (static_cast<float>(r.color[0]) / 255.0f - 0.5f) / sh_c0,
            (static_cast<float>(r.color[1]) / 255.0f - 0.5f) / sh_c0,
            (static_cast<float>(r.color[2]) / 255.0f - 0.5f) / sh_c0,
        };

        scene.gaussians()[i] = g;
    }

    return scene;
}

} // namespace io
} // namespace vksplat
