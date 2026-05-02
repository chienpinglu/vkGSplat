#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct HostTexture {
    std::vector<uint8_t> data;
    int w = 0, h = 0, bpp = 0;
};

struct HostModel {
    std::vector<float> verts;      // flattened per-face vertices, (nfaces*3) * 4 floats
    std::vector<float> norms;      // flattened per-face normals,  (nfaces*3) * 4 floats
    std::vector<float> tex;        // flattened per-face UVs,      (nfaces*3) * 2 floats
    std::vector<int>   indices;    // identity indices, nfaces*3
    int nfaces = 0;
    HostTexture diffuse;
    HostTexture normalmap;
    HostTexture specular;
};

HostModel load_model(const std::string &obj_path);
