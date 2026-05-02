#include "host_loader.h"
#include "model.h"
#include <cstring>
#include <iostream>

static HostTexture extract_texture(const TGAImage &img) {
    HostTexture ht;
    ht.w = img.width();
    ht.h = img.height();
    if (ht.w == 0 || ht.h == 0) return ht;

    TGAColor probe = img.get(0, 0);
    ht.bpp = probe.bytespp;

    size_t nbytes = static_cast<size_t>(ht.w) * ht.h * ht.bpp;
    ht.data.resize(nbytes);
    for (int y = 0; y < ht.h; y++) {
        for (int x = 0; x < ht.w; x++) {
            TGAColor c = img.get(x, y);
            memcpy(ht.data.data() + (x + y * ht.w) * ht.bpp, c.bgra, ht.bpp);
        }
    }
    return ht;
}

HostModel load_model(const std::string &obj_path) {
    Model m(obj_path);
    HostModel hm;
    hm.nfaces = m.nfaces();
    int total = hm.nfaces * 3;

    hm.verts.resize(total * 4);
    hm.norms.resize(total * 4);
    hm.tex.resize(total * 2);
    hm.indices.resize(total);

    for (int f = 0; f < hm.nfaces; f++) {
        for (int v = 0; v < 3; v++) {
            int idx = f * 3 + v;
            hm.indices[idx] = idx;

            vec4 vert = m.vert(f, v);
            hm.verts[idx*4+0] = static_cast<float>(vert.x);
            hm.verts[idx*4+1] = static_cast<float>(vert.y);
            hm.verts[idx*4+2] = static_cast<float>(vert.z);
            hm.verts[idx*4+3] = static_cast<float>(vert.w);

            vec4 n = m.normal(f, v);
            hm.norms[idx*4+0] = static_cast<float>(n.x);
            hm.norms[idx*4+1] = static_cast<float>(n.y);
            hm.norms[idx*4+2] = static_cast<float>(n.z);
            hm.norms[idx*4+3] = static_cast<float>(n.w);

            vec2 uv = m.uv(f, v);
            hm.tex[idx*2+0] = static_cast<float>(uv.x);
            hm.tex[idx*2+1] = static_cast<float>(uv.y);
        }
    }

    hm.diffuse  = extract_texture(m.diffuse());
    hm.specular = extract_texture(m.specular());

    // Normal map isn't exposed by Model's public API, load it manually
    {
        size_t dot = obj_path.find_last_of(".");
        if (dot != std::string::npos) {
            std::string nm_file = obj_path.substr(0, dot) + "_nm_tangent.tga";
            TGAImage nm_img;
            if (nm_img.read_tga_file(nm_file.c_str()))
                hm.normalmap = extract_texture(nm_img);
        }
    }

    return hm;
}
