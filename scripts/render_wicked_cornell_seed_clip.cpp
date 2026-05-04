// SPDX-License-Identifier: Apache-2.0
//
// Generates a visual capture-contract clip from Wicked Engine's Cornell-box
// OBJ asset. MoltenVK on this Mac does not expose Vulkan ray tracing, so this
// loads the same Wicked geometry and sends it through vkGSplat's CPU
// ray-tracing seed, reprojection, and CPU or Metal denoise path.

#include <vkgsplat/raytrace_seed.h>
#if defined(VKGSPLAT_ENABLE_METAL)
#include <vkgsplat/metal/denoise.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float pi = 3.14159265358979323846f;

enum class DenoiseBackend {
    Cpu,
    Metal,
};

struct ObjFace {
    std::vector<int> indices;
    std::string material;
    std::uint32_t primitive_id = 0;
};

struct ObjData {
    std::vector<vkgsplat::float3> vertices;
    std::vector<ObjFace> faces;
    std::map<std::string, vkgsplat::float4> materials;
};

vkgsplat::float4 color(float r, float g, float b) {
    return { r, g, b, 1.0f };
}

std::uint8_t to_byte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(value * 255.0f));
}

std::string trim(const std::string& line) {
    const auto begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = line.find_last_not_of(" \t\r\n");
    return line.substr(begin, end - begin + 1);
}

vkgsplat::float4 polished_material_color(const std::string& name, vkgsplat::float4 kd) {
    if (name == "white") {
        return color(0.78f, 0.78f, 0.72f);
    }
    if (name == "red") {
        return color(0.92f, 0.16f, 0.10f);
    }
    if (name == "green") {
        return color(0.12f, 0.76f, 0.22f);
    }
    return color(kd.x * 0.85f, kd.y * 0.85f, kd.z * 0.85f);
}

std::map<std::string, vkgsplat::float4> load_mtl(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open material file: " + path.string());
    }

    std::map<std::string, vkgsplat::float4> materials;
    std::string current;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "newmtl") {
            ss >> current;
        } else if (tag == "Kd" && !current.empty()) {
            float r = 1.0f;
            float g = 1.0f;
            float b = 1.0f;
            ss >> r >> g >> b;
            materials[current] = polished_material_color(current, color(r, g, b));
        }
    }

    return materials;
}

int parse_face_vertex_index(const std::string& token, int vertex_count) {
    const std::size_t slash = token.find('/');
    const std::string number = token.substr(0, slash);
    int index = std::stoi(number);
    if (index < 0) {
        index = vertex_count + index + 1;
    }
    if (index <= 0 || index > vertex_count) {
        throw std::runtime_error("OBJ face index out of bounds");
    }
    return index - 1;
}

ObjData load_obj(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open OBJ file: " + path.string());
    }

    ObjData obj;
    std::string current_material = "white";
    std::string line;
    std::uint32_t primitive_id = 100;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "mtllib") {
            std::string mtl_name;
            ss >> mtl_name;
            obj.materials = load_mtl(path.parent_path() / mtl_name);
        } else if (tag == "usemtl") {
            ss >> current_material;
        } else if (tag == "v") {
            vkgsplat::float3 vertex{};
            ss >> vertex.x >> vertex.y >> vertex.z;
            obj.vertices.push_back(vertex);
        } else if (tag == "f") {
            ObjFace face;
            face.material = current_material;
            face.primitive_id = primitive_id++;

            std::string token;
            while (ss >> token) {
                face.indices.push_back(parse_face_vertex_index(
                    token,
                    static_cast<int>(obj.vertices.size())));
            }

            if (face.indices.size() >= 3) {
                obj.faces.push_back(face);
            }
        }
    }

    if (obj.vertices.empty() || obj.faces.empty()) {
        throw std::runtime_error("OBJ did not contain renderable geometry");
    }
    if (obj.materials.empty()) {
        obj.materials["white"] = color(0.78f, 0.78f, 0.72f);
    }

    return obj;
}

vkgsplat::RayTracingScene make_scene_from_obj(const ObjData& obj) {
    vkgsplat::float3 min_v{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    vkgsplat::float3 max_v{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };
    for (const auto& v : obj.vertices) {
        min_v.x = std::min(min_v.x, v.x);
        min_v.y = std::min(min_v.y, v.y);
        min_v.z = std::min(min_v.z, v.z);
        max_v.x = std::max(max_v.x, v.x);
        max_v.y = std::max(max_v.y, v.y);
        max_v.z = std::max(max_v.z, v.z);
    }

    const vkgsplat::float3 center{
        (min_v.x + max_v.x) * 0.5f,
        (min_v.y + max_v.y) * 0.5f,
        (min_v.z + max_v.z) * 0.5f,
    };
    const float extent_x = max_v.x - min_v.x;
    const float extent_y = max_v.y - min_v.y;
    const float extent_z = max_v.z - min_v.z;
    const float scale = 2.35f / std::max({ extent_x, extent_y, extent_z });

    auto transform = [&](vkgsplat::float3 v) {
        return vkgsplat::float3{
            (v.x - center.x) * scale,
            (v.y - center.y) * scale,
            -(v.z - center.z) * scale + 3.25f,
        };
    };

    std::map<std::string, std::uint32_t> material_indices;
    vkgsplat::RayTracingScene scene;
    auto material_index = [&](const std::string& name) {
        const auto existing = material_indices.find(name);
        if (existing != material_indices.end()) {
            return existing->second;
        }

        const auto material = obj.materials.find(name);
        const vkgsplat::float4 base_color =
            material != obj.materials.end() ? material->second : color(0.78f, 0.78f, 0.72f);
        const std::uint32_t index = static_cast<std::uint32_t>(scene.materials.size());
        scene.materials.push_back({ base_color });
        material_indices[name] = index;
        return index;
    };

    for (const ObjFace& face : obj.faces) {
        const std::uint32_t material = material_index(face.material);
        for (std::size_t i = 1; i + 1 < face.indices.size(); ++i) {
            scene.triangles.push_back({
                transform(obj.vertices[static_cast<std::size_t>(face.indices[0])]),
                transform(obj.vertices[static_cast<std::size_t>(face.indices[i])]),
                transform(obj.vertices[static_cast<std::size_t>(face.indices[i + 1])]),
                material,
                face.primitive_id,
            });
        }
    }

    return scene;
}

std::vector<vkgsplat::float4> compose_panel(const vkgsplat::RayTracingSeedFrame& seed,
                                           const vkgsplat::SvgfDenoiseResult& denoised) {
    constexpr std::uint32_t divider = 4;
    const std::uint32_t panel_width = seed.width * 2 + divider;
    const std::uint32_t panel_height = seed.height;
    std::vector<vkgsplat::float4> panel(
        static_cast<std::size_t>(panel_width) * panel_height,
        color(0.02f, 0.02f, 0.025f));

    for (std::uint32_t y = 0; y < seed.height; ++y) {
        for (std::uint32_t x = 0; x < seed.width; ++x) {
            const std::size_t src = static_cast<std::size_t>(y) * seed.width + x;
            const std::size_t noisy_dst = static_cast<std::size_t>(y) * panel_width + x;
            const std::size_t denoise_dst =
                static_cast<std::size_t>(y) * panel_width + seed.width + divider + x;
            panel[noisy_dst] = seed.color[src];
            panel[denoise_dst] = denoised.color[src];
        }
        for (std::uint32_t x = seed.width; x < seed.width + divider; ++x) {
            panel[static_cast<std::size_t>(y) * panel_width + x] = color(0.88f, 0.88f, 0.88f);
        }
    }

    return panel;
}

void write_ppm(const std::filesystem::path& path,
               const std::vector<vkgsplat::float4>& pixels,
               std::uint32_t width,
               std::uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output image: " + path.string());
    }

    out << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& p : pixels) {
        const unsigned char rgb[] = { to_byte(p.x), to_byte(p.y), to_byte(p.z) };
        out.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
    }
}

std::filesystem::path frame_path(const std::filesystem::path& prefix, int frame) {
    std::ostringstream name;
    name << prefix.string() << "_" << std::setw(3) << std::setfill('0') << frame << ".ppm";
    return name.str();
}

DenoiseBackend parse_backend(const std::string& backend) {
    if (backend == "cpu") {
        return DenoiseBackend::Cpu;
    }
    if (backend == "metal") {
        return DenoiseBackend::Metal;
    }
    throw std::runtime_error("backend must be 'cpu' or 'metal'");
}

vkgsplat::SvgfDenoiseResult run_denoise_backend(
    DenoiseBackend backend,
    const vkgsplat::RayTracingSeedFrame& seed,
    const vkgsplat::ReprojectionResult& reprojected,
    const vkgsplat::SvgfDenoiseOptions& options) {
    if (backend == DenoiseBackend::Cpu) {
        return vkgsplat::denoise_svgf_baseline(
            vkgsplat::as_denoise_frame(seed),
            reprojected,
            options);
    }

#if defined(VKGSPLAT_ENABLE_METAL)
    if (!vkgsplat::metal::is_available()) {
        throw std::runtime_error("Metal backend was requested, but no Metal device is available");
    }
    return vkgsplat::metal::denoise_svgf_baseline(
        vkgsplat::as_denoise_frame(seed),
        reprojected,
        options);
#else
    throw std::runtime_error("Metal backend was requested, but this binary was built without VKGSPLAT_ENABLE_METAL");
#endif
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path prefix =
        argc > 1 ? std::filesystem::path(argv[1])
                 : std::filesystem::path("docs/images/wicked_cornell_seed_clip");
    const std::filesystem::path obj_path =
        argc > 2 ? std::filesystem::path(argv[2])
                 : std::filesystem::path("third_party/WickedEngine/Content/models/cornellbox.obj");
    const DenoiseBackend backend = argc > 3 ? parse_backend(argv[3]) : DenoiseBackend::Cpu;
    const std::filesystem::path parent = prefix.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    const ObjData obj = load_obj(obj_path);
    const vkgsplat::RayTracingScene scene = make_scene_from_obj(obj);

    vkgsplat::RayTracingSeedFrame previous_seed;
    bool has_previous = false;

    constexpr int frame_count = 36;
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        const float t = static_cast<float>(frame_index) / static_cast<float>(frame_count);
        const float angle = -0.72f + 1.44f * t;
        const float height = -0.04f + 0.12f * std::sin((t * 2.0f - 0.5f) * pi);
        const float radius = 1.18f;

        vkgsplat::RayTracingCamera camera;
        camera.eye = { std::sin(angle) * radius, height, 0.12f + (1.0f - std::cos(angle)) * 0.22f };
        camera.target = { -std::sin(angle) * 0.30f, -0.06f, 3.25f };
        camera.up = { 0.0f, 1.0f, 0.0f };
        camera.fov_y_radians = 0.92f;
        camera.z_near = 0.05f;
        camera.z_far = 16.0f;

        vkgsplat::RayTracingDispatch dispatch;
        dispatch.width = 256;
        dispatch.height = 144;
        dispatch.samples_per_pixel = 1;
        dispatch.seed = 19001u + static_cast<std::uint32_t>(frame_index * 131);
        dispatch.radiance_noise = 0.42f;

        const vkgsplat::RayTracingSeedFrame seed =
            vkgsplat::trace_raytracing_seed(scene, camera, dispatch);

        vkgsplat::ReprojectionResult reprojected;
        if (has_previous) {
            const vkgsplat::CameraMotionMap motion = vkgsplat::compute_camera_motion_map(
                seed.width,
                seed.height,
                seed.ndc_depth,
                seed.inverse_view_projection,
                previous_seed.view_projection);

            vkgsplat::ReprojectionOptions reprojection_options;
            reprojection_options.history_weight = 0.82f;
            reprojection_options.depth_threshold = 0.22f;
            reprojected = vkgsplat::reproject_history(
                vkgsplat::as_reprojection_frame(previous_seed),
                vkgsplat::as_reprojection_frame(seed),
                motion.current_to_previous_px,
                reprojection_options);
        } else {
            const std::size_t count = static_cast<std::size_t>(seed.width) * seed.height;
            reprojected.width = seed.width;
            reprojected.height = seed.height;
            reprojected.color = seed.color;
            reprojected.valid_history.assign(count, 0);
        }

        vkgsplat::SvgfDenoiseOptions denoise_options;
        denoise_options.history_weight = 0.48f;
        denoise_options.depth_threshold = 0.18f;
        denoise_options.spatial_radius = 1;

        const vkgsplat::SvgfDenoiseResult denoised =
            run_denoise_backend(backend, seed, reprojected, denoise_options);

        const auto panel = compose_panel(seed, denoised);
        write_ppm(frame_path(prefix, frame_index),
                  panel,
                  seed.width * 2 + 4,
                  seed.height);

        previous_seed = seed;
        has_previous = true;
    }

    return 0;
}
