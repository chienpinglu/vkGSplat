// SPDX-License-Identifier: Apache-2.0
//
// vkgsplat_viewer — minimal CLI that loads a scene, builds a renderer,
// and produces a single headless image. The interactive swapchain path
// is staged but disabled in v1; the production SDG flow is headless.

#include <vkgsplat/vkgsplat.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path scene_path;
    std::filesystem::path output_path = "out.ppm";
    std::string           backend = "cpp";
    std::uint32_t width  = 1024;
    std::uint32_t height = 1024;
    bool          present = false;
    bool          dump_caps = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <scene.ply|scene.splat> [options]\n"
        "  --backend <name>      renderer backend: cpp/cpu/reference/cuda (default: cpp)\n"
        "  --output <path.ppm>   write rendered image (default: out.ppm)\n"
        "  --size W H            output resolution (default: 1024 1024)\n"
        "  --present             open a debug swapchain (not supported in v1)\n"
        "  --dump-caps           dump Vulkan instance capabilities and exit\n",
        argv0);
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--help" || s == "-h") usage(argv[0]);
        else if (s == "--backend" && i + 1 < argc) a.backend = argv[++i];
        else if (s == "--output" && i + 1 < argc) a.output_path = argv[++i];
        else if (s == "--size" && i + 2 < argc) {
            a.width  = static_cast<std::uint32_t>(std::atoi(argv[++i]));
            a.height = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        }
        else if (s == "--present")   a.present = true;
        else if (s == "--dump-caps") a.dump_caps = true;
        else if (s.size() && s[0] != '-' && a.scene_path.empty()) a.scene_path = s;
        else usage(argv[0]);
    }
    return a;
}

bool write_ppm(const std::filesystem::path& path,
               std::uint32_t width,
               std::uint32_t height,
               const std::vector<std::uint8_t>& rgba) {
    FILE* f = std::fopen(path.string().c_str(), "wb");
    if (!f) {
        return false;
    }
    std::fprintf(f, "P6\n%u %u\n255\n", width, height);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4;
            const std::uint8_t rgb[3] = { rgba[i + 0], rgba[i + 1], rgba[i + 2] };
            if (std::fwrite(rgb, sizeof(rgb), 1, f) != 1) {
                std::fclose(f);
                return false;
            }
        }
    }
    return std::fclose(f) == 0;
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);

    if (args.dump_caps) {
#if defined(VKGSPLAT_ENABLE_VULKAN)
        vkgsplat::vk::dump_instance_capabilities();
        return 0;
#else
        std::fprintf(stderr, "[vkgsplat] Vulkan support is not available in this build\n");
        return 1;
#endif
    }
    if (args.scene_path.empty()) usage(argv[0]);

    std::fprintf(stderr, "[vkgsplat] vkgsplat_viewer %s\n", vkgsplat::version_string);
    std::fprintf(stderr, "[vkgsplat] loading scene: %s\n", args.scene_path.string().c_str());
    auto scene = vkgsplat::Scene::load(args.scene_path);
    std::fprintf(stderr, "[vkgsplat] scene loaded: %zu gaussians\n", scene.size());

    vkgsplat::Camera camera;
    camera.set_resolution(args.width, args.height);
    camera.set_perspective(0.785398f /* 45deg */,
                           static_cast<float>(args.width) / static_cast<float>(args.height),
                           0.1f, 1000.0f);
    camera.look_at({ 0.0f, 0.0f, 3.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    auto renderer = vkgsplat::make_renderer(args.backend);
    if (!renderer) {
        std::fprintf(stderr, "[vkgsplat] backend '%s' is not available in this build\n",
                     args.backend.c_str());
        return 1;
    }
    std::fprintf(stderr, "[vkgsplat] backend: %.*s\n",
                 static_cast<int>(renderer->backend_name().size()),
                 renderer->backend_name().data());
    renderer->upload(scene);

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(args.width) * args.height * 4);

    vkgsplat::RenderTarget target;
    target.kind = vkgsplat::RenderTargetKind::HOST_BUFFER;
    target.desc = { args.width, args.height, vkgsplat::PixelFormat::R8G8B8A8_UNORM, 1, 1 };
    target.user_handle = pixels.data();

    vkgsplat::RenderParams params;
    params.deterministic = true;
    params.seed          = 42;

    const auto frame = renderer->render(camera, params, target);
    renderer->wait(frame);

    if (!write_ppm(args.output_path, args.width, args.height, pixels)) {
        std::fprintf(stderr, "[vkgsplat] failed to write output image: %s\n",
                     args.output_path.string().c_str());
        return 1;
    }

    std::fprintf(stderr, "[vkgsplat] frame %llu rendered (%ux%u) -> %s\n",
                 static_cast<unsigned long long>(frame),
                 args.width, args.height,
                 args.output_path.string().c_str());

    if (args.present) {
        std::fprintf(stderr, "[vkgsplat] --present is not supported in v1 (headless only).\n");
    }
    return 0;
}
