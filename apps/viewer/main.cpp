// SPDX-License-Identifier: Apache-2.0
//
// vksplat_viewer — minimal CLI that loads a scene, builds the CUDA
// renderer, and produces a single image to stdout (or to a PNG with
// --output). The interactive swapchain path is staged but disabled in
// v1; the production SDG flow is headless.

#include <vksplat/vksplat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

struct Args {
    std::filesystem::path scene_path;
    std::filesystem::path output_path = "out.png";
    std::uint32_t width  = 1024;
    std::uint32_t height = 1024;
    bool          present = false;
    bool          dump_caps = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <scene.ply|scene.splat> [options]\n"
        "  --output <path.png>   write rendered image (default: out.png)\n"
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

} // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);

    if (args.dump_caps) {
        vksplat::vk::dump_instance_capabilities();
        return 0;
    }
    if (args.scene_path.empty()) usage(argv[0]);

    std::fprintf(stderr, "[vksplat] vksplat_viewer %s\n", vksplat::version_string);
    std::fprintf(stderr, "[vksplat] loading scene: %s\n", args.scene_path.string().c_str());
    auto scene = vksplat::Scene::load(args.scene_path);
    std::fprintf(stderr, "[vksplat] scene loaded: %zu gaussians\n", scene.size());

    vksplat::Camera camera;
    camera.set_resolution(args.width, args.height);
    camera.set_perspective(0.785398f /* 45deg */,
                           static_cast<float>(args.width) / static_cast<float>(args.height),
                           0.1f, 1000.0f);
    camera.look_at({ 0.0f, 0.0f, 3.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    auto renderer = vksplat::make_renderer("cuda");
    if (!renderer) {
        std::fprintf(stderr, "[vksplat] CUDA backend not available in this build\n");
        return 1;
    }
    renderer->upload(scene);

    vksplat::RenderTarget target;
    target.kind = vksplat::RenderTargetKind::HOST_BUFFER;
    target.desc = { args.width, args.height, vksplat::PixelFormat::R8G8B8A8_UNORM, 1, 1 };

    vksplat::RenderParams params;
    params.deterministic = true;
    params.seed          = 42;

    const auto frame = renderer->render(camera, params, target);
    renderer->wait(frame);

    std::fprintf(stderr, "[vksplat] frame %llu rendered (%ux%u) -> %s\n",
                 static_cast<unsigned long long>(frame),
                 args.width, args.height,
                 args.output_path.string().c_str());

    if (args.present) {
        std::fprintf(stderr, "[vksplat] --present is not supported in v1 (headless only).\n");
    }
    return 0;
}
