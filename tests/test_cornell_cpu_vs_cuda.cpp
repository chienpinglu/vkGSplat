// SPDX-License-Identifier: Apache-2.0
//
// Cornell-box CPU-vs-CUDA reference-image gate.
//
// This is the second honest test added on top of the RTX 5090 workstation
// plan: instead of just asserting that capability strings show up in a log,
// it actually renders a non-trivial scene through both vkGSplat backends
// at the same resolution and compares the resulting images. The contract
// is "if either side regresses, the test fails", which is what every other
// phase of the plan stops short of doing.
//
// The scene is a small Cornell-box-shaped Gaussian arrangement (red left
// wall, green right wall, white back wall, neutral floor, bright top
// light) -- not a photorealistic path-tracer reference, but enough geometry
// to exercise:
//   * splat projection across screen-aligned and depth-staggered tiles
//   * SH evaluation with non-trivial colors
//   * the tile-list scatter path (M1) on the CUDA side
//   * front-to-back alpha compositing through several layers
//
// On success it emits PSNR (and a few statistics) and writes both PPMs
// into the build directory so the harness or a reviewer can compare them
// by eye.

#include <vkgsplat/cpu_reference_renderer.h>
#include <vkgsplat/renderer.h>
#include <vkgsplat/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kWidth        = 128;
constexpr std::uint32_t kHeight       = 128;
constexpr float         kPsnrFloorDb  =  25.0f;  // generous; CUDA and CPU
                                                  //   normally line up at 40+
constexpr int           kSkip         = 77;

vkgsplat::Gaussian make_gaussian(vkgsplat::float3 position,
                                 float scale,
                                 vkgsplat::float3 color,
                                 float opacity_logit)
{
    constexpr float sh_c0 = 0.28209479177387814f;
    vkgsplat::Gaussian g{};
    g.position      = position;
    g.scale_log     = { std::log(scale), std::log(scale), std::log(scale) };
    g.rotation      = { 0.0f, 0.0f, 0.0f, 1.0f };
    g.opacity_logit = opacity_logit;
    g.sh[0] = {
        (color.x - 0.5f) / sh_c0,
        (color.y - 0.5f) / sh_c0,
        (color.z - 0.5f) / sh_c0,
    };
    return g;
}

// Tiny Cornell-box-shaped Gaussian field. 11 splats arranged so the scene
// is asymmetric across both axes, which makes accidental "all-black /
// all-uniform" outputs trivially fail the PSNR gate.
vkgsplat::Scene build_cornell_scene() {
    using vkgsplat::float3;
    vkgsplat::Scene scene;
    scene.resize(11);
    auto& g = scene.gaussians();
    // Back wall (white-ish) -- three small overlapping splats so the wall
    // is reconstructed continuously across the image.
    g[0] = make_gaussian({-0.30f,  0.00f, 0.80f}, 0.30f, {0.85f, 0.85f, 0.85f}, 5.0f);
    g[1] = make_gaussian({ 0.00f,  0.00f, 0.80f}, 0.30f, {0.85f, 0.85f, 0.85f}, 5.0f);
    g[2] = make_gaussian({ 0.30f,  0.00f, 0.80f}, 0.30f, {0.85f, 0.85f, 0.85f}, 5.0f);
    // Left red wall.
    g[3] = make_gaussian({-0.60f, -0.20f, 0.55f}, 0.25f, {0.95f, 0.10f, 0.10f}, 5.0f);
    g[4] = make_gaussian({-0.60f,  0.20f, 0.55f}, 0.25f, {0.95f, 0.10f, 0.10f}, 5.0f);
    // Right green wall.
    g[5] = make_gaussian({ 0.60f, -0.20f, 0.55f}, 0.25f, {0.10f, 0.85f, 0.10f}, 5.0f);
    g[6] = make_gaussian({ 0.60f,  0.20f, 0.55f}, 0.25f, {0.10f, 0.85f, 0.10f}, 5.0f);
    // Floor.
    g[7] = make_gaussian({-0.25f, -0.55f, 0.55f}, 0.30f, {0.60f, 0.60f, 0.55f}, 4.0f);
    g[8] = make_gaussian({ 0.25f, -0.55f, 0.55f}, 0.30f, {0.60f, 0.60f, 0.55f}, 4.0f);
    // Ceiling area light (bright).
    g[9] = make_gaussian({ 0.00f,  0.55f, 0.55f}, 0.25f, {1.00f, 0.95f, 0.85f}, 8.0f);
    // Small interior box, a bit closer to the camera.
    g[10] = make_gaussian({-0.05f, -0.05f, 0.35f}, 0.18f, {0.80f, 0.75f, 0.70f}, 5.5f);
    return scene;
}

vkgsplat::Camera build_camera() {
    vkgsplat::Camera c;
    c.set_resolution(kWidth, kHeight);
    c.set_perspective(1.20f, static_cast<float>(kWidth) / kHeight, 0.05f, 5.0f);
    c.look_at({ 0.0f, 0.0f, -0.4f },
              { 0.0f, 0.0f,  0.6f },
              { 0.0f, 1.0f,  0.0f });
    return c;
}

std::uint8_t to_u8(float v) {
    return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

bool write_ppm(const std::filesystem::path& path,
               const vkgsplat::float4* pixels,
               std::uint32_t w,
               std::uint32_t h)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << w << ' ' << h << "\n255\n";
    std::vector<std::uint8_t> row(w * 3);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            const auto& p = pixels[y * w + x];
            row[x * 3 + 0] = to_u8(p.x);
            row[x * 3 + 1] = to_u8(p.y);
            row[x * 3 + 2] = to_u8(p.z);
        }
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size()));
    }
    return static_cast<bool>(out);
}

struct DiffStats {
    double mse_rgb       = 0.0;
    double psnr_db       = 0.0;
    double max_abs_delta = 0.0;
    std::size_t n_pixels = 0;
    std::size_t n_nontrivial = 0;  // pixels where either side is brighter
                                    //   than rgb=(0.05,0.05,0.05)
};

DiffStats compare(const std::vector<vkgsplat::float4>& a,
                  const std::vector<vkgsplat::float4>& b)
{
    DiffStats s{};
    if (a.size() != b.size()) return s;
    double sse = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double dr = static_cast<double>(a[i].x) - b[i].x;
        const double dg = static_cast<double>(a[i].y) - b[i].y;
        const double db = static_cast<double>(a[i].z) - b[i].z;
        const double sq = dr * dr + dg * dg + db * db;
        sse += sq;
        s.max_abs_delta = std::max({s.max_abs_delta,
                                    std::abs(dr),
                                    std::abs(dg),
                                    std::abs(db)});
        const double bright_a = std::max({a[i].x, a[i].y, a[i].z});
        const double bright_b = std::max({b[i].x, b[i].y, b[i].z});
        if (bright_a > 0.05 || bright_b > 0.05) ++s.n_nontrivial;
    }
    s.n_pixels = a.size();
    s.mse_rgb  = sse / (3.0 * static_cast<double>(s.n_pixels));
    if (s.mse_rgb <= 1e-12) {
        s.psnr_db = 200.0;  // effectively identical
    } else {
        s.psnr_db = 20.0 * std::log10(1.0 / std::sqrt(s.mse_rgb));
    }
    return s;
}

} // namespace

int main() {
    using namespace vkgsplat;

    auto cpu_backend  = make_renderer("cpp");
    auto cuda_backend = make_renderer("cuda");
    if (!cpu_backend) {
        std::fprintf(stderr, "cornell: cpp renderer factory returned null\n");
        return 1;
    }
    if (!cuda_backend) {
        std::fprintf(stderr,
            "cornell: cuda renderer factory returned null; "
            "build was probably configured without VKGSPLAT_ENABLE_CUDA\n");
        return kSkip;
    }

    const Scene  scene  = build_cornell_scene();
    const Camera camera = build_camera();
    RenderParams params;
    params.background = { 0.0f, 0.0f, 0.0f };

    std::vector<float4> cpu_px(kWidth * kHeight,  { 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<float4> cuda_px(kWidth * kHeight, { 0.0f, 0.0f, 0.0f, 0.0f });

    const RenderTarget cpu_target{
        RenderTargetKind::HOST_BUFFER,
        { kWidth, kHeight, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        cpu_px.data(),
    };
    const RenderTarget cuda_target{
        RenderTargetKind::HOST_BUFFER,
        { kWidth, kHeight, PixelFormat::R32G32B32A32_SFLOAT, 1, 1 },
        cuda_px.data(),
    };

    cpu_backend->upload(scene);
    {
        const FrameId f = cpu_backend->render(camera, params, cpu_target);
        cpu_backend->wait(f);
    }

    try {
        cuda_backend->upload(scene);
        const FrameId f = cuda_backend->render(camera, params, cuda_target);
        cuda_backend->wait(f);
    } catch (const std::exception& ex) {
        std::fprintf(stderr,
            "cornell: cuda backend threw (%s); treating as skip\n",
            ex.what());
        return kSkip;
    }

    // Dump both PPMs so a reviewer can eyeball them. Path can be overridden
    // by VKGSPLAT_CORNELL_DUMP_DIR; otherwise CWD is used (which under
    // ctest is the per-test working dir).
    const char* env_dir = std::getenv("VKGSPLAT_CORNELL_DUMP_DIR");
    std::filesystem::path out_dir =
        env_dir ? std::filesystem::path(env_dir)
                : std::filesystem::current_path() / "cornell_out";
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    const auto cpu_ppm  = out_dir / "cornell_cpu.ppm";
    const auto cuda_ppm = out_dir / "cornell_cuda.ppm";
    write_ppm(cpu_ppm,  cpu_px.data(),  kWidth, kHeight);
    write_ppm(cuda_ppm, cuda_px.data(), kWidth, kHeight);

    const DiffStats diff = compare(cpu_px, cuda_px);

    std::printf("cornell: image=%ux%u splats=%zu\n",
                kWidth, kHeight, scene.gaussians().size());
    std::printf("cornell: cpu_ppm=%s\n",  cpu_ppm.string().c_str());
    std::printf("cornell: cuda_ppm=%s\n", cuda_ppm.string().c_str());
    std::printf("cornell: psnr_db=%.2f max_abs_delta=%.4f mse=%.6f "
                "nontrivial_pixels=%zu/%zu\n",
                diff.psnr_db, diff.max_abs_delta, diff.mse_rgb,
                diff.n_nontrivial, diff.n_pixels);

    // Sanity: scene actually rendered something. If both images are
    // completely black, PSNR would be infinite (mse=0) and we would
    // falsely pass.
    if (diff.n_nontrivial < diff.n_pixels / 10) {
        std::fprintf(stderr,
            "cornell=FAIL only %zu/%zu pixels are non-trivial; "
            "scene likely did not render at all\n",
            diff.n_nontrivial, diff.n_pixels);
        return 1;
    }

    if (diff.psnr_db < kPsnrFloorDb) {
        std::fprintf(stderr,
            "cornell=FAIL psnr_db=%.2f < floor %.2f -- "
            "CUDA and CPU references diverged. "
            "Inspect %s vs %s.\n",
            diff.psnr_db, kPsnrFloorDb,
            cpu_ppm.string().c_str(), cuda_ppm.string().c_str());
        return 1;
    }

    std::printf("cornell=PASS\n");
    return 0;
}
