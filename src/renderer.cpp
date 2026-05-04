// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/renderer.h"

#if defined(VKGSPLAT_ENABLE_3DGS)
#  include "vkgsplat/cpu_reference_renderer.h"
#endif

#if defined(VKGSPLAT_ENABLE_3DGS)
#include <algorithm>
#include <cstdint>
#include <cstring>
#endif
#include <stdexcept>

namespace vkgsplat {

namespace {

#if defined(VKGSPLAT_ENABLE_3DGS)
std::uint8_t to_unorm8(float v) {
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(clamped * 255.0f + 0.5f);
}

class CppRenderer final : public Renderer {
public:
    [[nodiscard]] std::string_view backend_name() const override { return "cpp"; }

    void upload(const Scene& scene) override {
        scene_ = scene;
    }

    FrameId render(const Camera& camera,
                   const RenderParams& params,
                   const RenderTarget& target) override {
        if (target.kind != RenderTargetKind::HOST_BUFFER) {
            throw std::runtime_error("cpp renderer only supports HOST_BUFFER targets");
        }
        if (target.user_handle == nullptr) {
            throw std::runtime_error("cpp renderer requires RenderTarget::user_handle");
        }

        const ImageDesc render_desc{
            target.desc.width,
            target.desc.height,
            PixelFormat::R32G32B32A32_SFLOAT,
            target.desc.mip_levels,
            target.desc.array_layers,
        };
        const auto result = render_3dgs_cpu_reference(scene_, camera, params, render_desc);

        switch (target.desc.format) {
        case PixelFormat::R8G8B8A8_UNORM:
        case PixelFormat::R8G8B8A8_SRGB: {
            auto* out = static_cast<std::uint8_t*>(target.user_handle);
            for (std::size_t i = 0; i < result.pixels.size(); ++i) {
                out[i * 4 + 0] = to_unorm8(result.pixels[i].x);
                out[i * 4 + 1] = to_unorm8(result.pixels[i].y);
                out[i * 4 + 2] = to_unorm8(result.pixels[i].z);
                out[i * 4 + 3] = to_unorm8(result.pixels[i].w);
            }
            break;
        }
        case PixelFormat::R32G32B32A32_SFLOAT: {
            auto* out = static_cast<float4*>(target.user_handle);
            std::memcpy(out,
                        result.pixels.data(),
                        result.pixels.size() * sizeof(float4));
            break;
        }
        default:
            throw std::runtime_error("cpp renderer target format is not supported");
        }

        return ++last_frame_;
    }

    void wait(FrameId /*frame*/) override {}

private:
    Scene scene_;
    FrameId last_frame_ = 0;
};
#endif

} // namespace

#if defined(VKGSPLAT_ENABLE_CUDA) && defined(VKGSPLAT_ENABLE_3DGS)
std::unique_ptr<Renderer> make_cuda_renderer();
#endif

std::unique_ptr<Renderer> make_renderer(std::string_view backend_name) {
#if defined(VKGSPLAT_ENABLE_3DGS)
    if (backend_name.empty() || backend_name == "cpp" || backend_name == "cpu" ||
        backend_name == "reference") {
        return std::make_unique<CppRenderer>();
    }
#endif
#if defined(VKGSPLAT_ENABLE_CUDA) && defined(VKGSPLAT_ENABLE_3DGS)
    if (backend_name == "cuda") {
        return make_cuda_renderer();
    }
#endif
    return nullptr;
}

} // namespace vkgsplat
