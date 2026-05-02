// SPDX-License-Identifier: Apache-2.0
//
// CUDA backend for vkSplat: 3D Gaussian Splatting forward rasterizer.
//
// Pipeline (Kerbl et al. 2023, with SDG-flavoured tweaks):
//
//   preprocess_kernel   per-Gaussian: cull, project mean, build 2D conic,
//                       compute per-tile bin counts.
//   sort_pairs          radix-sort (key,value) = (depth|tile_id, gaussian_id).
//                       v1 uses CUB device-wide radix sort.
//   identify_ranges     scan to find the (start,end) of each tile in the
//                       sorted list.
//   blend_kernel        per-tile, per-pixel front-to-back alpha blend.
//
// The implementation here is the host-side scaffold (memory layout,
// stream + event plumbing, kernel launch boilerplate). The actual
// device code is staged behind clearly marked TODO blocks; v1 will
// fill them in by porting the reference CUDA implementation.

#include "vksplat/cuda/rasterizer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vksplat::cuda {

namespace {

#define VKSPLAT_CUDA_CHECK(expr)                                       \
    do {                                                               \
        cudaError_t err__ = (expr);                                    \
        if (err__ != cudaSuccess) {                                    \
            throw std::runtime_error(std::string{"CUDA error in "} +   \
                #expr + ": " + cudaGetErrorString(err__));             \
        }                                                              \
    } while (0)

// Device-side per-Gaussian preprocessed state. Two-pass: fill, then
// the binning pass derives tile spans. Layout-of-arrays would cut
// memory traffic; the AoS form here is for clarity.
struct PreprocessedGaussian {
    float2 mean_screen;     // pixel-space mean
    float3 conic;           // 2D inverse covariance (a, b, c)
    float  alpha;
    float  depth;
    float3 color;           // SH-evaluated RGB at this view
    int    tile_min_x, tile_min_y;
    int    tile_max_x, tile_max_y;
};

} // namespace

// ---------------------------------------------------------------------
// CUDA kernels (forward declarations / launch glue)
// ---------------------------------------------------------------------

__global__ void preprocess_kernel(int /*num_gaussians*/,
                                  const Gaussian* __restrict__ /*gaussians*/,
                                  PreprocessedGaussian* __restrict__ /*out*/,
                                  /* view, projection, image extent */
                                  int /*image_w*/, int /*image_h*/,
                                  int /*tile_size*/)
{
    // TODO(v1): port preprocess_cuda from reference 3DGS. Steps:
    //   1. Project mean to clip then to pixel space; cull on z<near.
    //   2. Compute 3D covariance from scale_log + rotation; project to
    //      2D screen-space conic.
    //   3. Evaluate SH at view direction -> RGB.
    //   4. Compute touched tile range (clamped to image extent).
}

__global__ void duplicate_with_keys_kernel(int /*num_gaussians*/,
                                           const PreprocessedGaussian* /*pre*/,
                                           // outputs:
                                           std::uint64_t* /*keys*/,
                                           int*           /*values*/)
{
    // TODO(v1): emit one (key,value) pair per (gaussian, touched-tile)
    // where key = (tile_id << 32) | depth_bits, value = gaussian_id.
}

__global__ void identify_tile_ranges_kernel(int /*num_pairs*/,
                                            const std::uint64_t* /*sorted_keys*/,
                                            int2* /*tile_ranges*/)
{
    // TODO(v1): scan sorted keys, write (start,end) for each tile.
}

__global__ void blend_kernel(int /*image_w*/, int /*image_h*/,
                             int /*tile_size*/,
                             const int2* /*tile_ranges*/,
                             const int*  /*sorted_gaussian_ids*/,
                             const PreprocessedGaussian* /*pre*/,
                             /* output */
                             float4* /*out_image*/,
                             float3 /*background*/)
{
    // TODO(v1): per-tile cooperative load into shared memory, then
    // per-pixel front-to-back alpha blend until alpha threshold.
}

// ---------------------------------------------------------------------
// RasterizerImpl — opaque device state for the public Rasterizer class
// ---------------------------------------------------------------------

class RasterizerImpl {
public:
    RasterizerImpl() {
        VKSPLAT_CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~RasterizerImpl() {
        if (gaussians_dev_) cudaFree(gaussians_dev_);
        if (preprocessed_dev_) cudaFree(preprocessed_dev_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void bind_to_uuid(const std::array<unsigned char, 16>& uuid) {
        int count = 0;
        VKSPLAT_CUDA_CHECK(cudaGetDeviceCount(&count));
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp p{};
            VKSPLAT_CUDA_CHECK(cudaGetDeviceProperties(&p, i));
            if (std::memcmp(p.uuid.bytes, uuid.data(), 16) == 0) {
                VKSPLAT_CUDA_CHECK(cudaSetDevice(i));
                std::fprintf(stderr, "[vksplat][cuda] bound to device %d (%s)\n", i, p.name);
                return;
            }
        }
        throw std::runtime_error("cuda::Rasterizer: no CUDA device matches Vulkan UUID");
    }

    void upload(const Scene& scene) {
        if (gaussians_dev_) {
            cudaFree(gaussians_dev_);
            gaussians_dev_ = nullptr;
        }
        gaussian_count_ = scene.size();
        if (gaussian_count_ == 0) return;

        const std::size_t bytes = sizeof(Gaussian) * gaussian_count_;
        VKSPLAT_CUDA_CHECK(cudaMalloc(&gaussians_dev_, bytes));
        VKSPLAT_CUDA_CHECK(cudaMemcpyAsync(gaussians_dev_,
                                           scene.gaussians().data(),
                                           bytes,
                                           cudaMemcpyHostToDevice,
                                           stream_));

        if (preprocessed_dev_) cudaFree(preprocessed_dev_);
        VKSPLAT_CUDA_CHECK(cudaMalloc(&preprocessed_dev_,
                                      sizeof(PreprocessedGaussian) * gaussian_count_));
    }

    FrameId render(const Camera& camera,
                   const RenderParams& params,
                   const RenderTarget& /*target*/) {
        ++frame_;

        if (gaussian_count_ == 0) {
            // Empty scene: nothing to do, but advance the timeline so
            // wait() callers do not deadlock.
            return frame_;
        }

        const int W = static_cast<int>(camera.width());
        const int H = static_cast<int>(camera.height());
        const int tile = tunables_.tile_size;
        (void)params;

        const int threads = 256;
        const int blocks  = (static_cast<int>(gaussian_count_) + threads - 1) / threads;

        preprocess_kernel<<<blocks, threads, 0, stream_>>>(
            static_cast<int>(gaussian_count_),
            reinterpret_cast<const Gaussian*>(gaussians_dev_),
            preprocessed_dev_,
            W, H, tile);

        // TODO(v1): allocate (keys,values), launch
        // duplicate_with_keys_kernel, run CUB radix sort, then
        // identify_tile_ranges_kernel, then blend_kernel into the
        // RenderTarget surface (host buffer or interop image).

        return frame_;
    }

    void wait(FrameId /*frame*/) {
        VKSPLAT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_ = nullptr;

    void*                 gaussians_dev_     = nullptr; // Gaussian[gaussian_count_]
    PreprocessedGaussian* preprocessed_dev_  = nullptr;
    std::size_t           gaussian_count_    = 0;

    RasterizerTunables tunables_{};
    FrameId            frame_ = 0;
};

// ---------------------------------------------------------------------
// Public Rasterizer (thin pimpl over RasterizerImpl)
// ---------------------------------------------------------------------

Rasterizer::Rasterizer()
    : impl_(std::make_unique<RasterizerImpl>()) {}

Rasterizer::~Rasterizer() = default;

void Rasterizer::upload(const Scene& scene) { impl_->upload(scene); }

FrameId Rasterizer::render(const Camera& camera,
                           const RenderParams& params,
                           const RenderTarget& target) {
    return impl_->render(camera, params, target);
}

void Rasterizer::wait(FrameId frame) { impl_->wait(frame); }

void Rasterizer::bind_to_device_uuid(const std::array<unsigned char, 16>& uuid) {
    impl_->bind_to_uuid(uuid);
}

} // namespace vksplat::cuda

// ---------------------------------------------------------------------
// Backend factory
// ---------------------------------------------------------------------

namespace vksplat {

std::unique_ptr<Renderer> make_renderer(std::string_view backend_name) {
    if (backend_name == "cuda") {
        return std::make_unique<cuda::Rasterizer>();
    }
    return nullptr;
}

} // namespace vksplat
