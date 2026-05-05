// SPDX-License-Identifier: Apache-2.0
//
// Optional PyTorch binding for tensor-native nvdiffrast ingestion. The CUDA
// kernel remains in src/cuda/gaussian_reconstruction.cu; this file validates
// torch.Tensor layout, allocates SoA tensor outputs, and launches on PyTorch's
// current CUDA stream.

#include <vkgsplat/cuda/gaussian_reconstruction.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <string>

namespace py = pybind11;

namespace {

void check_float_cuda_contiguous(const torch::Tensor& tensor,
                                 const char* name,
                                 std::int64_t channels) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B,H,W,C]");
    TORCH_CHECK(tensor.size(3) == channels, name, " must have ", channels, " channels");
}

void check_optional_like(const c10::optional<torch::Tensor>& tensor,
                         const torch::Tensor& raster,
                         const char* name,
                         std::int64_t channels) {
    if (!tensor.has_value()) {
        return;
    }
    check_float_cuda_contiguous(*tensor, name, channels);
    TORCH_CHECK(tensor->size(0) == raster.size(0) && tensor->size(1) == raster.size(1) &&
                tensor->size(2) == raster.size(2),
                name, " must match raster [B,H,W]");
    TORCH_CHECK(tensor->device() == raster.device(), name, " must be on the same CUDA device as raster");
}

template <typename T>
T* optional_data(const c10::optional<torch::Tensor>& tensor) {
    if (!tensor.has_value()) {
        return nullptr;
    }
    return reinterpret_cast<T*>(tensor->data_ptr<float>());
}

py::dict extract_nvdiffrast_samples_tensor(const torch::Tensor& raster,
                                           std::int64_t max_samples,
                                           const c10::optional<torch::Tensor>& color_rgba,
                                           const c10::optional<torch::Tensor>& world_position,
                                           const c10::optional<torch::Tensor>& normal,
                                           const c10::optional<torch::Tensor>& motion_px,
                                           double min_confidence) {
    check_float_cuda_contiguous(raster, "raster", 4);
    check_optional_like(color_rgba, raster, "color_rgba", 4);
    check_optional_like(world_position, raster, "world_position", 3);
    check_optional_like(normal, raster, "normal", 3);
    check_optional_like(motion_px, raster, "motion_px", 2);
    TORCH_CHECK(max_samples > 0, "max_samples must be positive");
    TORCH_CHECK(max_samples <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()),
                "max_samples exceeds uint32 capacity");

    const c10::cuda::CUDAGuard device_guard(raster.device());
    const auto float_opts = raster.options().dtype(torch::kFloat32);
    const auto int_opts = raster.options().dtype(torch::kInt32);

    auto position = torch::empty({ max_samples, 3 }, float_opts);
    auto out_normal = torch::empty({ max_samples, 3 }, float_opts);
    auto radiance = torch::empty({ max_samples, 4 }, float_opts);
    auto pixel = torch::empty({ max_samples, 2 }, float_opts);
    auto out_motion = torch::empty({ max_samples, 2 }, float_opts);
    auto barycentric_uv = torch::empty({ max_samples, 2 }, float_opts);
    auto depth_confidence = torch::empty({ max_samples, 2 }, float_opts);
    auto ids = torch::empty({ max_samples, 4 }, int_opts);
    auto count = torch::empty({ 1 }, int_opts);
    auto counters = torch::empty({ 6 }, int_opts);

    const vkgsplat::cuda::NvDiffrastRasterLaunch launch{
        static_cast<std::uint32_t>(raster.size(2)),
        static_cast<std::uint32_t>(raster.size(1)),
        static_cast<std::uint32_t>(raster.size(0)),
        static_cast<std::uint32_t>(max_samples),
        static_cast<float>(min_confidence),
    };
    const vkgsplat::cuda::NvDiffrastRasterInputs inputs{
        reinterpret_cast<const vkgsplat::float4*>(raster.data_ptr<float>()),
        optional_data<const vkgsplat::float4>(color_rgba),
        optional_data<const vkgsplat::float3>(world_position),
        optional_data<const vkgsplat::float3>(normal),
        optional_data<const vkgsplat::float2>(motion_px),
    };
    const vkgsplat::cuda::GaussianReconstructionTensorOutputs outputs{
        reinterpret_cast<vkgsplat::float3*>(position.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float3*>(out_normal.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(pixel.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(out_motion.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(barycentric_uv.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };

    vkgsplat::cuda::launch_extract_nvdiffrast_sample_tensors(
        launch,
        inputs,
        outputs,
        reinterpret_cast<std::uint32_t*>(count.data_ptr<int>()),
        reinterpret_cast<vkgsplat::cuda::NvDiffrastExtractCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["position"] = position;
    result["normal"] = out_normal;
    result["radiance"] = radiance;
    result["pixel"] = pixel;
    result["motion_px"] = out_motion;
    result["barycentric_uv"] = barycentric_uv;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["count"] = count;
    result["counters"] = counters;
    return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("extract_nvdiffrast_samples",
          &extract_nvdiffrast_samples_tensor,
          py::arg("raster"),
          py::arg("max_samples"),
          py::arg("color_rgba") = py::none(),
          py::arg("world_position") = py::none(),
          py::arg("normal") = py::none(),
          py::arg("motion_px") = py::none(),
          py::arg("min_confidence") = 0.0,
          "Extract nvdiffrast [B,H,W,C] CUDA tensors into vkGSplat SoA reconstruction tensors.");
}
