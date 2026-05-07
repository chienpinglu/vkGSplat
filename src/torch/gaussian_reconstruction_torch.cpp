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

void check_int_cuda_contiguous(const torch::Tensor& tensor,
                               const char* name,
                               std::int64_t channels) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
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

void check_int_like(const torch::Tensor& tensor,
                    const torch::Tensor& reference,
                    const char* name,
                    std::int64_t channels) {
    check_int_cuda_contiguous(tensor, name, channels);
    TORCH_CHECK(tensor.size(0) == reference.size(0) && tensor.size(1) == reference.size(1) &&
                tensor.size(2) == reference.size(2),
                name, " must match reference [B,H,W]");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as reference");
}

template <typename T>
T* optional_data(const c10::optional<torch::Tensor>& tensor) {
    if (!tensor.has_value()) {
        return nullptr;
    }
    return reinterpret_cast<T*>(tensor->data_ptr<float>());
}


torch::Tensor dict_tensor(const py::dict& dict, const char* key) {
    const py::str py_key(key);
    TORCH_CHECK(dict.contains(py_key), "missing tensor field '", key, "'");
    return py::cast<torch::Tensor>(dict[py_key]);
}

void check_float_cuda_matrix(const torch::Tensor& tensor,
                             const char* name,
                             std::int64_t channels) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [N,C]");
    TORCH_CHECK(tensor.size(1) == channels, name, " must have ", channels, " channels");
}

void check_int_cuda_matrix(const torch::Tensor& tensor,
                           const char* name,
                           std::int64_t channels) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [N,C]");
    TORCH_CHECK(tensor.size(1) == channels, name, " must have ", channels, " channels");
}

void check_int_cuda_vector(const torch::Tensor& tensor,
                           const char* name,
                           std::int64_t min_elements) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, name, " must have shape [N]");
    TORCH_CHECK(tensor.numel() >= min_elements, name, " must have at least ", min_elements, " elements");
}

void check_float_cuda_image(const torch::Tensor& tensor,
                            const char* name,
                            std::int64_t height,
                            std::int64_t width,
                            std::int64_t channels) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape [H,W,C]");
    TORCH_CHECK(tensor.size(0) == height && tensor.size(1) == width && tensor.size(2) == channels,
                name, " must have shape [", height, ",", width, ",", channels, "]");
}

void check_int_cuda_image(const torch::Tensor& tensor,
                          const char* name,
                          std::int64_t height,
                          std::int64_t width,
                          std::int64_t channels) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape [H,W,C]");
    TORCH_CHECK(tensor.size(0) == height && tensor.size(1) == width && tensor.size(2) == channels,
                name, " must have shape [", height, ",", width, ",", channels, "]");
}

void check_same_device(const torch::Tensor& tensor,
                       const torch::Tensor& reference,
                       const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(),
                name, " must be on the same CUDA device as the reference tensor");
}

void check_uint32_range(std::int64_t value, const char* name) {
    TORCH_CHECK(value >= 0, name, " must be non-negative");
    TORCH_CHECK(value <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()),
                name, " exceeds uint32 capacity");
}

vkgsplat::cuda::GaussianReconstructionState state_view_from_dict(const py::dict& state) {
    const auto position = dict_tensor(state, "position");
    check_float_cuda_matrix(position, "state.position", 3);
    const std::int64_t capacity = position.size(0);

    const auto normal = dict_tensor(state, "normal");
    const auto radiance = dict_tensor(state, "radiance");
    const auto pixel = dict_tensor(state, "pixel");
    const auto motion_px = dict_tensor(state, "motion_px");
    const auto covariance_diag = dict_tensor(state, "covariance_diag");
    const auto depth_confidence = dict_tensor(state, "depth_confidence");
    const auto mass_variance = dict_tensor(state, "mass_variance");
    const auto ids = dict_tensor(state, "ids");

    check_float_cuda_matrix(normal, "state.normal", 3);
    check_float_cuda_matrix(radiance, "state.radiance", 4);
    check_float_cuda_matrix(pixel, "state.pixel", 2);
    check_float_cuda_matrix(motion_px, "state.motion_px", 2);
    check_float_cuda_matrix(covariance_diag, "state.covariance_diag", 3);
    check_float_cuda_matrix(depth_confidence, "state.depth_confidence", 2);
    check_float_cuda_matrix(mass_variance, "state.mass_variance", 2);
    check_int_cuda_matrix(ids, "state.ids", 4);

    TORCH_CHECK(normal.size(0) == capacity && radiance.size(0) == capacity &&
                pixel.size(0) == capacity && motion_px.size(0) == capacity &&
                covariance_diag.size(0) == capacity && depth_confidence.size(0) == capacity &&
                mass_variance.size(0) == capacity && ids.size(0) == capacity,
                "all state tensors must have the same first dimension");
    check_same_device(normal, position, "state.normal");
    check_same_device(radiance, position, "state.radiance");
    check_same_device(pixel, position, "state.pixel");
    check_same_device(motion_px, position, "state.motion_px");
    check_same_device(covariance_diag, position, "state.covariance_diag");
    check_same_device(depth_confidence, position, "state.depth_confidence");
    check_same_device(mass_variance, position, "state.mass_variance");
    check_same_device(ids, position, "state.ids");

    return {
        reinterpret_cast<vkgsplat::float3*>(position.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float3*>(normal.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(pixel.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(motion_px.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float3*>(covariance_diag.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(mass_variance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
}

vkgsplat::cuda::GaussianReconstructionTensorSamples sample_view_from_dict(const py::dict& samples) {
    const auto position = dict_tensor(samples, "position");
    check_float_cuda_matrix(position, "samples.position", 3);
    const std::int64_t sample_capacity = position.size(0);

    const auto normal = dict_tensor(samples, "normal");
    const auto radiance = dict_tensor(samples, "radiance");
    const auto pixel = dict_tensor(samples, "pixel");
    const auto motion_px = dict_tensor(samples, "motion_px");
    const auto barycentric_uv = dict_tensor(samples, "barycentric_uv");
    const auto depth_confidence = dict_tensor(samples, "depth_confidence");
    const auto ids = dict_tensor(samples, "ids");

    check_float_cuda_matrix(normal, "samples.normal", 3);
    check_float_cuda_matrix(radiance, "samples.radiance", 4);
    check_float_cuda_matrix(pixel, "samples.pixel", 2);
    check_float_cuda_matrix(motion_px, "samples.motion_px", 2);
    check_float_cuda_matrix(barycentric_uv, "samples.barycentric_uv", 2);
    check_float_cuda_matrix(depth_confidence, "samples.depth_confidence", 2);
    check_int_cuda_matrix(ids, "samples.ids", 4);

    TORCH_CHECK(normal.size(0) == sample_capacity && radiance.size(0) == sample_capacity &&
                pixel.size(0) == sample_capacity && motion_px.size(0) == sample_capacity &&
                barycentric_uv.size(0) == sample_capacity && depth_confidence.size(0) == sample_capacity &&
                ids.size(0) == sample_capacity,
                "all sample tensors must have the same first dimension");
    check_same_device(normal, position, "samples.normal");
    check_same_device(radiance, position, "samples.radiance");
    check_same_device(pixel, position, "samples.pixel");
    check_same_device(motion_px, position, "samples.motion_px");
    check_same_device(barycentric_uv, position, "samples.barycentric_uv");
    check_same_device(depth_confidence, position, "samples.depth_confidence");
    check_same_device(ids, position, "samples.ids");

    return {
        reinterpret_cast<const vkgsplat::float3*>(position.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::float3*>(normal.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::float2*>(pixel.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::float2*>(motion_px.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::float2*>(barycentric_uv.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
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


py::dict extract_seed_frame_samples_tensor(const torch::Tensor& radiance,
                                           const torch::Tensor& depth_confidence,
                                           const torch::Tensor& ids,
                                           std::int64_t max_samples,
                                           const c10::optional<torch::Tensor>& world_position,
                                           const c10::optional<torch::Tensor>& normal,
                                           const c10::optional<torch::Tensor>& motion_px,
                                           double min_confidence) {
    check_float_cuda_contiguous(radiance, "radiance", 4);
    check_float_cuda_contiguous(depth_confidence, "depth_confidence", 2);
    TORCH_CHECK(depth_confidence.size(0) == radiance.size(0) &&
                depth_confidence.size(1) == radiance.size(1) &&
                depth_confidence.size(2) == radiance.size(2),
                "depth_confidence must match radiance [B,H,W]");
    TORCH_CHECK(depth_confidence.device() == radiance.device(),
                "depth_confidence must be on the same CUDA device as radiance");
    check_int_like(ids, radiance, "ids", 4);
    check_optional_like(world_position, radiance, "world_position", 3);
    check_optional_like(normal, radiance, "normal", 3);
    check_optional_like(motion_px, radiance, "motion_px", 2);
    TORCH_CHECK(max_samples > 0, "max_samples must be positive");
    check_uint32_range(max_samples, "max_samples");

    const c10::cuda::CUDAGuard device_guard(radiance.device());
    const auto float_opts = radiance.options().dtype(torch::kFloat32);
    const auto int_opts = radiance.options().dtype(torch::kInt32);

    auto position = torch::empty({ max_samples, 3 }, float_opts);
    auto out_normal = torch::empty({ max_samples, 3 }, float_opts);
    auto out_radiance = torch::empty({ max_samples, 4 }, float_opts);
    auto pixel = torch::empty({ max_samples, 2 }, float_opts);
    auto out_motion = torch::empty({ max_samples, 2 }, float_opts);
    auto barycentric_uv = torch::empty({ max_samples, 2 }, float_opts);
    auto out_depth_confidence = torch::empty({ max_samples, 2 }, float_opts);
    auto out_ids = torch::empty({ max_samples, 4 }, int_opts);
    auto count = torch::empty({ 1 }, int_opts);
    auto counters = torch::empty({ 6 }, int_opts);

    const vkgsplat::cuda::GaussianSeedFrameTensorLaunch launch{
        static_cast<std::uint32_t>(radiance.size(2)),
        static_cast<std::uint32_t>(radiance.size(1)),
        static_cast<std::uint32_t>(radiance.size(0)),
        static_cast<std::uint32_t>(max_samples),
        static_cast<float>(min_confidence),
    };
    const vkgsplat::cuda::GaussianSeedFrameTensorInputs inputs{
        reinterpret_cast<const vkgsplat::float4*>(radiance.data_ptr<float>()),
        optional_data<const vkgsplat::float3>(world_position),
        optional_data<const vkgsplat::float3>(normal),
        optional_data<const vkgsplat::float2>(motion_px),
        reinterpret_cast<const vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<const vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianReconstructionTensorOutputs outputs{
        reinterpret_cast<vkgsplat::float3*>(position.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float3*>(out_normal.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float4*>(out_radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(pixel.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(out_motion.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(barycentric_uv.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(out_depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(out_ids.data_ptr<int>()),
    };

    vkgsplat::cuda::launch_extract_seed_frame_sample_tensors(
        launch,
        inputs,
        outputs,
        reinterpret_cast<std::uint32_t*>(count.data_ptr<int>()),
        reinterpret_cast<vkgsplat::cuda::GaussianSeedFrameExtractCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["position"] = position;
    result["normal"] = out_normal;
    result["radiance"] = out_radiance;
    result["pixel"] = pixel;
    result["motion_px"] = out_motion;
    result["barycentric_uv"] = barycentric_uv;
    result["depth_confidence"] = out_depth_confidence;
    result["ids"] = out_ids;
    result["count"] = count;
    result["counters"] = counters;
    return result;
}


py::dict create_gaussian_state_tensor(const torch::Tensor& like,
                                      std::int64_t gaussian_capacity,
                                      bool clear_state) {
    TORCH_CHECK(like.defined(), "like must be defined");
    TORCH_CHECK(like.is_cuda(), "like must be a CUDA tensor");
    TORCH_CHECK(gaussian_capacity > 0, "gaussian_capacity must be positive");
    check_uint32_range(gaussian_capacity, "gaussian_capacity");

    const c10::cuda::CUDAGuard device_guard(like.device());
    const auto float_opts = like.options().dtype(torch::kFloat32);
    const auto int_opts = like.options().dtype(torch::kInt32);

    py::dict state;
    state["position"] = torch::empty({ gaussian_capacity, 3 }, float_opts);
    state["normal"] = torch::empty({ gaussian_capacity, 3 }, float_opts);
    state["radiance"] = torch::empty({ gaussian_capacity, 4 }, float_opts);
    state["pixel"] = torch::empty({ gaussian_capacity, 2 }, float_opts);
    state["motion_px"] = torch::empty({ gaussian_capacity, 2 }, float_opts);
    state["covariance_diag"] = torch::empty({ gaussian_capacity, 3 }, float_opts);
    state["depth_confidence"] = torch::empty({ gaussian_capacity, 2 }, float_opts);
    state["mass_variance"] = torch::empty({ gaussian_capacity, 2 }, float_opts);
    state["ids"] = torch::empty({ gaussian_capacity, 4 }, int_opts);

    if (clear_state) {
        const auto state_view = state_view_from_dict(state);
        vkgsplat::cuda::launch_clear_gaussian_reconstruction_state(
            state_view,
            static_cast<std::uint32_t>(gaussian_capacity),
            reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
    }
    return state;
}

py::dict clear_gaussian_state_tensor(py::dict state) {
    const auto position = dict_tensor(state, "position");
    check_float_cuda_matrix(position, "state.position", 3);
    check_uint32_range(position.size(0), "gaussian_capacity");
    const c10::cuda::CUDAGuard device_guard(position.device());
    const auto state_view = state_view_from_dict(state);
    vkgsplat::cuda::launch_clear_gaussian_reconstruction_state(
        state_view,
        static_cast<std::uint32_t>(position.size(0)),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
    return state;
}

torch::Tensor sample_count_info_tensor(const torch::Tensor& sample_count,
                                      std::int64_t max_samples) {
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_uint32_range(max_samples, "max_samples");

    const c10::cuda::CUDAGuard device_guard(sample_count.device());
    auto info = torch::empty({ 4 }, sample_count.options().dtype(torch::kInt32));
    const vkgsplat::cuda::GaussianSampleCountInfoLaunch launch{
        static_cast<std::uint32_t>(max_samples),
    };
    vkgsplat::cuda::launch_read_gaussian_sample_count_info(
        launch,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        reinterpret_cast<vkgsplat::cuda::GaussianSampleCountInfo*>(info.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
    return info;
}

torch::Tensor accumulate_gaussian_state_from_samples_tensor(const py::dict& samples,
                                                            const torch::Tensor& sample_count,
                                                            py::dict state,
                                                            double min_confidence) {
    const auto sample_position = dict_tensor(samples, "position");
    const auto state_position = dict_tensor(state, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_float_cuda_matrix(state_position, "state.position", 3);
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_same_device(state_position, sample_position, "state.position");
    check_same_device(sample_count, sample_position, "sample_count");
    check_uint32_range(sample_position.size(0), "max_samples");
    check_uint32_range(state_position.size(0), "gaussian_capacity");

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto state_view = state_view_from_dict(state);
    auto counters = torch::empty({ 6 }, state_position.options().dtype(torch::kInt32));

    const vkgsplat::cuda::GaussianStateTensorCountUpdateLaunch launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(state_position.size(0)),
        static_cast<float>(min_confidence),
    };
    vkgsplat::cuda::launch_accumulate_gaussian_state_from_sample_tensors_counted(
        launch,
        sample_view,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        state_view,
        reinterpret_cast<vkgsplat::cuda::GaussianStateUpdateCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
    return counters;
}

py::dict build_gaussian_sample_tile_bins_tensor(const py::dict& samples,
                                                const torch::Tensor& sample_count,
                                                std::int64_t height,
                                                std::int64_t width,
                                                std::int64_t tile_height,
                                                std::int64_t tile_width) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_same_device(sample_count, sample_position, "sample_count");
    check_uint32_range(sample_position.size(0), "max_samples");

    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);
    auto sample_tile = torch::empty({ sample_position.size(0) }, int_opts);
    auto tile_counts = torch::empty({ tiles_y, tiles_x }, int_opts);
    auto counters = torch::empty({ 5 }, int_opts);

    const vkgsplat::cuda::GaussianSampleTileBinningLaunch launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
    };
    const vkgsplat::cuda::GaussianSampleTileBinningOutputs outputs{
        reinterpret_cast<std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_bins(
        launch,
        sample_view,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileBinningCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["sample_tile"] = sample_tile;
    result["tile_counts"] = tile_counts;
    result["counters"] = counters;
    result["tiles_y"] = tiles_y;
    result["tiles_x"] = tiles_x;
    return result;
}

py::dict build_gaussian_sample_tile_offsets_tensor(const torch::Tensor& tile_counts) {
    TORCH_CHECK(tile_counts.defined(), "tile_counts must be defined");
    TORCH_CHECK(tile_counts.is_cuda(), "tile_counts must be a CUDA tensor");
    TORCH_CHECK(tile_counts.scalar_type() == torch::kInt32, "tile_counts must be int32");
    TORCH_CHECK(tile_counts.is_contiguous(), "tile_counts must be contiguous");
    TORCH_CHECK(tile_counts.dim() == 1 || tile_counts.dim() == 2,
                "tile_counts must have shape [T] or [Ty,Tx]");
    check_uint32_range(tile_counts.numel(), "tile_count");

    const c10::cuda::CUDAGuard device_guard(tile_counts.device());
    const auto flat_tile_counts = tile_counts.reshape({ tile_counts.numel() });
    const auto int_opts = tile_counts.options().dtype(torch::kInt32);
    auto tile_offsets = torch::empty({ tile_counts.numel() + 1 }, int_opts);
    auto counters = torch::empty({ 2 }, int_opts);

    const vkgsplat::cuda::GaussianSampleTileOffsetLaunch launch{
        static_cast<std::uint32_t>(tile_counts.numel()),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetInputs inputs{
        reinterpret_cast<const std::uint32_t*>(flat_tile_counts.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetOutputs outputs{
        reinterpret_cast<std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_offsets(
        launch,
        inputs,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileOffsetCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["tile_offsets"] = tile_offsets;
    result["counters"] = counters;
    return result;
}

py::dict compact_gaussian_sample_tile_bins_tensor(const torch::Tensor& sample_tile,
                                                  const torch::Tensor& tile_offsets,
                                                  const torch::Tensor& sample_count) {
    check_int_cuda_vector(sample_tile, "sample_tile", 1);
    check_int_cuda_vector(tile_offsets, "tile_offsets", 2);
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_same_device(tile_offsets, sample_tile, "tile_offsets");
    check_same_device(sample_count, sample_tile, "sample_count");
    check_uint32_range(sample_tile.numel(), "max_samples");
    const std::int64_t tile_count = tile_offsets.numel() - 1;
    check_uint32_range(tile_count, "tile_count");

    const c10::cuda::CUDAGuard device_guard(sample_tile.device());
    const auto int_opts = sample_tile.options().dtype(torch::kInt32);
    auto tile_write_counts = torch::empty({ tile_count }, int_opts);
    auto tile_sample_indices = torch::empty({ sample_tile.numel() }, int_opts);
    auto counters = torch::empty({ 4 }, int_opts);

    const vkgsplat::cuda::GaussianSampleTileCompactionLaunch launch{
        static_cast<std::uint32_t>(sample_tile.numel()),
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionInputs inputs{
        reinterpret_cast<const std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionOutputs outputs{
        reinterpret_cast<std::uint32_t*>(tile_write_counts.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_compact_gaussian_sample_tile_bins(
        launch,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        inputs,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileCompactionCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["tile_write_counts"] = tile_write_counts;
    result["tile_sample_indices"] = tile_sample_indices;
    result["counters"] = counters;
    return result;
}


py::dict build_gaussian_sample_tile_spans_tensor(const py::dict& samples,
                                                 const torch::Tensor& sample_count,
                                                 std::int64_t height,
                                                 std::int64_t width,
                                                 std::int64_t tile_height,
                                                 std::int64_t tile_width) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_same_device(sample_count, sample_position, "sample_count");
    check_uint32_range(sample_position.size(0), "max_samples");

    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");
    const std::int64_t tile_count = tiles_x * tiles_y;

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);
    auto sample_tile = torch::empty({ sample_position.size(0) }, int_opts);
    auto tile_counts = torch::empty({ tiles_y, tiles_x }, int_opts);
    auto tile_offsets = torch::empty({ tile_count + 1 }, int_opts);
    auto tile_write_counts = torch::empty({ tile_count }, int_opts);
    auto tile_sample_indices = torch::empty({ sample_position.size(0) }, int_opts);
    auto binning_counters = torch::empty({ 5 }, int_opts);
    auto offset_counters = torch::empty({ 2 }, int_opts);
    auto compaction_counters = torch::empty({ 4 }, int_opts);

    const vkgsplat::cuda::GaussianSampleTileBinningLaunch binning_launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
    };
    const vkgsplat::cuda::GaussianSampleTileBinningOutputs binning_outputs{
        reinterpret_cast<std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_bins(
        binning_launch,
        sample_view,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        binning_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileBinningCounters*>(binning_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    const vkgsplat::cuda::GaussianSampleTileOffsetLaunch offset_launch{
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetInputs offset_inputs{
        reinterpret_cast<const std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetOutputs offset_outputs{
        reinterpret_cast<std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_offsets(
        offset_launch,
        offset_inputs,
        offset_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileOffsetCounters*>(offset_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    const vkgsplat::cuda::GaussianSampleTileCompactionLaunch compaction_launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionInputs compaction_inputs{
        reinterpret_cast<const std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionOutputs compaction_outputs{
        reinterpret_cast<std::uint32_t*>(tile_write_counts.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_compact_gaussian_sample_tile_bins(
        compaction_launch,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        compaction_inputs,
        compaction_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileCompactionCounters*>(compaction_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["sample_tile"] = sample_tile;
    result["tile_counts"] = tile_counts;
    result["tile_offsets"] = tile_offsets;
    result["tile_write_counts"] = tile_write_counts;
    result["tile_sample_indices"] = tile_sample_indices;
    result["binning_counters"] = binning_counters;
    result["offset_counters"] = offset_counters;
    result["compaction_counters"] = compaction_counters;
    result["tiles_y"] = tiles_y;
    result["tiles_x"] = tiles_x;
    return result;
}

py::dict resolve_gaussian_sample_frame_weighted_tensor(const py::dict& samples,
                                                       const torch::Tensor& sample_count,
                                                       std::int64_t height,
                                                       std::int64_t width,
                                                       std::int64_t tile_height,
                                                       std::int64_t tile_width,
                                                       std::int64_t radius_px,
                                                       double sigma_px,
                                                       double min_confidence) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    TORCH_CHECK(radius_px >= 0 && radius_px <= 8, "radius_px must be in [0, 8]");
    TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");
    check_uint32_range(radius_px, "radius_px");

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_same_device(sample_count, sample_position, "sample_count");
    check_uint32_range(sample_position.size(0), "max_samples");

    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");
    const std::int64_t tile_count = tiles_x * tiles_y;

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto float_opts = sample_position.options().dtype(torch::kFloat32);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);

    auto sample_tile = torch::empty({ sample_position.size(0) }, int_opts);
    auto tile_counts = torch::empty({ tiles_y, tiles_x }, int_opts);
    auto tile_offsets = torch::empty({ tile_count + 1 }, int_opts);
    auto tile_write_counts = torch::empty({ tile_count }, int_opts);
    auto tile_sample_indices = torch::empty({ sample_position.size(0) }, int_opts);
    auto binning_counters = torch::empty({ 5 }, int_opts);
    auto offset_counters = torch::empty({ 2 }, int_opts);
    auto compaction_counters = torch::empty({ 4 }, int_opts);

    const vkgsplat::cuda::GaussianSampleTileBinningLaunch binning_launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
    };
    const vkgsplat::cuda::GaussianSampleTileBinningOutputs binning_outputs{
        reinterpret_cast<std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_bins(
        binning_launch,
        sample_view,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        binning_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileBinningCounters*>(binning_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    const vkgsplat::cuda::GaussianSampleTileOffsetLaunch offset_launch{
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetInputs offset_inputs{
        reinterpret_cast<const std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetOutputs offset_outputs{
        reinterpret_cast<std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_offsets(
        offset_launch,
        offset_inputs,
        offset_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileOffsetCounters*>(offset_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    const vkgsplat::cuda::GaussianSampleTileCompactionLaunch compaction_launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionInputs compaction_inputs{
        reinterpret_cast<const std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionOutputs compaction_outputs{
        reinterpret_cast<std::uint32_t*>(tile_write_counts.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_compact_gaussian_sample_tile_bins(
        compaction_launch,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        compaction_inputs,
        compaction_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileCompactionCounters*>(compaction_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto resolve_counters = torch::empty({ 9 }, int_opts);

    const vkgsplat::cuda::GaussianTileWeightedResolveLaunch resolve_launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
        static_cast<std::uint32_t>(tile_count),
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(radius_px),
        static_cast<float>(sigma_px),
        static_cast<float>(min_confidence),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveInputs resolve_inputs{
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveOutputs resolve_outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_resolve_gaussian_sample_tiles_weighted(
        resolve_launch,
        sample_view,
        resolve_inputs,
        resolve_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianTileWeightedResolveCounters*>(resolve_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["sample_tile"] = sample_tile;
    result["tile_counts"] = tile_counts;
    result["tile_offsets"] = tile_offsets;
    result["tile_write_counts"] = tile_write_counts;
    result["tile_sample_indices"] = tile_sample_indices;
    result["binning_counters"] = binning_counters;
    result["offset_counters"] = offset_counters;
    result["compaction_counters"] = compaction_counters;
    result["resolve_counters"] = resolve_counters;
    result["tiles_y"] = tiles_y;
    result["tiles_x"] = tiles_x;
    return result;
}


py::dict resolve_gaussian_sample_frame_weighted_gated_tensor(const py::dict& samples,
                                                             const torch::Tensor& sample_count,
                                                             const py::dict& guides,
                                                             std::int64_t height,
                                                             std::int64_t width,
                                                             std::int64_t tile_height,
                                                             std::int64_t tile_width,
                                                             std::int64_t radius_px,
                                                             double sigma_px,
                                                             double min_confidence,
                                                             std::int64_t gate_flags,
                                                             double depth_epsilon,
                                                             double normal_dot_min,
                                                             double motion_epsilon_px) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    TORCH_CHECK(radius_px >= 0 && radius_px <= 8, "radius_px must be in [0, 8]");
    TORCH_CHECK(gate_flags >= 0, "gate_flags must be non-negative");
    TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");
    TORCH_CHECK(depth_epsilon >= 0.0, "depth_epsilon must be non-negative");
    TORCH_CHECK(normal_dot_min >= -1.0 && normal_dot_min <= 1.0,
                "normal_dot_min must be in [-1, 1]");
    TORCH_CHECK(motion_epsilon_px >= 0.0, "motion_epsilon_px must be non-negative");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");
    check_uint32_range(radius_px, "radius_px");
    check_uint32_range(gate_flags, "gate_flags");

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_int_cuda_vector(sample_count, "sample_count", 1);
    check_same_device(sample_count, sample_position, "sample_count");
    check_uint32_range(sample_position.size(0), "max_samples");

    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");
    const std::int64_t tile_count = tiles_x * tiles_y;

    const std::uint32_t gate_mask = static_cast<std::uint32_t>(gate_flags);
    const bool primitive_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_PRIMITIVE_ID) != 0u;
    const bool depth_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_DEPTH) != 0u;
    const bool normal_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_NORMAL) != 0u;
    const bool motion_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_MOTION) != 0u;

    torch::Tensor guide_depth_confidence;
    torch::Tensor guide_ids;
    torch::Tensor guide_normal;
    torch::Tensor guide_motion;
    const vkgsplat::float2* guide_depth_ptr = nullptr;
    const vkgsplat::cuda::GaussianReconstructionSampleIds* guide_ids_ptr = nullptr;
    const vkgsplat::float3* guide_normal_ptr = nullptr;
    const vkgsplat::float2* guide_motion_ptr = nullptr;

    if (depth_gate) {
        guide_depth_confidence = dict_tensor(guides, "depth_confidence");
        check_float_cuda_image(guide_depth_confidence, "guides.depth_confidence", height, width, 2);
        check_same_device(guide_depth_confidence, sample_position, "guides.depth_confidence");
        guide_depth_ptr = reinterpret_cast<const vkgsplat::float2*>(guide_depth_confidence.data_ptr<float>());
    }
    if (primitive_gate) {
        guide_ids = dict_tensor(guides, "ids");
        check_int_cuda_image(guide_ids, "guides.ids", height, width, 4);
        check_same_device(guide_ids, sample_position, "guides.ids");
        guide_ids_ptr = reinterpret_cast<const vkgsplat::cuda::GaussianReconstructionSampleIds*>(
            guide_ids.data_ptr<int>());
    }
    if (normal_gate) {
        guide_normal = dict_tensor(guides, "normal");
        check_float_cuda_image(guide_normal, "guides.normal", height, width, 3);
        check_same_device(guide_normal, sample_position, "guides.normal");
        guide_normal_ptr = reinterpret_cast<const vkgsplat::float3*>(guide_normal.data_ptr<float>());
    }
    if (motion_gate) {
        guide_motion = dict_tensor(guides, "motion_px");
        check_float_cuda_image(guide_motion, "guides.motion_px", height, width, 2);
        check_same_device(guide_motion, sample_position, "guides.motion_px");
        guide_motion_ptr = reinterpret_cast<const vkgsplat::float2*>(guide_motion.data_ptr<float>());
    }

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto float_opts = sample_position.options().dtype(torch::kFloat32);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);

    auto sample_tile = torch::empty({ sample_position.size(0) }, int_opts);
    auto tile_counts = torch::empty({ tiles_y, tiles_x }, int_opts);
    auto tile_offsets = torch::empty({ tile_count + 1 }, int_opts);
    auto tile_write_counts = torch::empty({ tile_count }, int_opts);
    auto tile_sample_indices = torch::empty({ sample_position.size(0) }, int_opts);
    auto binning_counters = torch::empty({ 5 }, int_opts);
    auto offset_counters = torch::empty({ 2 }, int_opts);
    auto compaction_counters = torch::empty({ 4 }, int_opts);

    const vkgsplat::cuda::GaussianSampleTileBinningLaunch binning_launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
    };
    const vkgsplat::cuda::GaussianSampleTileBinningOutputs binning_outputs{
        reinterpret_cast<std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_bins(
        binning_launch,
        sample_view,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        binning_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileBinningCounters*>(binning_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    const vkgsplat::cuda::GaussianSampleTileOffsetLaunch offset_launch{
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetInputs offset_inputs{
        reinterpret_cast<const std::uint32_t*>(tile_counts.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileOffsetOutputs offset_outputs{
        reinterpret_cast<std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_build_gaussian_sample_tile_offsets(
        offset_launch,
        offset_inputs,
        offset_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileOffsetCounters*>(offset_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    const vkgsplat::cuda::GaussianSampleTileCompactionLaunch compaction_launch{
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(tile_count),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionInputs compaction_inputs{
        reinterpret_cast<const std::uint32_t*>(sample_tile.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianSampleTileCompactionOutputs compaction_outputs{
        reinterpret_cast<std::uint32_t*>(tile_write_counts.data_ptr<int>()),
        reinterpret_cast<std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_compact_gaussian_sample_tile_bins(
        compaction_launch,
        reinterpret_cast<const std::uint32_t*>(sample_count.data_ptr<int>()),
        compaction_inputs,
        compaction_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianSampleTileCompactionCounters*>(compaction_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto resolve_counters = torch::empty({ 13 }, int_opts);

    const vkgsplat::cuda::GaussianTileGatedWeightedResolveLaunch resolve_launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
        static_cast<std::uint32_t>(tile_count),
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(radius_px),
        gate_mask,
        static_cast<float>(sigma_px),
        static_cast<float>(min_confidence),
        static_cast<float>(depth_epsilon),
        static_cast<float>(normal_dot_min),
        static_cast<float>(motion_epsilon_px),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveInputs resolve_inputs{
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianTileResolveGuides guide_view{
        guide_depth_ptr,
        guide_ids_ptr,
        guide_normal_ptr,
        guide_motion_ptr,
    };
    const vkgsplat::cuda::GaussianTileSampleResolveOutputs resolve_outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_resolve_gaussian_sample_tiles_weighted_gated(
        resolve_launch,
        sample_view,
        resolve_inputs,
        guide_view,
        resolve_outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianTileGatedWeightedResolveCounters*>(resolve_counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["sample_tile"] = sample_tile;
    result["tile_counts"] = tile_counts;
    result["tile_offsets"] = tile_offsets;
    result["tile_write_counts"] = tile_write_counts;
    result["tile_sample_indices"] = tile_sample_indices;
    result["binning_counters"] = binning_counters;
    result["offset_counters"] = offset_counters;
    result["compaction_counters"] = compaction_counters;
    result["resolve_counters"] = resolve_counters;
    result["tiles_y"] = tiles_y;
    result["tiles_x"] = tiles_x;
    return result;
}


py::dict resolve_gaussian_sample_tiles_tensor(const py::dict& samples,
                                              const torch::Tensor& tile_offsets,
                                              const torch::Tensor& tile_sample_indices,
                                              std::int64_t height,
                                              std::int64_t width,
                                              std::int64_t tile_height,
                                              std::int64_t tile_width,
                                              double min_confidence) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");
    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");
    const std::int64_t tile_count = tiles_x * tiles_y;

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_uint32_range(sample_position.size(0), "max_samples");
    check_int_cuda_vector(tile_offsets, "tile_offsets", tile_count + 1);
    check_int_cuda_vector(tile_sample_indices, "tile_sample_indices", sample_position.size(0));
    TORCH_CHECK(tile_offsets.numel() == tile_count + 1,
                "tile_offsets must have exactly tile_count + 1 elements");
    TORCH_CHECK(tile_sample_indices.numel() >= sample_position.size(0),
                "tile_sample_indices must have at least max_samples elements");
    check_same_device(tile_offsets, sample_position, "tile_offsets");
    check_same_device(tile_sample_indices, sample_position, "tile_sample_indices");

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto float_opts = sample_position.options().dtype(torch::kFloat32);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);
    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto counters = torch::empty({ 7 }, int_opts);

    const vkgsplat::cuda::GaussianTileSampleResolveLaunch launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
        static_cast<std::uint32_t>(tile_count),
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<float>(min_confidence),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveInputs inputs{
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveOutputs outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_resolve_gaussian_sample_tiles(
        launch,
        sample_view,
        inputs,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianTileSampleResolveCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["counters"] = counters;
    return result;
}

py::dict resolve_gaussian_sample_tiles_weighted_tensor(const py::dict& samples,
                                                       const torch::Tensor& tile_offsets,
                                                       const torch::Tensor& tile_sample_indices,
                                                       std::int64_t height,
                                                       std::int64_t width,
                                                       std::int64_t tile_height,
                                                       std::int64_t tile_width,
                                                       std::int64_t radius_px,
                                                       double sigma_px,
                                                       double min_confidence) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    TORCH_CHECK(radius_px >= 0 && radius_px <= 8, "radius_px must be in [0, 8]");
    TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");
    check_uint32_range(radius_px, "radius_px");
    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");
    const std::int64_t tile_count = tiles_x * tiles_y;

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_uint32_range(sample_position.size(0), "max_samples");
    check_int_cuda_vector(tile_offsets, "tile_offsets", tile_count + 1);
    check_int_cuda_vector(tile_sample_indices, "tile_sample_indices", sample_position.size(0));
    TORCH_CHECK(tile_offsets.numel() == tile_count + 1,
                "tile_offsets must have exactly tile_count + 1 elements");
    TORCH_CHECK(tile_sample_indices.numel() >= sample_position.size(0),
                "tile_sample_indices must have at least max_samples elements");
    check_same_device(tile_offsets, sample_position, "tile_offsets");
    check_same_device(tile_sample_indices, sample_position, "tile_sample_indices");

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto float_opts = sample_position.options().dtype(torch::kFloat32);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);
    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto counters = torch::empty({ 9 }, int_opts);

    const vkgsplat::cuda::GaussianTileWeightedResolveLaunch launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
        static_cast<std::uint32_t>(tile_count),
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(radius_px),
        static_cast<float>(sigma_px),
        static_cast<float>(min_confidence),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveInputs inputs{
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveOutputs outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_resolve_gaussian_sample_tiles_weighted(
        launch,
        sample_view,
        inputs,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianTileWeightedResolveCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["counters"] = counters;
    return result;
}

py::dict resolve_gaussian_sample_tiles_weighted_gated_tensor(const py::dict& samples,
                                                             const torch::Tensor& tile_offsets,
                                                             const torch::Tensor& tile_sample_indices,
                                                             const py::dict& guides,
                                                             std::int64_t height,
                                                             std::int64_t width,
                                                             std::int64_t tile_height,
                                                             std::int64_t tile_width,
                                                             std::int64_t radius_px,
                                                             double sigma_px,
                                                             double min_confidence,
                                                             std::int64_t gate_flags,
                                                             double depth_epsilon,
                                                             double normal_dot_min,
                                                             double motion_epsilon_px) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(tile_height > 0 && tile_width > 0, "tile dimensions must be positive");
    TORCH_CHECK(radius_px >= 0 && radius_px <= 8, "radius_px must be in [0, 8]");
    TORCH_CHECK(gate_flags >= 0, "gate_flags must be non-negative");
    TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");
    TORCH_CHECK(depth_epsilon >= 0.0, "depth_epsilon must be non-negative");
    TORCH_CHECK(normal_dot_min >= -1.0 && normal_dot_min <= 1.0,
                "normal_dot_min must be in [-1, 1]");
    TORCH_CHECK(motion_epsilon_px >= 0.0, "motion_epsilon_px must be non-negative");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(tile_height, "tile_height");
    check_uint32_range(tile_width, "tile_width");
    check_uint32_range(radius_px, "radius_px");
    check_uint32_range(gate_flags, "gate_flags");
    const std::int64_t tiles_y = (height + tile_height - 1) / tile_height;
    const std::int64_t tiles_x = (width + tile_width - 1) / tile_width;
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(tiles_x <= max_uint32 / tiles_y, "tile grid exceeds uint32 capacity");
    const std::int64_t tile_count = tiles_x * tiles_y;

    const auto sample_position = dict_tensor(samples, "position");
    check_float_cuda_matrix(sample_position, "samples.position", 3);
    check_uint32_range(sample_position.size(0), "max_samples");
    check_int_cuda_vector(tile_offsets, "tile_offsets", tile_count + 1);
    check_int_cuda_vector(tile_sample_indices, "tile_sample_indices", sample_position.size(0));
    TORCH_CHECK(tile_offsets.numel() == tile_count + 1,
                "tile_offsets must have exactly tile_count + 1 elements");
    TORCH_CHECK(tile_sample_indices.numel() >= sample_position.size(0),
                "tile_sample_indices must have at least max_samples elements");
    check_same_device(tile_offsets, sample_position, "tile_offsets");
    check_same_device(tile_sample_indices, sample_position, "tile_sample_indices");

    const std::uint32_t gate_mask = static_cast<std::uint32_t>(gate_flags);
    const bool primitive_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_PRIMITIVE_ID) != 0u;
    const bool depth_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_DEPTH) != 0u;
    const bool normal_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_NORMAL) != 0u;
    const bool motion_gate = (gate_mask & vkgsplat::cuda::GAUSSIAN_TILE_GATE_MOTION) != 0u;

    torch::Tensor guide_depth_confidence;
    torch::Tensor guide_ids;
    torch::Tensor guide_normal;
    torch::Tensor guide_motion;
    const vkgsplat::float2* guide_depth_ptr = nullptr;
    const vkgsplat::cuda::GaussianReconstructionSampleIds* guide_ids_ptr = nullptr;
    const vkgsplat::float3* guide_normal_ptr = nullptr;
    const vkgsplat::float2* guide_motion_ptr = nullptr;

    if (depth_gate) {
        guide_depth_confidence = dict_tensor(guides, "depth_confidence");
        check_float_cuda_image(guide_depth_confidence, "guides.depth_confidence", height, width, 2);
        check_same_device(guide_depth_confidence, sample_position, "guides.depth_confidence");
        guide_depth_ptr = reinterpret_cast<const vkgsplat::float2*>(guide_depth_confidence.data_ptr<float>());
    }
    if (primitive_gate) {
        guide_ids = dict_tensor(guides, "ids");
        check_int_cuda_image(guide_ids, "guides.ids", height, width, 4);
        check_same_device(guide_ids, sample_position, "guides.ids");
        guide_ids_ptr = reinterpret_cast<const vkgsplat::cuda::GaussianReconstructionSampleIds*>(
            guide_ids.data_ptr<int>());
    }
    if (normal_gate) {
        guide_normal = dict_tensor(guides, "normal");
        check_float_cuda_image(guide_normal, "guides.normal", height, width, 3);
        check_same_device(guide_normal, sample_position, "guides.normal");
        guide_normal_ptr = reinterpret_cast<const vkgsplat::float3*>(guide_normal.data_ptr<float>());
    }
    if (motion_gate) {
        guide_motion = dict_tensor(guides, "motion_px");
        check_float_cuda_image(guide_motion, "guides.motion_px", height, width, 2);
        check_same_device(guide_motion, sample_position, "guides.motion_px");
        guide_motion_ptr = reinterpret_cast<const vkgsplat::float2*>(guide_motion.data_ptr<float>());
    }

    const c10::cuda::CUDAGuard device_guard(sample_position.device());
    const auto sample_view = sample_view_from_dict(samples);
    const auto float_opts = sample_position.options().dtype(torch::kFloat32);
    const auto int_opts = sample_position.options().dtype(torch::kInt32);
    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto counters = torch::empty({ 13 }, int_opts);

    const vkgsplat::cuda::GaussianTileGatedWeightedResolveLaunch launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(tile_width),
        static_cast<std::uint32_t>(tile_height),
        static_cast<std::uint32_t>(tile_count),
        static_cast<std::uint32_t>(sample_position.size(0)),
        static_cast<std::uint32_t>(radius_px),
        gate_mask,
        static_cast<float>(sigma_px),
        static_cast<float>(min_confidence),
        static_cast<float>(depth_epsilon),
        static_cast<float>(normal_dot_min),
        static_cast<float>(motion_epsilon_px),
    };
    const vkgsplat::cuda::GaussianTileSampleResolveInputs inputs{
        reinterpret_cast<const std::uint32_t*>(tile_offsets.data_ptr<int>()),
        reinterpret_cast<const std::uint32_t*>(tile_sample_indices.data_ptr<int>()),
    };
    const vkgsplat::cuda::GaussianTileResolveGuides guide_view{
        guide_depth_ptr,
        guide_ids_ptr,
        guide_normal_ptr,
        guide_motion_ptr,
    };
    const vkgsplat::cuda::GaussianTileSampleResolveOutputs outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_resolve_gaussian_sample_tiles_weighted_gated(
        launch,
        sample_view,
        inputs,
        guide_view,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianTileGatedWeightedResolveCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["counters"] = counters;
    return result;
}

py::dict finalize_gaussian_state_tensor(py::dict state, std::int64_t gaussian_count) {
    const auto position = dict_tensor(state, "position");
    check_float_cuda_matrix(position, "state.position", 3);
    const std::int64_t capacity = position.size(0);
    if (gaussian_count < 0) {
        gaussian_count = capacity;
    }
    TORCH_CHECK(gaussian_count <= capacity, "gaussian_count exceeds state capacity");
    check_uint32_range(gaussian_count, "gaussian_count");

    const c10::cuda::CUDAGuard device_guard(position.device());
    const auto state_view = state_view_from_dict(state);
    vkgsplat::cuda::launch_finalize_gaussian_reconstruction_state(
        state_view,
        static_cast<std::uint32_t>(gaussian_count),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
    return state;
}

py::dict project_gaussian_state_features_tensor(const py::dict& state,
                                                std::int64_t height,
                                                std::int64_t width,
                                                std::int64_t gaussian_count,
                                                double min_confidence,
                                                double motion_alpha) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(height <= max_uint32 / width, "height * width exceeds uint32 capacity");

    const auto position = dict_tensor(state, "position");
    check_float_cuda_matrix(position, "state.position", 3);
    const std::int64_t capacity = position.size(0);
    if (gaussian_count < 0) {
        gaussian_count = capacity;
    }
    TORCH_CHECK(gaussian_count <= capacity, "gaussian_count exceeds state capacity");
    check_uint32_range(gaussian_count, "gaussian_count");

    const c10::cuda::CUDAGuard device_guard(position.device());
    const auto state_view = state_view_from_dict(state);
    const auto float_opts = position.options().dtype(torch::kFloat32);
    const auto int_opts = position.options().dtype(torch::kInt32);
    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto counters = torch::empty({ 4 }, int_opts);

    const vkgsplat::cuda::GaussianFeatureProjectionLaunch launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(gaussian_count),
        static_cast<float>(min_confidence),
        static_cast<float>(motion_alpha),
    };
    const vkgsplat::cuda::GaussianFeatureProjectionOutputs outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_project_gaussian_state_features(
        launch,
        state_view,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianFeatureProjectionCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
    result["counters"] = counters;
    return result;
}

py::dict project_gaussian_state_features_weighted_tensor(const py::dict& state,
                                                         std::int64_t height,
                                                         std::int64_t width,
                                                         std::int64_t gaussian_count,
                                                         std::int64_t radius_px,
                                                         double sigma_px,
                                                         double min_confidence,
                                                         double motion_alpha) {
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(radius_px >= 0 && radius_px <= 8, "radius_px must be in [0, 8]");
    TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");
    check_uint32_range(height, "height");
    check_uint32_range(width, "width");
    check_uint32_range(radius_px, "radius_px");
    const auto max_uint32 = static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max());
    TORCH_CHECK(height <= max_uint32 / width, "height * width exceeds uint32 capacity");

    const auto position = dict_tensor(state, "position");
    check_float_cuda_matrix(position, "state.position", 3);
    const std::int64_t capacity = position.size(0);
    if (gaussian_count < 0) {
        gaussian_count = capacity;
    }
    TORCH_CHECK(gaussian_count <= capacity, "gaussian_count exceeds state capacity");
    check_uint32_range(gaussian_count, "gaussian_count");

    const c10::cuda::CUDAGuard device_guard(position.device());
    const auto state_view = state_view_from_dict(state);
    const auto float_opts = position.options().dtype(torch::kFloat32);
    const auto int_opts = position.options().dtype(torch::kInt32);
    auto radiance = torch::empty({ height, width, 4 }, float_opts);
    auto depth_confidence = torch::empty({ height, width, 2 }, float_opts);
    auto ids = torch::empty({ height, width, 4 }, int_opts);
    auto counters = torch::empty({ 7 }, int_opts);

    const vkgsplat::cuda::GaussianStateWeightedProjectionLaunch launch{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(gaussian_count),
        static_cast<std::uint32_t>(radius_px),
        static_cast<float>(sigma_px),
        static_cast<float>(min_confidence),
        static_cast<float>(motion_alpha),
    };
    const vkgsplat::cuda::GaussianFeatureProjectionOutputs outputs{
        reinterpret_cast<vkgsplat::float4*>(radiance.data_ptr<float>()),
        reinterpret_cast<vkgsplat::float2*>(depth_confidence.data_ptr<float>()),
        reinterpret_cast<vkgsplat::cuda::GaussianReconstructionSampleIds*>(ids.data_ptr<int>()),
    };
    vkgsplat::cuda::launch_project_gaussian_state_features_weighted(
        launch,
        state_view,
        outputs,
        reinterpret_cast<vkgsplat::cuda::GaussianStateWeightedProjectionCounters*>(counters.data_ptr<int>()),
        reinterpret_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));

    py::dict result;
    result["radiance"] = radiance;
    result["depth_confidence"] = depth_confidence;
    result["ids"] = ids;
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

    m.def("extract_seed_frame_samples",
          &extract_seed_frame_samples_tensor,
          py::arg("radiance"),
          py::arg("depth_confidence"),
          py::arg("ids"),
          py::arg("max_samples"),
          py::arg("world_position") = py::none(),
          py::arg("normal") = py::none(),
          py::arg("motion_px") = py::none(),
          py::arg("min_confidence") = 0.0,
          "Extract native Vulkan/Wicked seed-buffer tensors into vkGSplat SoA reconstruction tensors.");

    m.def("sample_count_info",
          &sample_count_info_tensor,
          py::arg("sample_count"),
          py::arg("max_samples"),
          "Read a device-side vkGSplat sample-count tensor as [raw, clamped, overflow, available].");

    m.def("create_gaussian_state",
          &create_gaussian_state_tensor,
          py::arg("like"),
          py::arg("gaussian_capacity"),
          py::arg("clear") = true,
          "Allocate vkGSplat persistent Gaussian state tensors on like.device.");
    m.def("clear_gaussian_state",
          &clear_gaussian_state_tensor,
          py::arg("state"),
          "Clear vkGSplat persistent Gaussian state tensors in place.");
    m.def("build_sample_tile_bins",
          &build_gaussian_sample_tile_bins_tensor,
          py::arg("samples"),
          py::arg("sample_count"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          "Build per-sample tile IDs and per-tile counts from compact vkGSplat sample tensors.");

    m.def("build_sample_tile_offsets",
          &build_gaussian_sample_tile_offsets_tensor,
          py::arg("tile_counts"),
          "Build exclusive tile offsets from per-tile sample counts on the current CUDA stream.");

    m.def("compact_sample_tile_bins",
          &compact_gaussian_sample_tile_bins_tensor,
          py::arg("sample_tile"),
          py::arg("tile_offsets"),
          py::arg("sample_count"),
          "Compact per-sample tile IDs into tile-local sample-index spans using exclusive tile offsets.");

    m.def("build_sample_tile_spans",
          &build_gaussian_sample_tile_spans_tensor,
          py::arg("samples"),
          py::arg("sample_count"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          "Build per-sample tile IDs, exclusive offsets, and compact tile-local sample-index spans.");

    m.def("resolve_sample_frame_weighted",
          &resolve_gaussian_sample_frame_weighted_tensor,
          py::arg("samples"),
          py::arg("sample_count"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          py::arg("radius_px") = 1,
          py::arg("sigma_px") = 0.75,
          py::arg("min_confidence") = 0.0,
          "Build tile spans and weighted-resolve compact vkGSplat sample tensors into feature planes.");

    m.def("resolve_sample_frame_weighted_gated",
          &resolve_gaussian_sample_frame_weighted_gated_tensor,
          py::arg("samples"),
          py::arg("sample_count"),
          py::arg("guides"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          py::arg("radius_px") = 1,
          py::arg("sigma_px") = 0.75,
          py::arg("min_confidence") = 0.0,
          py::arg("gate_flags") = vkgsplat::cuda::GAUSSIAN_TILE_GATE_PRIMITIVE_ID |
                                  vkgsplat::cuda::GAUSSIAN_TILE_GATE_DEPTH,
          py::arg("depth_epsilon") = 1.0e-3,
          py::arg("normal_dot_min") = 0.5,
          py::arg("motion_epsilon_px") = 1.0,
          "Build tile spans and gated weighted-resolve compact vkGSplat sample tensors into feature planes.");


    m.def("resolve_sample_tiles",
          &resolve_gaussian_sample_tiles_tensor,
          py::arg("samples"),
          py::arg("tile_offsets"),
          py::arg("tile_sample_indices"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          py::arg("min_confidence") = 0.0,
          "Resolve compact tile-local sample spans into vkGSplat tensor feature planes.");

    m.def("resolve_sample_tiles_weighted",
          &resolve_gaussian_sample_tiles_weighted_tensor,
          py::arg("samples"),
          py::arg("tile_offsets"),
          py::arg("tile_sample_indices"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          py::arg("radius_px") = 1,
          py::arg("sigma_px") = 0.75,
          py::arg("min_confidence") = 0.0,
          "Weighted Gaussian resolve over compact tile-local sample spans.");

    m.def("resolve_sample_tiles_weighted_gated",
          &resolve_gaussian_sample_tiles_weighted_gated_tensor,
          py::arg("samples"),
          py::arg("tile_offsets"),
          py::arg("tile_sample_indices"),
          py::arg("guides"),
          py::arg("height"),
          py::arg("width"),
          py::arg("tile_height") = 16,
          py::arg("tile_width") = 16,
          py::arg("radius_px") = 1,
          py::arg("sigma_px") = 0.75,
          py::arg("min_confidence") = 0.0,
          py::arg("gate_flags") = vkgsplat::cuda::GAUSSIAN_TILE_GATE_PRIMITIVE_ID |
                                  vkgsplat::cuda::GAUSSIAN_TILE_GATE_DEPTH,
          py::arg("depth_epsilon") = 1.0e-3,
          py::arg("normal_dot_min") = 0.5,
          py::arg("motion_epsilon_px") = 1.0,
          "Weighted Gaussian resolve gated by dense depth/id/normal/motion guide tensors.");
    m.def("accumulate_gaussian_state",
          &accumulate_gaussian_state_from_samples_tensor,
          py::arg("samples"),
          py::arg("sample_count"),
          py::arg("state"),
          py::arg("min_confidence") = 0.0,
          "Accumulate compact vkGSplat sample tensors into persistent Gaussian state using a device-side count tensor.");
    m.def("finalize_gaussian_state",
          &finalize_gaussian_state_tensor,
          py::arg("state"),
          py::arg("gaussian_count") = -1,
          "Normalize accumulated vkGSplat Gaussian state in place.");
    m.def("project_gaussian_state_features",
          &project_gaussian_state_features_tensor,
          py::arg("state"),
          py::arg("height"),
          py::arg("width"),
          py::arg("gaussian_count") = -1,
          py::arg("min_confidence") = 0.0,
          py::arg("motion_alpha") = 0.0,
          "Project finalized vkGSplat Gaussian state into tensor feature planes, optionally shifted by stored motion.");
    m.def("project_gaussian_state_features_weighted",
          &project_gaussian_state_features_weighted_tensor,
          py::arg("state"),
          py::arg("height"),
          py::arg("width"),
          py::arg("gaussian_count") = -1,
          py::arg("radius_px") = 1,
          py::arg("sigma_px") = 0.75,
          py::arg("min_confidence") = 0.0,
          py::arg("motion_alpha") = 0.0,
          "Weighted-splat finalized vkGSplat Gaussian state into tensor feature planes for temporal/SR resolving.");
}
