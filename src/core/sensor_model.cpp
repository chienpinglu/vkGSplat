// SPDX-License-Identifier: Apache-2.0

#include "vkgsplat/sensor_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vkgsplat {
namespace {

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float safe_unit_score(float value, float limit) {
    if (limit <= 0.0f) {
        return value <= 0.0f ? 1.0f : 0.0f;
    }
    return clamp01(1.0f - value / limit);
}

std::uint32_t hash_u32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float uniform01(std::uint32_t seed, std::uint32_t pixel, std::uint32_t channel, std::uint32_t salt) {
    const std::uint32_t h = hash_u32(seed ^ hash_u32(pixel * 16777619u) ^
                                    hash_u32(channel * 2166136261u) ^ hash_u32(salt));
    return static_cast<float>(h & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

float normalish(std::uint32_t seed, std::uint32_t pixel, std::uint32_t channel) {
    // Sum of uniforms gives a deterministic, bounded normal approximation.
    const float u0 = uniform01(seed, pixel, channel, 0u);
    const float u1 = uniform01(seed, pixel, channel, 1u);
    const float u2 = uniform01(seed, pixel, channel, 2u);
    const float u3 = uniform01(seed, pixel, channel, 3u);
    return (u0 + u1 + u2 + u3 - 2.0f) * 0.86602540378f;
}

float quantize(float value, std::uint32_t bits) {
    if (bits == 0u) {
        return value;
    }
    const std::uint32_t clamped_bits = std::min(bits, 24u);
    const float levels = static_cast<float>((1u << clamped_bits) - 1u);
    return std::round(value * levels) / levels;
}

float apply_channel(float value,
                    const CameraSensorModel& model,
                    float vignette,
                    std::uint32_t pixel,
                    std::uint32_t channel) {
    value = std::max(0.0f, value * std::max(0.0f, model.exposure) * vignette);
    const float variance =
        std::max(0.0f, model.shot_noise_scale) * value +
        std::max(0.0f, model.read_noise_stddev) * std::max(0.0f, model.read_noise_stddev);
    if (variance > 0.0f) {
        value += std::sqrt(variance) * normalish(model.noise_seed, pixel, channel);
    }

    const float range = model.white_level - model.black_level;
    if (range <= 0.0f) {
        throw std::runtime_error("camera sensor model requires white_level > black_level");
    }
    value = clamp01((value - model.black_level) / range);
    if (model.gamma > 0.0f && std::abs(model.gamma - 1.0f) > 1.0e-6f) {
        value = std::pow(value, 1.0f / model.gamma);
    }
    return quantize(clamp01(value), model.quantization_bits);
}

float vignette_for_pixel(const CameraSensorModel& model,
                         std::uint32_t x,
                         std::uint32_t y,
                         std::uint32_t width,
                         std::uint32_t height) {
    const float strength = std::clamp(model.vignetting_strength, 0.0f, 1.0f);
    if (strength <= 0.0f || width == 0u || height == 0u) {
        return 1.0f;
    }
    const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f;
    const float ny = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f - 1.0f;
    const float r2 = std::min(1.0f, nx * nx + ny * ny);
    return clamp01(1.0f - strength * r2);
}

} // namespace

float row_exposure_mid_time(
    const RollingShutterTiming& timing,
    std::uint32_t row,
    std::uint32_t height) {
    if (height == 0u || row >= height) {
        throw std::runtime_error("invalid rolling-shutter row");
    }
    const float row_center = (static_cast<float>(row) + 0.5f) / static_cast<float>(height);
    return timing.frame_start_time +
           std::max(0.0f, timing.readout_duration) * row_center +
           std::max(0.0f, timing.exposure_duration) * 0.5f;
}

SensorImage apply_camera_sensor_model(
    const SensorImage& linear_input,
    const CameraSensorModel& model) {
    const std::size_t expected =
        static_cast<std::size_t>(linear_input.width) * static_cast<std::size_t>(linear_input.height);
    if (linear_input.color.size() != expected) {
        throw std::runtime_error("sensor image color size does not match dimensions");
    }

    SensorImage output;
    output.width = linear_input.width;
    output.height = linear_input.height;
    output.color.resize(expected);

    for (std::uint32_t y = 0; y < linear_input.height; ++y) {
        for (std::uint32_t x = 0; x < linear_input.width; ++x) {
            const std::uint32_t pixel = y * linear_input.width + x;
            const float vignette = vignette_for_pixel(model, x, y, linear_input.width, linear_input.height);
            const float4 in = linear_input.color[pixel];
            output.color[pixel] = {
                apply_channel(in.x, model, vignette, pixel, 0u),
                apply_channel(in.y, model, vignette, pixel, 1u),
                apply_channel(in.z, model, vignette, pixel, 2u),
                clamp01(in.w),
            };
        }
    }

    return output;
}

SensorPreservationDecision assess_sensor_preservation(
    const SensorPreservationInputs& inputs,
    const SensorPreservationPolicy& policy) {
    const float acquisition_score = clamp01(inputs.acquisition_confidence);
    const float gaussian_score = clamp01(inputs.gaussian_confidence);
    const float disocclusion_score = safe_unit_score(std::max(0.0f, inputs.disocclusion),
                                                     policy.max_disocclusion);
    const float residual_score = safe_unit_score(std::max(0.0f, inputs.model_residual),
                                                 policy.max_model_residual);
    const float noise_score = safe_unit_score(std::max(0.0f, inputs.sensor_noise_variance),
                                              policy.max_sensor_noise_variance);

    SensorPreservationDecision decision;
    decision.confidence =
        std::min({ acquisition_score, gaussian_score, disocclusion_score, residual_score, noise_score });
    decision.require_oracle =
        (policy.require_oracle_for_safety_critical && inputs.safety_critical) ||
        acquisition_score < policy.min_acquisition_confidence ||
        gaussian_score < policy.min_gaussian_confidence ||
        inputs.disocclusion > policy.max_disocclusion ||
        inputs.model_residual > policy.max_model_residual ||
        inputs.sensor_noise_variance > policy.max_sensor_noise_variance;
    decision.use_gaussian_reconstruction = !decision.require_oracle;
    return decision;
}

} // namespace vkgsplat
