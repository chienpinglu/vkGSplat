// SPDX-License-Identifier: Apache-2.0
//
// Lightweight sensor-modeling contracts for Physical AI experiments. These
// utilities keep calibrated acquisition separate from Gaussian reconstruction:
// the sensor model produces or validates observations, while 3DGS may only
// preserve/reconstruct them when confidence gates pass.
#pragma once

#include "types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vkgsplat {

enum class SensorKind : std::uint32_t {
    RgbCamera = 0,
    DepthCamera,
    Lidar,
    Radar,
    EventCamera,
    Imu,
    Contact,
    Joint,
};

struct SensorImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float4> color;
};

struct CameraSensorModel {
    // Applied to linear radiance before clipping/noise/quantization.
    float exposure = 1.0f;
    float black_level = 0.0f;
    float white_level = 1.0f;

    // gamma <= 0 or gamma == 1 leaves values linear. Otherwise output is
    // pow(linear, 1/gamma), after clipping to [0, 1].
    float gamma = 1.0f;

    // Simple deterministic noise model for tests and synthetic-data probes.
    // Variance per channel is shot_noise_scale * signal + read_noise_stddev^2.
    float shot_noise_scale = 0.0f;
    float read_noise_stddev = 0.0f;
    std::uint32_t noise_seed = 0;

    // 0 disables quantization; otherwise values are quantized to this many bits.
    std::uint32_t quantization_bits = 0;

    // Radial vignetting in normalized image coordinates. 0 disables.
    float vignetting_strength = 0.0f;
};

struct RollingShutterTiming {
    float frame_start_time = 0.0f;
    float exposure_duration = 0.0f;
    float readout_duration = 0.0f;
};

[[nodiscard]] float row_exposure_mid_time(
    const RollingShutterTiming& timing,
    std::uint32_t row,
    std::uint32_t height);

[[nodiscard]] SensorImage apply_camera_sensor_model(
    const SensorImage& linear_input,
    const CameraSensorModel& model);

struct SensorPreservationInputs {
    SensorKind sensor = SensorKind::RgbCamera;
    float acquisition_confidence = 1.0f;
    float gaussian_confidence = 1.0f;
    float disocclusion = 0.0f;
    float model_residual = 0.0f;
    float sensor_noise_variance = 0.0f;
    bool safety_critical = false;
};

struct SensorPreservationPolicy {
    float min_acquisition_confidence = 0.5f;
    float min_gaussian_confidence = 0.5f;
    float max_disocclusion = 0.25f;
    float max_model_residual = 0.25f;
    float max_sensor_noise_variance = 0.25f;
    bool require_oracle_for_safety_critical = true;
};

struct SensorPreservationDecision {
    bool use_gaussian_reconstruction = false;
    bool require_oracle = true;
    float confidence = 0.0f;
};

[[nodiscard]] SensorPreservationDecision assess_sensor_preservation(
    const SensorPreservationInputs& inputs,
    const SensorPreservationPolicy& policy = {});

} // namespace vkgsplat
