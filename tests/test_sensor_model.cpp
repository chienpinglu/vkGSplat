// SPDX-License-Identifier: Apache-2.0
//
// Sensor modeling contract: Physical AI realism comes from calibrated
// acquisition. Gaussian reconstruction can preserve it only when confidence
// gates allow; otherwise callers must fall back to an authoritative oracle.

#include <vkgsplat/sensor_model.h>

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

bool near(float a, float b, float eps = 1.0e-5f) {
    return std::abs(a - b) <= eps;
}

vkgsplat::float4 color(float r, float g, float b) {
    return { r, g, b, 1.0f };
}

} // namespace

int main() {
    using namespace vkgsplat;

    SensorImage linear;
    linear.width = 2;
    linear.height = 2;
    linear.color = {
        color(0.25f, 0.50f, 0.75f),
        color(0.50f, 0.50f, 0.50f),
        color(0.10f, 0.20f, 0.30f),
        color(1.50f, 0.00f, 0.25f),
    };

    CameraSensorModel ideal;
    ideal.exposure = 2.0f;
    ideal.quantization_bits = 8;
    const SensorImage observed = apply_camera_sensor_model(linear, ideal);
    if (observed.color.size() != linear.color.size() ||
        !near(observed.color[0].x, 0.5019608f) ||
        !near(observed.color[0].y, 1.0f) ||
        !near(observed.color[0].z, 1.0f) ||
        !near(observed.color[3].x, 1.0f)) {
        std::fprintf(stderr, "camera sensor exposure/quantization mismatch\n");
        return 1;
    }

    CameraSensorModel noisy;
    noisy.exposure = 1.0f;
    noisy.shot_noise_scale = 0.02f;
    noisy.read_noise_stddev = 0.01f;
    noisy.noise_seed = 1234u;
    const SensorImage noisy_a = apply_camera_sensor_model(linear, noisy);
    const SensorImage noisy_b = apply_camera_sensor_model(linear, noisy);
    for (std::size_t i = 0; i < noisy_a.color.size(); ++i) {
        if (!near(noisy_a.color[i].x, noisy_b.color[i].x) ||
            !near(noisy_a.color[i].y, noisy_b.color[i].y) ||
            !near(noisy_a.color[i].z, noisy_b.color[i].z)) {
            std::fprintf(stderr, "camera sensor noise is not deterministic\n");
            return 1;
        }
    }

    RollingShutterTiming timing;
    timing.frame_start_time = 10.0f;
    timing.exposure_duration = 0.002f;
    timing.readout_duration = 0.008f;
    if (!near(row_exposure_mid_time(timing, 0, 4), 10.002f) ||
        !near(row_exposure_mid_time(timing, 3, 4), 10.008f)) {
        std::fprintf(stderr, "rolling shutter row timing mismatch\n");
        return 1;
    }

    SensorPreservationInputs good;
    good.acquisition_confidence = 0.95f;
    good.gaussian_confidence = 0.90f;
    good.disocclusion = 0.05f;
    good.model_residual = 0.02f;
    good.sensor_noise_variance = 0.01f;
    const SensorPreservationDecision keep = assess_sensor_preservation(good);
    if (!keep.use_gaussian_reconstruction || keep.require_oracle || !(keep.confidence > 0.75f)) {
        std::fprintf(stderr, "good sensor observation was not preserved\n");
        return 1;
    }

    SensorPreservationInputs unsafe = good;
    unsafe.safety_critical = true;
    const SensorPreservationDecision fallback = assess_sensor_preservation(unsafe);
    if (fallback.use_gaussian_reconstruction || !fallback.require_oracle) {
        std::fprintf(stderr, "safety critical observation did not require oracle fallback\n");
        return 1;
    }

    SensorPreservationInputs disoccluded = good;
    disoccluded.disocclusion = 0.75f;
    const SensorPreservationDecision reject = assess_sensor_preservation(disoccluded);
    if (reject.use_gaussian_reconstruction || !reject.require_oracle) {
        std::fprintf(stderr, "disoccluded observation incorrectly used Gaussian reconstruction\n");
        return 1;
    }

    return 0;
}
