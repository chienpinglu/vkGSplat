// SPDX-License-Identifier: Apache-2.0

#include "vksplat/metal/denoise.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vksplat::metal {
namespace {

constexpr const char* kDenoiseShader = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    uint height;
    uint count;
    uint spatial_radius;
    float history_weight;
    float depth_threshold;
    uint reject_primitive_mismatch;
    uint _pad;
};

static inline float luminance(float4 c) {
    return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
}

kernel void temporal_accumulate(
    device const float4* current_color [[buffer(0)]],
    device const float4* history_color [[buffer(1)]],
    device const uchar* valid_history [[buffer(2)]],
    device float4* temporal_color [[buffer(3)]],
    device float* temporal_variance [[buffer(4)]],
    device uchar* history_used [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.count) {
        return;
    }

    if (valid_history[gid] != 0) {
        const float4 current = current_color[gid];
        const float4 history = history_color[gid];
        temporal_color[gid] = mix(current, history, params.history_weight);
        const float dl = luminance(current) - luminance(history);
        temporal_variance[gid] = dl * dl;
        history_used[gid] = 1;
    } else {
        temporal_color[gid] = current_color[gid];
        temporal_variance[gid] = 0.0f;
        history_used[gid] = 0;
    }
}

kernel void spatial_filter(
    device const float* depth [[buffer(0)]],
    device const uint* primitive_id [[buffer(1)]],
    device const float4* temporal_color [[buffer(2)]],
    device const float* temporal_variance [[buffer(3)]],
    device float4* out_color [[buffer(4)]],
    device float* out_variance [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.count) {
        return;
    }

    const uint x = gid % params.width;
    const uint y = gid / params.width;
    const int radius = int(params.spatial_radius);

    float4 sum = float4(0.0f);
    float variance_sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int ny = int(y) + dy;
        if (ny < 0 || ny >= int(params.height)) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = int(x) + dx;
            if (nx < 0 || nx >= int(params.width)) {
                continue;
            }

            const uint neighbor = uint(ny) * params.width + uint(nx);
            if (!isfinite(depth[gid]) || !isfinite(depth[neighbor])) {
                continue;
            }
            if (fabs(depth[gid] - depth[neighbor]) > params.depth_threshold) {
                continue;
            }
            if (params.reject_primitive_mismatch != 0 &&
                primitive_id[gid] != primitive_id[neighbor]) {
                continue;
            }

            const float spatial_distance = float(dx * dx + dy * dy);
            const float variance_weight = 1.0f / (1.0f + temporal_variance[neighbor]);
            const float weight = variance_weight / (1.0f + spatial_distance);
            sum += temporal_color[neighbor] * weight;
            variance_sum += temporal_variance[neighbor] * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum > 0.0f) {
        const float inv_weight = 1.0f / weight_sum;
        out_color[gid] = sum * inv_weight;
        out_variance[gid] = variance_sum * inv_weight;
    } else {
        out_color[gid] = temporal_color[gid];
        out_variance[gid] = temporal_variance[gid];
    }
}
)";

struct Params {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t count = 0;
    std::uint32_t spatial_radius = 0;
    float history_weight = 0.0f;
    float depth_threshold = 0.0f;
    std::uint32_t reject_primitive_mismatch = 0;
    std::uint32_t pad = 0;
};

std::size_t pixel_count(std::uint32_t width, std::uint32_t height) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

void validate_frame(const DenoiseFrame& frame) {
    const std::size_t expected = pixel_count(frame.width, frame.height);
    if (frame.color.size() != expected ||
        frame.depth.size() != expected ||
        frame.primitive_id.size() != expected) {
        throw std::runtime_error("invalid Metal denoise frame sizes");
    }
}

void validate_history(const ReprojectionResult& history, std::uint32_t width, std::uint32_t height) {
    const std::size_t expected = pixel_count(width, height);
    if (history.width != width ||
        history.height != height ||
        history.color.size() != expected ||
        history.valid_history.size() != expected) {
        throw std::runtime_error("invalid Metal denoise history sizes");
    }
}

std::string error_message(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    NSString* description = [error localizedDescription];
    return description != nil ? std::string([description UTF8String]) : "unknown Metal error";
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, const void* data, std::size_t bytes) {
    if (bytes == 0) {
        return nil;
    }
    return [device newBufferWithBytes:data
                               length:bytes
                              options:MTLResourceStorageModeShared];
}

id<MTLBuffer> make_empty_buffer(id<MTLDevice> device, std::size_t bytes) {
    if (bytes == 0) {
        return nil;
    }
    return [device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> pipeline,
                 std::uint32_t count) {
    const NSUInteger thread_count =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error("Metal function not found");
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error("failed to create Metal compute pipeline: " + error_message(error));
    }
    return pipeline;
}

} // namespace

bool is_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

std::string device_name() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return {};
        }
        return std::string([[device name] UTF8String]);
    }
}

SvgfDenoiseResult denoise_svgf_baseline(
    const DenoiseFrame& current,
    const ReprojectionResult& reprojected_history,
    const SvgfDenoiseOptions& options)
{
    validate_frame(current);
    validate_history(reprojected_history, current.width, current.height);

    const std::size_t count = pixel_count(current.width, current.height);
    if (count == 0) {
        return {};
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("no Metal device is available");
        }

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kDenoiseShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            throw std::runtime_error("failed to compile Metal denoise library: " + error_message(error));
        }

        id<MTLComputePipelineState> temporal_pipeline =
            make_pipeline(device, library, @"temporal_accumulate");
        id<MTLComputePipelineState> spatial_pipeline =
            make_pipeline(device, library, @"spatial_filter");

        const Params params{
            current.width,
            current.height,
            static_cast<std::uint32_t>(count),
            options.spatial_radius,
            std::clamp(options.history_weight, 0.0f, 1.0f),
            options.depth_threshold,
            options.reject_primitive_mismatch ? 1u : 0u,
            0u,
        };

        id<MTLBuffer> current_color =
            make_buffer(device, current.color.data(), current.color.size() * sizeof(float4));
        id<MTLBuffer> history_color =
            make_buffer(device, reprojected_history.color.data(),
                        reprojected_history.color.size() * sizeof(float4));
        id<MTLBuffer> valid_history =
            make_buffer(device, reprojected_history.valid_history.data(),
                        reprojected_history.valid_history.size() * sizeof(std::uint8_t));
        id<MTLBuffer> depth =
            make_buffer(device, current.depth.data(), current.depth.size() * sizeof(float));
        id<MTLBuffer> primitive_id =
            make_buffer(device, current.primitive_id.data(),
                        current.primitive_id.size() * sizeof(std::uint32_t));
        id<MTLBuffer> params_buffer = make_buffer(device, &params, sizeof(params));

        id<MTLBuffer> temporal_color = make_empty_buffer(device, count * sizeof(float4));
        id<MTLBuffer> temporal_variance = make_empty_buffer(device, count * sizeof(float));
        id<MTLBuffer> history_used = make_empty_buffer(device, count * sizeof(std::uint8_t));
        id<MTLBuffer> out_color = make_empty_buffer(device, count * sizeof(float4));
        id<MTLBuffer> out_variance = make_empty_buffer(device, count * sizeof(float));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("failed to create Metal command queue");
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            throw std::runtime_error("failed to create Metal command buffer");
        }

        id<MTLComputeCommandEncoder> temporal_encoder = [command_buffer computeCommandEncoder];
        [temporal_encoder setComputePipelineState:temporal_pipeline];
        [temporal_encoder setBuffer:current_color offset:0 atIndex:0];
        [temporal_encoder setBuffer:history_color offset:0 atIndex:1];
        [temporal_encoder setBuffer:valid_history offset:0 atIndex:2];
        [temporal_encoder setBuffer:temporal_color offset:0 atIndex:3];
        [temporal_encoder setBuffer:temporal_variance offset:0 atIndex:4];
        [temporal_encoder setBuffer:history_used offset:0 atIndex:5];
        [temporal_encoder setBuffer:params_buffer offset:0 atIndex:6];
        dispatch_1d(temporal_encoder, temporal_pipeline, static_cast<std::uint32_t>(count));
        [temporal_encoder endEncoding];

        id<MTLComputeCommandEncoder> spatial_encoder = [command_buffer computeCommandEncoder];
        [spatial_encoder setComputePipelineState:spatial_pipeline];
        [spatial_encoder setBuffer:depth offset:0 atIndex:0];
        [spatial_encoder setBuffer:primitive_id offset:0 atIndex:1];
        [spatial_encoder setBuffer:temporal_color offset:0 atIndex:2];
        [spatial_encoder setBuffer:temporal_variance offset:0 atIndex:3];
        [spatial_encoder setBuffer:out_color offset:0 atIndex:4];
        [spatial_encoder setBuffer:out_variance offset:0 atIndex:5];
        [spatial_encoder setBuffer:params_buffer offset:0 atIndex:6];
        dispatch_1d(spatial_encoder, spatial_pipeline, static_cast<std::uint32_t>(count));
        [spatial_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            throw std::runtime_error("Metal denoise command failed: " +
                                     error_message([command_buffer error]));
        }

        SvgfDenoiseResult result;
        result.width = current.width;
        result.height = current.height;
        result.color.resize(count);
        result.variance.resize(count);
        result.history_used.resize(count);

        std::memcpy(result.color.data(), [out_color contents], count * sizeof(float4));
        std::memcpy(result.variance.data(), [out_variance contents], count * sizeof(float));
        std::memcpy(result.history_used.data(), [history_used contents],
                    count * sizeof(std::uint8_t));
        return result;
    }
}

} // namespace vksplat::metal
