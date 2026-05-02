// SPDX-License-Identifier: Apache-2.0
//
// Presentation fragment shader. Samples the CUDA-written render target
// and emits it to the swapchain. v1 is a straight passthrough; future
// work plugs DLSS-style neural reconstruction into this stage so the
// CUDA seed remains low-resolution and noisy (Section 3.4 of the
// position paper).
#version 450

layout(set = 0, binding = 0) uniform sampler2D u_render_target;

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = texture(u_render_target, v_uv);
}
