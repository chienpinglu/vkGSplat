// SPDX-License-Identifier: Apache-2.0
//
// Fullscreen triangle presentation vertex shader. Emits NDC positions
// directly from gl_VertexIndex so no vertex buffer is required. The
// fragment shader samples a texture written by the CUDA backend (via
// VK_KHR_external_memory) and blits it to the swapchain image.
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
