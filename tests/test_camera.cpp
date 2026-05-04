// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for the Camera class: identity-ish look_at + perspective.

#include <vkgsplat/camera.h>

#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    using namespace vkgsplat;

    Camera c;
    c.set_resolution(800, 600);
    c.set_perspective(1.0f, 800.0f / 600.0f, 0.1f, 100.0f);
    c.look_at({ 0.0f, 0.0f, 5.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

    if (c.width()  != 800) { std::fprintf(stderr, "width mismatch\n");  return 1; }
    if (c.height() != 600) { std::fprintf(stderr, "height mismatch\n"); return 1; }
    if (std::abs(c.fov_y() - 1.0f) > 1e-6f) { std::fprintf(stderr, "fov mismatch\n"); return 1; }

    // The view matrix should not be all zeros.
    bool any_nonzero = false;
    for (float v : c.view().m) if (v != 0.0f) { any_nonzero = true; break; }
    if (!any_nonzero) { std::fprintf(stderr, "view matrix all zero\n"); return 1; }

    return 0;
}
