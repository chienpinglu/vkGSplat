// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for the Scene container — does not require I/O.

#include <vksplat/scene.h>

#include <cstdio>

int main() {
    using namespace vksplat;

    Scene s;
    if (!s.empty()) { std::fprintf(stderr, "fresh scene not empty\n"); return 1; }

    s.resize(10);
    if (s.size() != 10) { std::fprintf(stderr, "resize failed\n"); return 1; }

    s.set_name("synthetic");
    if (s.name() != "synthetic") { std::fprintf(stderr, "name mismatch\n"); return 1; }

    return 0;
}
