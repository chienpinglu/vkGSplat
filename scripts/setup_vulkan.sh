#!/usr/bin/env bash
# Bootstrap the Vulkan SDK into third_party/vulkan/sdk/.
# Idempotent: skips download if the SDK is already present.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/vulkan/sdk"
VK_VERSION="${VK_VERSION:-1.3.290.0}"

mkdir -p "$DEST"

uname_s="$(uname -s)"

case "$uname_s" in
    Linux)
        if command -v vulkaninfo >/dev/null 2>&1; then
            echo "[setup_vulkan] system Vulkan detected — skipping local SDK"
            vulkaninfo --summary | head -n 20 || true
            exit 0
        fi
        echo "[setup_vulkan] Install via LunarG apt repo:"
        echo "  wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc"
        echo "  sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \\"
        echo "    https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list"
        echo "  sudo apt update && sudo apt install vulkan-sdk"
        ;;
    Darwin)
        if command -v vulkaninfo >/dev/null 2>&1; then
            echo "[setup_vulkan] system Vulkan detected — skipping local SDK"
            exit 0
        fi
        echo "[setup_vulkan] Download macOS installer from:"
        echo "  https://vulkan.lunarg.com/sdk/home#mac"
        echo "Or: brew install --cask vulkan-sdk"
        echo "Note: NVIDIA CUDA is not supported on macOS — use Linux/Windows for the full build."
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo "[setup_vulkan] On Windows, download and run the installer from:"
        echo "  https://vulkan.lunarg.com/sdk/home#windows"
        ;;
    *)
        echo "[setup_vulkan] Unknown platform: $uname_s" >&2
        exit 1
        ;;
esac

echo "[setup_vulkan] Done. Verify with: vulkaninfo --summary"
