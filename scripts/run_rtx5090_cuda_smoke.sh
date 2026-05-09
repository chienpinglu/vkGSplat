#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${VKGSPLAT_CUDA_BUILD_DIR:-$ROOT_DIR/build-cuda-rtx5090}"
CUDA_ARCH="${VKGSPLAT_CUDA_ARCHITECTURES:-120}"
ALLOW_SKIP=0
DO_BUILD=1

usage() {
    cat <<'USAGE'
Usage: scripts/run_rtx5090_cuda_smoke.sh [--allow-skip] [--no-build]

Configures, builds, and runs the CUDA vkGSplat smoke tests for an RTX 5090 /
Blackwell workstation. With --allow-skip, missing local hardware/dependencies
exit 77 for CTest skip behavior.

Environment:
  VKGSPLAT_CUDA_BUILD_DIR      Build directory (default: build-cuda-rtx5090)
  VKGSPLAT_CUDA_ARCHITECTURES  CUDA arch list (default: 120)
  VKGSPLAT_BUILD_JOBS          Parallel build jobs
USAGE
}

skip_or_fail() {
    local message="$1"
    if [[ "$ALLOW_SKIP" == "1" ]]; then
        echo "vkGSplat RTX5090 CUDA smoke: SKIP: $message"
        exit 77
    fi
    echo "vkGSplat RTX5090 CUDA smoke: FAIL: $message" >&2
    exit 1
}

fail() {
    echo "vkGSplat RTX5090 CUDA smoke: FAIL: $1" >&2
    exit 1
}

require_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        skip_or_fail "$name was not found"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-skip)
            ALLOW_SKIP=1
            ;;
        --no-build)
            DO_BUILD=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown argument: $1"
            ;;
    esac
    shift
done

require_command nvidia-smi
GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)"
if [[ -z "$GPU_INFO" ]]; then
    skip_or_fail "nvidia-smi did not report a visible NVIDIA GPU"
fi
echo "vkGSplat RTX5090 CUDA smoke: NVIDIA GPU(s):"
echo "$GPU_INFO" | sed 's/^/  /'

require_command nvcc
require_command cmake
echo "vkGSplat RTX5090 CUDA smoke: nvcc:"
nvcc --version | sed 's/^/  /'

if [[ "$DO_BUILD" == "1" ]]; then
    JOBS="${VKGSPLAT_BUILD_JOBS:-}"
    if [[ -z "$JOBS" ]]; then
        JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
    fi

    cmake -S "$ROOT_DIR" \
        -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DVKGSPLAT_ENABLE_CUDA=ON \
        -DVKGSPLAT_ENABLE_3DGS=ON \
        -DVKGSPLAT_ENABLE_VULKAN=OFF \
        -DVKGSPLAT_ENABLE_TORCH=OFF \
        -DVKGSPLAT_CUDA_ARCHITECTURES="$CUDA_ARCH" ||
        fail "CMake configure failed"

    cmake --build "$BUILD_DIR" --parallel "$JOBS" ||
        fail "CUDA build failed"
fi

ctest --test-dir "$BUILD_DIR" \
    -R 'test_cuda_tile_renderer|test_cuda_rasterizer_smoke|test_cuda_gaussian_reconstruction|test_raytrace_seed|test_reprojection|test_denoise' \
    --output-on-failure ||
    fail "CUDA smoke tests failed"

echo "vkGSplat RTX5090 CUDA smoke: PASS"
