#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WICKED_ROOT="${WICKED_ROOT:-$ROOT_DIR/third_party/WickedEngine}"
WICKED_BUILD_DIR="${WICKED_BUILD_DIR:-$WICKED_ROOT/build-vkgsplat-nvidia}"
LOG_DIR="${VKGSPLAT_WICKED_LOG_DIR:-$ROOT_DIR/build/wicked-nvidia}"
ALLOW_SKIP=0
DO_BUILD=1

usage() {
    cat <<'USAGE'
Usage: scripts/run_wicked_nvidia_smoke.sh [--allow-skip] [--no-build]

Runs the Wicked Engine Cornell capture smoke test on a real NVIDIA Vulkan GPU.
The test requires Wicked to select an NVIDIA adapter and report SPIR-V, mesh
shader support, Vulkan ray tracing support, and a ready Cornell capture
contract. With --allow-skip, missing local hardware/dependencies exit 77 for
CTest skip behavior.

Environment:
  WICKED_ROOT             Wicked checkout path (default: third_party/WickedEngine)
  WICKED_BUILD_DIR        Wicked build dir (default: third_party/WickedEngine/build-vkgsplat-nvidia)
  VKGSPLAT_WICKED_LOG_DIR Output log dir (default: build/wicked-nvidia)
  VKGSPLAT_BUILD_JOBS     Parallel build jobs
USAGE
}

skip_or_fail() {
    local message="$1"
    if [[ "$ALLOW_SKIP" == "1" ]]; then
        echo "vkGSplat Wicked NVIDIA smoke: SKIP: $message"
        exit 77
    fi
    echo "vkGSplat Wicked NVIDIA smoke: FAIL: $message" >&2
    exit 1
}

fail() {
    echo "vkGSplat Wicked NVIDIA smoke: FAIL: $1" >&2
    exit 1
}

require_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        skip_or_fail "$name was not found"
    fi
}

check_log() {
    local pattern="$1"
    local message="$2"
    if ! grep -Eq "$pattern" "$LOG_FILE"; then
        echo "vkGSplat Wicked NVIDIA smoke: log tail:" >&2
        tail -80 "$LOG_FILE" >&2 || true
        fail "$message"
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
GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
if [[ -z "$GPU_INFO" ]]; then
    skip_or_fail "nvidia-smi did not report a visible NVIDIA GPU"
fi
echo "vkGSplat Wicked NVIDIA smoke: NVIDIA GPU(s):"
echo "$GPU_INFO" | sed 's/^/  /'

require_command vulkaninfo
VULKAN_SUMMARY="$(vulkaninfo --summary 2>&1 || true)"
if ! grep -Eiq 'NVIDIA|GeForce|RTX|Quadro|Tesla|RTX|L[0-9]{2}|A[0-9]{2,}|H[0-9]{3}|B[0-9]{3}' <<<"$VULKAN_SUMMARY"; then
    echo "$VULKAN_SUMMARY" | tail -120 >&2
    skip_or_fail "vulkaninfo did not expose an NVIDIA Vulkan physical device"
fi

if [[ ! -d "$WICKED_ROOT/WickedEngine" ]]; then
    skip_or_fail "Wicked Engine checkout not found at $WICKED_ROOT"
fi
if [[ ! -f "$WICKED_ROOT/Samples/vkSplatCapture/CMakeLists.txt" ]]; then
    skip_or_fail "Wicked vkGSplat/vkSplat capture sample is missing in $WICKED_ROOT/Samples"
fi

if [[ "$DO_BUILD" == "1" ]]; then
    require_command cmake
    JOBS="${VKGSPLAT_BUILD_JOBS:-}"
    if [[ -z "$JOBS" ]]; then
        JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
    fi

    cmake -S "$WICKED_ROOT" \
        -B "$WICKED_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DWICKED_EDITOR=OFF \
        -DWICKED_TESTS=OFF \
        -DWICKED_IMGUI_EXAMPLE=OFF \
        -DWICKED_LINUX_TEMPLATE=ON \
        -DWICKED_VKSPLAT_CAPTURE=ON \
        -DWICKED_ENABLE_SYMLINKS=OFF || fail "Wicked CMake configure failed"

    cmake --build "$WICKED_BUILD_DIR" --target vkSplatCapture --parallel "$JOBS" ||
        fail "Wicked vkSplatCapture build failed"
fi

CAPTURE_BIN="$WICKED_BUILD_DIR/Samples/vkSplatCapture/vkSplatCapture"
if [[ ! -x "$CAPTURE_BIN" && -x "$CAPTURE_BIN.exe" ]]; then
    CAPTURE_BIN="$CAPTURE_BIN.exe"
fi
if [[ ! -x "$CAPTURE_BIN" ]]; then
    skip_or_fail "capture binary not found at $CAPTURE_BIN"
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/wicked_nvidia_smoke.log"

pushd "$WICKED_ROOT/WickedEngine" >/dev/null || fail "could not enter WickedEngine runtime directory"
set +e
LD_LIBRARY_PATH="$(dirname "$CAPTURE_BIN"):$PWD:${LD_LIBRARY_PATH:-}" \
    "$CAPTURE_BIN" --scene >"$LOG_FILE" 2>&1
STATUS=$?
set -e
popd >/dev/null || true

if [[ "$STATUS" -ne 0 ]]; then
    tail -120 "$LOG_FILE" >&2 || true
    fail "capture harness exited with status $STATUS"
fi

check_log 'initialized=yes' "Wicked did not initialize"
check_log 'adapter=.*(NVIDIA|GeForce|RTX|Quadro|Tesla|L[0-9]{2}|A[0-9]{2,}|H[0-9]{3}|B[0-9]{3})' \
    "Wicked did not select an NVIDIA adapter"
check_log 'shader_format=spirv' "Wicked Vulkan backend did not report SPIR-V shaders"
check_log 'capability\.mesh_shader=yes' "NVIDIA Vulkan adapter did not expose mesh shader support through Wicked"
check_log 'capability\.raytracing=yes' "NVIDIA Vulkan adapter did not expose ray tracing support through Wicked"
check_log 'scene\.loaded=yes' "Cornell scene metadata did not load"
check_log 'render_path=RenderPath3D_PathTracing' "Wicked path tracing render path was not selected"
check_log 'capture\.ready=yes' "Wicked Cornell capture contract is not ready"
check_log 'capture\.mode=raytracing-ready' "Wicked Cornell capture did not reach raytracing-ready mode"

echo "vkGSplat Wicked NVIDIA smoke: PASS"
echo "vkGSplat Wicked NVIDIA smoke: log=$LOG_FILE"
