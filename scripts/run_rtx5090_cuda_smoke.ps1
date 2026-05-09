param(
    [switch]$AllowSkip,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$BuildDir = if ($env:VKGSPLAT_CUDA_BUILD_DIR) { $env:VKGSPLAT_CUDA_BUILD_DIR } else { Join-Path $RootDir "build-cuda-rtx5090-win" }
$CudaArch = if ($env:VKGSPLAT_CUDA_ARCHITECTURES) { $env:VKGSPLAT_CUDA_ARCHITECTURES } else { "120" }

function Exit-SkipOrFail([string]$Message) {
    if ($AllowSkip) {
        Write-Host "vkGSplat RTX5090 CUDA smoke: SKIP: $Message"
        exit 77
    }
    Write-Error "vkGSplat RTX5090 CUDA smoke: FAIL: $Message"
    exit 1
}

function Exit-Fail([string]$Message) {
    Write-Error "vkGSplat RTX5090 CUDA smoke: FAIL: $Message"
    exit 1
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Exit-SkipOrFail "$Name was not found"
    }
}

Require-Command "nvidia-smi"
$GpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
if (-not $GpuInfo) {
    Exit-SkipOrFail "nvidia-smi did not report a visible NVIDIA GPU"
}
Write-Host "vkGSplat RTX5090 CUDA smoke: NVIDIA GPU(s):"
$GpuInfo | ForEach-Object { Write-Host "  $_" }

Require-Command "nvcc"
Require-Command "cmake"
Write-Host "vkGSplat RTX5090 CUDA smoke: nvcc:"
(& nvcc --version) | ForEach-Object { Write-Host "  $_" }

if (-not $NoBuild) {
    $Jobs = if ($env:VKGSPLAT_BUILD_JOBS) { $env:VKGSPLAT_BUILD_JOBS } else { [Environment]::ProcessorCount }

    & cmake -S $RootDir `
        -B $BuildDir `
        -G "Visual Studio 17 2022" -A x64 `
        -DCMAKE_BUILD_TYPE=Release `
        -DVKGSPLAT_ENABLE_CUDA=ON `
        -DVKGSPLAT_ENABLE_3DGS=ON `
        -DVKGSPLAT_ENABLE_VULKAN=OFF `
        -DVKGSPLAT_ENABLE_TORCH=OFF `
        -DVKGSPLAT_CUDA_ARCHITECTURES=$CudaArch
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "CMake configure failed"
    }

    & cmake --build $BuildDir --config Release --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "CUDA build failed"
    }
}

& ctest --test-dir $BuildDir `
    -C Release `
    -R "test_cuda_tile_renderer|test_cuda_rasterizer_smoke|test_cuda_gaussian_reconstruction|test_raytrace_seed|test_reprojection|test_denoise" `
    --output-on-failure
if ($LASTEXITCODE -ne 0) {
    Exit-Fail "CUDA smoke tests failed"
}

Write-Host "vkGSplat RTX5090 CUDA smoke: PASS"
