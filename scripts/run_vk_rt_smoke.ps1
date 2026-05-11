# Phase-4 substitute driver: invokes the in-tree `vk_rt_capture` probe and
# verifies its log against the same regex contract used by the Wicked smoke
# script. Keeps the original `run_wicked_nvidia_smoke.ps1` untouched so the
# author's Wicked-based gate can come back unchanged.

param(
    [switch]$AllowSkip,
    [string]$BuildDir = "",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = (Resolve-Path (Join-Path $ScriptDir "..")).Path
if (-not $BuildDir) {
    $BuildDir = if ($env:VKGSPLAT_VK_RT_BUILD_DIR) {
        $env:VKGSPLAT_VK_RT_BUILD_DIR
    } else {
        Join-Path $RootDir "build-vulkan-rtx5090-win"
    }
}
$LogDir  = Join-Path $RootDir "build/vk-rt-capture"
$LogFile = Join-Path $LogDir "vk_rt_smoke.log"

function Exit-SkipOrFail([string]$msg) {
    if ($AllowSkip) {
        Write-Host "vk_rt_capture smoke: SKIP: $msg"
        exit 77
    }
    Write-Error "vk_rt_capture smoke: FAIL: $msg"
    exit 1
}

function Exit-Fail([string]$msg) {
    Write-Error "vk_rt_capture smoke: FAIL: $msg"
    exit 1
}

# Best-effort NVIDIA presence check; if nvidia-smi is not visible in this
# shell (e.g. ctest legacy PowerShell host), let the probe itself decide via
# Vulkan adapter enumeration.
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpu = & $nvidiaSmi.Source --query-gpu=name --format=csv,noheader 2>$null
    if (-not $gpu) {
        Exit-SkipOrFail "nvidia-smi did not report any NVIDIA GPU"
    }
}

# Find the built executable. Visual Studio generator nests under $Config.
$candidates = @(
    Join-Path $BuildDir "apps/vk_rt_capture/$Config/vk_rt_capture.exe"
    Join-Path $BuildDir "apps/vk_rt_capture/vk_rt_capture.exe"
    Join-Path $BuildDir "apps/vk_rt_capture/vk_rt_capture"
)
$exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
    Exit-SkipOrFail "vk_rt_capture binary not found under $BuildDir (build first with -DVKGSPLAT_ENABLE_VK_RT_CAPTURE=ON)"
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
& $exe --scene 2>&1 | Tee-Object -FilePath $LogFile | Out-Host
if ($LASTEXITCODE -ne 0) {
    Exit-Fail "vk_rt_capture exited with code $LASTEXITCODE"
}

$text = Get-Content -Raw -Path $LogFile

function Assert-Log([string]$pattern, [string]$msg) {
    if ($text -notmatch $pattern) {
        Write-Error "vk_rt_capture smoke: log tail:"
        Get-Content -Path $LogFile -Tail 50 | Write-Error
        Exit-Fail $msg
    }
}

Assert-Log "vkSplatCapture: initialized=yes"                 "probe did not report initialized=yes"
Assert-Log "vkSplatCapture: adapter=.*NVIDIA"                 "adapter did not select NVIDIA"
Assert-Log "vkSplatCapture: shader_format=spirv"             "shader_format is not spirv"
Assert-Log "vkSplatCapture: capability.mesh_shader=yes"      "mesh shader capability missing"
Assert-Log "vkSplatCapture: capability.raytracing=yes"       "ray tracing capability missing"
Assert-Log "vkSplatCapture: scene.loaded=yes"                "scene was not declared loaded"
Assert-Log "vkSplatCapture: render_path=RenderPath3D_PathTracing" "render_path mismatch"
Assert-Log "vkSplatCapture: capture.ready=yes"               "capture was not declared ready"
Assert-Log "vkSplatCapture: capture.mode=raytracing-ready"   "capture mode is not raytracing-ready"

Write-Host "vk_rt_capture smoke: PASS"
