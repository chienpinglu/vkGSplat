param(
    [switch]$AllowSkip,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$WickedRoot = if ($env:WICKED_ROOT) { $env:WICKED_ROOT } else { Join-Path $RootDir "third_party\WickedEngine" }
$WickedBuildDir = if ($env:WICKED_BUILD_DIR) { $env:WICKED_BUILD_DIR } else { Join-Path $WickedRoot "build-vkgsplat-nvidia" }
$LogDir = if ($env:VKGSPLAT_WICKED_LOG_DIR) { $env:VKGSPLAT_WICKED_LOG_DIR } else { Join-Path $RootDir "build\wicked-nvidia" }
$LogFile = Join-Path $LogDir "wicked_nvidia_smoke.log"

function Exit-SkipOrFail([string]$Message) {
    if ($AllowSkip) {
        Write-Host "vkGSplat Wicked NVIDIA smoke: SKIP: $Message"
        exit 77
    }
    Write-Error "vkGSplat Wicked NVIDIA smoke: FAIL: $Message"
    exit 1
}

function Exit-Fail([string]$Message) {
    Write-Error "vkGSplat Wicked NVIDIA smoke: FAIL: $Message"
    exit 1
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Exit-SkipOrFail "$Name was not found"
    }
}

function Assert-Log([string]$Pattern, [string]$Message) {
    $Text = Get-Content -Raw -Path $LogFile
    if ($Text -notmatch $Pattern) {
        Write-Error "vkGSplat Wicked NVIDIA smoke: log tail:"
        Get-Content -Path $LogFile -Tail 100 | Write-Error
        Exit-Fail $Message
    }
}

Require-Command "nvidia-smi"
$GpuInfo = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
if (-not $GpuInfo) {
    Exit-SkipOrFail "nvidia-smi did not report a visible NVIDIA GPU"
}
Write-Host "vkGSplat Wicked NVIDIA smoke: NVIDIA GPU(s):"
$GpuInfo | ForEach-Object { Write-Host "  $_" }

Require-Command "vulkaninfo"
$VulkanSummary = & vulkaninfo --summary 2>&1
if (($VulkanSummary -join "`n") -notmatch "NVIDIA|GeForce|RTX|Quadro|Tesla|L[0-9]{2}|A[0-9]{2,}|H[0-9]{3}|B[0-9]{3}") {
    $VulkanSummary | Select-Object -Last 120 | Write-Error
    Exit-SkipOrFail "vulkaninfo did not expose an NVIDIA Vulkan physical device"
}

if (-not (Test-Path (Join-Path $WickedRoot "WickedEngine"))) {
    Exit-SkipOrFail "Wicked Engine checkout not found at $WickedRoot"
}
if (-not (Test-Path (Join-Path $WickedRoot "Samples\vkSplatCapture\CMakeLists.txt"))) {
    Exit-SkipOrFail "Wicked vkGSplat/vkSplat capture sample is missing in $WickedRoot\Samples"
}

if (-not $NoBuild) {
    Require-Command "cmake"
    $Jobs = if ($env:VKGSPLAT_BUILD_JOBS) { $env:VKGSPLAT_BUILD_JOBS } else { [Environment]::ProcessorCount }

    & cmake -S $WickedRoot `
        -B $WickedBuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DWICKED_EDITOR=OFF `
        -DWICKED_TESTS=OFF `
        -DWICKED_IMGUI_EXAMPLE=OFF `
        -DWICKED_VKSPLAT_CAPTURE=ON `
        -DWICKED_ENABLE_SYMLINKS=OFF
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "Wicked CMake configure failed"
    }

    & cmake --build $WickedBuildDir --config Release --target vkSplatCapture --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        Exit-Fail "Wicked vkSplatCapture build failed"
    }
}

$CandidateBins = @(
    Join-Path $WickedBuildDir "Samples\vkSplatCapture\Release\vkSplatCapture.exe",
    Join-Path $WickedBuildDir "Samples\vkSplatCapture\vkSplatCapture.exe",
    Join-Path $WickedBuildDir "Samples\vkSplatCapture\RelWithDebInfo\vkSplatCapture.exe",
    Join-Path $WickedBuildDir "Samples\vkSplatCapture\Debug\vkSplatCapture.exe"
)
$CaptureBin = $CandidateBins | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $CaptureBin) {
    Exit-SkipOrFail "capture binary not found under $WickedBuildDir\Samples\vkSplatCapture"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RuntimeDir = Join-Path $WickedRoot "WickedEngine"
$OldPath = $env:PATH
$env:PATH = "$(Split-Path -Parent $CaptureBin);$RuntimeDir;$OldPath"
Push-Location $RuntimeDir
try {
    & $CaptureBin --scene *> $LogFile
    $Status = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PATH = $OldPath
}

if ($Status -ne 0) {
    Get-Content -Path $LogFile -Tail 120 | Write-Error
    Exit-Fail "capture harness exited with status $Status"
}

Assert-Log "initialized=yes" "Wicked did not initialize"
Assert-Log "adapter=.*(NVIDIA|GeForce|RTX|Quadro|Tesla|L[0-9]{2}|A[0-9]{2,}|H[0-9]{3}|B[0-9]{3})" "Wicked did not select an NVIDIA adapter"
Assert-Log "shader_format=spirv" "Wicked Vulkan backend did not report SPIR-V shaders"
Assert-Log "capability\.mesh_shader=yes" "NVIDIA Vulkan adapter did not expose mesh shader support through Wicked"
Assert-Log "capability\.raytracing=yes" "NVIDIA Vulkan adapter did not expose ray tracing support through Wicked"
Assert-Log "scene\.loaded=yes" "Cornell scene metadata did not load"
Assert-Log "render_path=RenderPath3D_PathTracing" "Wicked path tracing render path was not selected"
Assert-Log "capture\.ready=yes" "Wicked Cornell capture contract is not ready"
Assert-Log "capture\.mode=raytracing-ready" "Wicked Cornell capture did not reach raytracing-ready mode"

Write-Host "vkGSplat Wicked NVIDIA smoke: PASS"
Write-Host "vkGSplat Wicked NVIDIA smoke: log=$LogFile"
