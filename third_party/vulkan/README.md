# Vulkan SDK

This folder is where the Vulkan SDK and related loaders/layers are installed.
The SDK is **not** committed to git (see top-level `.gitignore`); follow the
steps below per platform. Target version: **Vulkan SDK 1.3.x or newer**.

## Layout after install

```
third_party/vulkan/
├── README.md            (this file)
├── sdk/                 Extracted SDK (gitignored)
│   ├── include/vulkan/
│   ├── lib/
│   ├── bin/             (glslc, glslangValidator, spirv-*, vulkaninfo)
│   └── share/vulkan/    (validation layers, ICDs)
└── setup-env.sh         Sourced by scripts to set VULKAN_SDK / PATH
```

## macOS (Apple Silicon / Intel) — MoltenVK

The official LunarG SDK on macOS is built on top of MoltenVK (Vulkan-on-Metal).

```bash
# 1. Download the SDK installer
#    https://vulkan.lunarg.com/sdk/home#mac
#    -> "vulkansdk-macos-1.3.x.x.dmg"

# 2. Or via brew (unofficial but convenient):
brew install --cask vulkan-sdk

# 3. Verify
vulkaninfo --summary
```

Note: For CUDA + Vulkan interop work, **macOS is not viable** — NVIDIA CUDA does
not target macOS. Use Linux or Windows for the main build; macOS is fine only
for the Vulkan-frontend prototyping pieces.

## Linux (Ubuntu 22.04 / 24.04)

```bash
# Option A: LunarG apt repo (recommended, latest SDK)
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | \
    sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update
sudo apt install vulkan-sdk

# Option B: Distro packages (older SDK, fine for many uses)
sudo apt install libvulkan-dev vulkan-tools vulkan-validationlayers \
                 spirv-tools glslang-tools

# Verify
vulkaninfo --summary
glslc --version
```

## Windows

```powershell
# 1. Download from https://vulkan.lunarg.com/sdk/home#windows
#    Run VulkanSDK-1.3.x.x-Installer.exe

# 2. The installer sets VULKAN_SDK and adds bin/ to PATH automatically.

# 3. Verify
vulkaninfo --summary
glslc --version
```

## CUDA + Vulkan interop notes

For the interop layer in `src/interop/` you need:

- A discrete NVIDIA GPU with a recent driver (>= 535 on Linux).
- CUDA Toolkit 12.x.
- Vulkan extensions:
  - `VK_KHR_external_memory` (+ platform variant: `_fd` on Linux, `_win32` on Windows)
  - `VK_KHR_external_semaphore` (+ platform variant)
  - `VK_KHR_timeline_semaphore`
- CUDA driver APIs: `cuImportExternalMemory`, `cuImportExternalSemaphore`.

Verify your GPU exposes these with:

```bash
vulkaninfo | grep -E "external_memory|external_semaphore|timeline_semaphore"
```

## Helper script

A convenience script lives at `scripts/setup_vulkan.sh` — it detects platform
and downloads/extracts the SDK into `third_party/vulkan/sdk/`.
