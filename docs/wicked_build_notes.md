# Wicked Engine Local Build Notes

This records the local build used for vkGSplat's first Wicked Engine example.

## Built Target

- Source: `third_party/WickedEngine`
- Build directory: `third_party/WickedEngine/build-vkgsplat-example`
- Targets: `Template_Linux`, `vkGSplatCapture`
- Template output: `third_party/WickedEngine/build-vkgsplat-example/Samples/Template_Linux/Template_Linux`
- Capture output: `third_party/WickedEngine/build-vkgsplat-example/Samples/vkGSplatCapture/vkGSplatCapture`
- Platform used here: macOS / Apple Silicon / AppleClang

## Configure Command

```bash
cmake -S third_party/WickedEngine \
  -B third_party/WickedEngine/build-vkgsplat-example \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,-no_fixup_chains \
  -DWICKED_EDITOR=OFF \
  -DWICKED_TESTS=OFF \
  -DWICKED_IMGUI_EXAMPLE=OFF \
  -DWICKED_LINUX_TEMPLATE=ON \
  -DWICKED_VKGSPLAT_CAPTURE=ON \
  -DWICKED_ENABLE_SYMLINKS=OFF \
  -DUSE_SSE4_1=OFF \
  -DUSE_SSE4_2=OFF \
  -DUSE_AVX=OFF \
  -DUSE_AVX2=OFF \
  -DUSE_AVX512=OFF \
  -DUSE_LZCNT=OFF \
  -DUSE_TZCNT=OFF \
  -DUSE_F16C=OFF \
  -DUSE_FMADD=OFF
```

Build command:

```bash
cmake --build third_party/WickedEngine/build-vkgsplat-example --target Template_Linux --parallel 8
cmake --build third_party/WickedEngine/build-vkgsplat-example --target vkGSplatCapture --parallel 8
```

## Local Patches Needed

The upstream checkout did not build this CMake sample cleanly on the current macOS/Apple Silicon environment. The local build required:

- disabling x86 SIMD CMake options so DirectXMath does not include x86-only intrinsic headers,
- adding Wicked's vendored `Utility/metal` headers to the include path,
- excluding the Metal graphics backend for this Vulkan-oriented sample build,
- routing `wiApplication.cpp` through the Vulkan/DX12 backend selection path instead of the macOS Metal-only branch,
- preferring SDL cursor handling over Apple cursor handling when both `__APPLE__` and `SDL2` are defined,
- adding `wiAppleHelper.mm` and `wiInput_Apple.mm` to the CMake source list,
- skipping precompiled headers for the Objective-C++ `.mm` files,
- linking macOS frameworks needed by the Apple helper sources,
- enabling Vulkan portability enumeration for MoltenVK instance creation,
- using SDL drawable sizing for the SDL sample on macOS,
- adding tolerance for zero swapchain extent capability bounds,
- reserving but not null-writing Wicked's bindless descriptor safety slot on macOS/MoltenVK,
- giving Vulkan bindless descriptor pools slack for MoltenVK's variable descriptor count accounting,
- using `-Wl,-no_fixup_chains` so the Apple linker accepts Wicked's bundled FAudio objects on arm64.

OpenImageDenoise was not found, so Wicked's optional OIDN path-tracing denoiser is disabled in this build.

## Runtime Smoke Test

The sample requires Homebrew's Vulkan loader and MoltenVK runtime on this machine:

```bash
brew install vulkan-loader molten-vk vulkan-tools
```

Run from `third_party/WickedEngine/WickedEngine` so Wicked finds `./libdxcompiler.dylib`, shader sources, and the generated `shaders/spirv` cache:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/Cellar/molten-vk/1.4.1/etc/vulkan/icd.d/MoltenVK_icd.json \
../build-vkgsplat-example/Samples/Template_Linux/Template_Linux vulkan
```

Observed smoke-test result on Apple M4:

- `GraphicsDevice_Vulkan` is created through MoltenVK.
- Adapter reports `Apple M4`.
- Wicked loads `./libdxcompiler.dylib`.
- First run compiles the SPIR-V shader cache under `third_party/WickedEngine/WickedEngine/shaders/spirv`.
- Warm-cache run reaches `wi::renderer Initialized` and `[wi::initializer] Wicked Engine Initialized`.
- In the Codex desktop/sandbox launch context, the SDL template exits immediately after initialization with status `-1`; no Vulkan instance, descriptor pool, or Metal zero-texture assertion remains.

## vkGSplat Capture Harness

`Samples/vkGSplatCapture` is a dedicated deterministic harness for vkGSplat integration. It creates a small hidden SDL Vulkan window only so Wicked can query the platform Vulkan instance extensions, constructs `wi::graphics::GraphicsDevice_Vulkan` directly, assigns `wi::graphics::GetDevice()`, initializes Wicked, prints adapter/capability facts, and exits without entering the generic SDL template loop.

Run from `third_party/WickedEngine/WickedEngine`:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/Cellar/molten-vk/1.4.1/etc/vulkan/icd.d/MoltenVK_icd.json \
../build-vkgsplat-example/Samples/vkGSplatCapture/vkGSplatCapture
```

For the Cornell capture-contract probe:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/Cellar/molten-vk/1.4.1/etc/vulkan/icd.d/MoltenVK_icd.json \
../build-vkgsplat-example/Samples/vkGSplatCapture/vkGSplatCapture --scene
```

Observed result on Apple M4 / MoltenVK 1.4.1:

```text
vkGSplatCapture: initialized=yes
vkGSplatCapture: adapter=Apple M4
vkGSplatCapture: driver=MoltenVK: 1.4.1
vkGSplatCapture: shader_format=spirv
vkGSplatCapture: capability.mesh_shader=no
vkGSplatCapture: capability.raytracing=no
```

This verifies that Wicked's Vulkan backend is available and consumes SPIR-V on this machine. It also shows that MoltenVK does not expose the Vulkan mesh shader or Vulkan ray tracing capabilities needed for the eventual hardware-RT/mesh-shader acceptance test, so that test will need either an RTX-class Vulkan backend or a non-RT fallback path for local Apple runs.

The `--scene` probe currently avoids Wicked's full OBJ import/render-prep path on MoltenVK because that path repeatedly attempts graphics pipeline variants which fail with `VK_ERROR_INITIALIZATION_FAILED` before any ray-tracing capture can happen. Instead, it scans `cornellbox.obj` metadata, configures the intended path-tracing capture contract, and exits deterministically:

```text
vkGSplatCapture: scene.loaded=yes
vkGSplatCapture: scene.importer=obj-metadata-scan
vkGSplatCapture: scene.path=../Content/models/cornellbox.obj
vkGSplatCapture: scene.meshes=1
vkGSplatCapture: scene.objects=1
vkGSplatCapture: scene.materials=3
vkGSplatCapture: scene.vertices=68
vkGSplatCapture: scene.faces=17
vkGSplatCapture: camera.resolution=256x256
vkGSplatCapture: render_path=RenderPath3D_PathTracing
vkGSplatCapture: capture.surface.color=traceResult
vkGSplatCapture: capture.surface.depth=traceDepth
vkGSplatCapture: capture.surface.primitive_id=rtPrimitiveID
vkGSplatCapture: capture.surface.motion=derived_screen_space_motion
vkGSplatCapture: capture.ready=no
vkGSplatCapture: capture.mode=metadata-only-no-vulkan-raytracing
```

## Next Step

Run the real Wicked importer and `RenderPath3D_PathTracing` capture on an RTX-class Vulkan backend. On MoltenVK, keep the metadata-only capability-gated fallback so local Apple runs report `raytracing=no` instead of failing ambiguously.
