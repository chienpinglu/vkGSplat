# assets/

Scene files (`.ply`, `.splat`) for testing the v1 3DGS path. Binaries
are **not** committed to the repo (see top-level `.gitignore`); fetch
them locally with the recipes below.

## Recommended starter scenes

The following are the standard novel-view-synthesis benchmarks the
position paper (`paper/main.tex`, Section "Evaluation plan") commits to
reporting against.

### Mip-NeRF 360 — `garden`, `bicycle`, `bonsai`

Trained 3DGS `.ply` files are mirrored by the original Kerbl 2023
release:

```bash
mkdir -p assets/mipnerf360
cd assets/mipnerf360
# See https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
# for the official trained models tarball; download manually due to
# license and click-through.
```

### Tanks & Temples — `truck`, `train`

Same source as above; the trained models tarball includes both the
T&T scenes and the Deep Blending scenes used in the paper.

### Synthetic test scene

For CI we maintain a tiny synthetic Gaussian cloud (a few thousand
primitives forming a colored cube). Generate it with:

```bash
python scripts/make_test_cube.py --out assets/test_cube.ply
```

The script does not exist yet; placeholder for v1.

## File-format notes

| Extension | Loader               | Source                                |
|-----------|----------------------|---------------------------------------|
| `.ply`    | `vksplat::io::load_ply`   | Kerbl 2023 trained-model layout       |
| `.splat`  | `vksplat::io::load_splat` | antimatter15/splat 32-byte record     |

Both loaders are skeletons in v1 — the full property walk is on the
roadmap (see `src/core/scene.cpp`). Until then, prefer the `.ply` files
from the official 3DGS release for the load path.
