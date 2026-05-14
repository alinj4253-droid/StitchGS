# CUDA Extensions

This public release does not bundle third-party CUDA extension source trees. The code expects the following local layout under `submodules/`:

```text
submodules/
  diff-gaussian-rasterization/
  simple-knn/
  fused-ssim/
```

## Required

- `diff-gaussian-rasterization`
  Upstream: `https://github.com/graphdeco-inria/diff-gaussian-rasterization.git`
- `simple-knn`
  Upstream: `https://github.com/graphdeco-inria/simple-knn.git`

## Optional

- `fused-ssim`
  Upstream: `https://github.com/rahul-goel/fused-ssim.git`
  `train.py` falls back to the Python SSIM implementation if this package is not installed.

## Installation

```bash
mkdir -p submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization
git clone https://github.com/graphdeco-inria/simple-knn.git submodules/simple-knn
git clone https://github.com/rahul-goel/fused-ssim.git submodules/fused-ssim

pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim --no-build-isolation
```

If you use forks or patched copies of these extensions, keep the same directory names so the import paths remain unchanged.

