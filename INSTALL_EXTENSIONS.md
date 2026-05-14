# CUDA Extensions

This repository tracks third-party CUDA extension source trees as Git submodules. A normal `git clone` only checks out the submodule pointers; use `--recursive` or run `git submodule update --init --recursive` before installing.

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
  Upstream: `https://gitlab.inria.fr/bkerbl/simple-knn.git`

## Optional

- `fused-ssim`
  Upstream: `https://github.com/rahul-goel/fused-ssim.git`
  `train.py` falls back to the Python SSIM implementation if this package is not installed.

## Installation

```bash
git submodule update --init --recursive

pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim --no-build-isolation
```

You can also clone everything in one step:

```bash
git clone --recursive https://github.com/alinj4253-droid/StitchGS.git
```

If you use forks or patched copies of these extensions, keep the same directory names so the import paths remain unchanged.
