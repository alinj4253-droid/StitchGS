#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p submodules

if [ ! -d submodules/diff-gaussian-rasterization ]; then
  git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization
fi

if [ ! -d submodules/simple-knn ]; then
  git clone https://github.com/graphdeco-inria/simple-knn.git submodules/simple-knn
fi

if [ ! -d submodules/fused-ssim ]; then
  git clone https://github.com/rahul-goel/fused-ssim.git submodules/fused-ssim
fi

pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim --no-build-isolation

