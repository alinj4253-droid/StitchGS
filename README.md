# StitchGS: Towards Seamless and Lightweight Large-Scale 3D Gaussian Splatting

**Jinhe Su, [Shengfang Pan](https://orcid.org/0009-0005-0879-599X), Huanxin Zhu, Siyu Chen, Yaoming Huang, Yixin Zhou**  
School of Computer Engineering, Jimei University

*Remote Sensing*, 2026, Vol. 18, No. 10, Article 1460  
[Paper](https://www.mdpi.com/2072-4292/18/10/1460) | [PDF](https://www.mdpi.com/2072-4292/18/10/1460/pdf) | [中文说明](README_CN.md)

---

## Overview

Large-scale 3DGS partitions urban scenes into independent blocks, which creates two problems: visible seams at block boundaries and large model sizes. StitchGS addresses both.

- **Seamless merging**: Stochastic Interwoven Stitching resolves overlapping Gaussians at boundaries using quality-weighted probabilistic gating, followed by Global Consistency Refinement to restore photometric coherence across the full scene.
- **Lightweight representation**: Spectral-Aware Adaptive Compression prunes high-order SH coefficients for diffuse regions; Mixed-Precision Storage packs attributes in float16 and SH-rest in uint8.

---


## Setup

```bash
conda env create -f environment.yml
conda activate stitchgs
bash scripts/install_extensions.sh
```

CUDA extensions (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`) are not bundled. See [INSTALL_EXTENSIONS.md](INSTALL_EXTENSIONS.md). Requires Python 3.10, PyTorch 1.12.1, CUDA 11.6.

---

## Usage

```bash
cd StitchGS
```

### 1 — Scene Partitioning *(interactive)*

```bash
python scene_partition.py -c configs/rubble.yaml
```

### 2 — Block Training

```bash
python train.py -c configs/rubble.yaml
# Train specific blocks only:
python train.py -c configs/rubble.yaml --block_ids 0 1 2
```

### 3 — Stitching

```bash
python merge.py -c configs/rubble.yaml --k_neighbors 5 --seed 0
```

### 4 — Global Refinement + QAT

```bash
python finetune_qat.py -c configs/rubble.yaml --iterations 7000
# If merge used a non-default seed:
python finetune_qat.py -c configs/rubble.yaml --iterations 7000 --seed 1
```

### 5 — Compression

```bash
python compress_adaptive.py -c configs/rubble.yaml -t 0.02
```

### 6 & 7 — Render and Evaluate

```bash
python render.py  -c configs/rubble.yaml --train_eval_split --eval_only
python metrics.py -c configs/rubble.yaml --train_eval_split --eval_only
```

### One-command Pipeline

```bash
bash scripts/run_pipeline_example.sh configs/rubble.yaml
```

Replace `configs/rubble.yaml` with any config under `configs/` to switch scenes.

---

## Data Preparation

The pipeline expects a COLMAP-style scene directory:

```
<scene_dir>/
  images/
  sparse/0/
    cameras.bin / cameras.txt
    images.bin  / images.txt
    points3D.bin / points3D.txt
```

For train/eval split datasets, provide a sibling `val/` directory alongside `train/` and use `--train_eval_split`.

| Dataset | Configs |
|---|---|
| Mill-19 | rubble, building |
| UrbanScene3D | residence, sciart |
| MatrixCity | mc_street, mc_aerial |

---

## Citation

```bibtex
@article{stitchgs2026,
  title     = {StitchGS: Towards Seamless and Lightweight Large-Scale 3D Gaussian Splatting},
  author    = {Su, Jinhe and Pan, Shengfang and Zhu, Huanxin and Chen, Siyu and Huang, Yaoming and Zhou, Yixin},
  journal   = {Remote Sensing},
  volume    = {18},
  number    = {10},
  pages     = {1460},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/rs18101460}
}
```

---

## Authors

| Name | ORCID |
|---|---|
| Jinhe Su | [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--1707--5685-green)](https://orcid.org/0000-0003-1707-5685) |
| Shengfang Pan | [![ORCID](https://img.shields.io/badge/ORCID-0009--0005--0879--599X-green)](https://orcid.org/0009-0005-0879-599X) |
| Huanxin Zhu | — |
| Siyu Chen | — |
| Yaoming Huang | — |
| Yixin Zhou | — |

---

## License

Released for non-commercial research use. See [LICENSE](LICENSE). Parts derived from [3D Gaussian Splatting (INRIA)](https://github.com/graphdeco-inria/gaussian-splatting) retain their original license.
