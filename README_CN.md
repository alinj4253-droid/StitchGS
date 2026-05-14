# StitchGS：面向无缝、轻量化大规模 3D 高斯泼溅

发表于 *Remote Sensing*，2026 年，第 18 卷，第 10 期，文章编号 1460  
[论文主页](https://www.mdpi.com/2072-4292/18/10/1460) | [PDF](https://www.mdpi.com/2072-4292/18/10/1460/pdf) | [English README](README.md)

---

## 简介

大规模 3D 高斯泼溅（3DGS）通常将城市场景切分为若干独立子块分别训练，由此带来两个核心问题：块边界处出现明显接缝，以及模型体积偏大。StitchGS 针对这两个问题提出系统性解决方案。

- **无缝合并**：随机交织缝合（Stochastic Interwoven Stitching）通过质量加权概率门控在边界处决定高斯点的保留或丢弃；随后通过全局一致性优化（Global Consistency Refinement）恢复全场景光度一致性。
- **轻量化表示**：频谱感知自适应压缩（Spectral-Aware Adaptive Compression）对漫反射区域裁剪高阶球谐系数；混合精度存储（float16 + uint8）大幅压缩模型体积。

---

## 环境配置

```bash
conda env create -f environment.yml
conda activate stitchgs
bash scripts/install_extensions.sh
```

CUDA 扩展（`diff-gaussian-rasterization`、`simple-knn`、`fused-ssim`）不随本仓库发布，详见 [INSTALL_EXTENSIONS.md](INSTALL_EXTENSIONS.md)。  
依赖：Python 3.10、PyTorch 1.12.1、CUDA 11.6。

---

## 使用方法

先进入项目根目录：

```bash
cd StitchGS
```


### 第 1 步 — 场景分块（交互式）

```bash
python scene_partition.py -c configs/rubble.yaml
```

会弹出 matplotlib 窗口，手动点击框定 ROI 多边形，按 Enter 确认。

### 第 2 步 — 分块训练

```bash
python train.py -c configs/rubble.yaml
# 只训练指定块：
python train.py -c configs/rubble.yaml --block_ids 0 1 2
```

### 第 3 步 — 随机交织缝合

```bash
python merge.py -c configs/rubble.yaml --k_neighbors 5 --seed 0
```

### 第 4 步 — 全局一致性优化 + QAT

```bash
python finetune_qat.py -c configs/rubble.yaml --iterations 7000
# 若 merge 使用了非默认 seed：
python finetune_qat.py -c configs/rubble.yaml --iterations 7000 --seed 1
```

### 第 5 步 — 压缩

```bash
python compress_adaptive.py -c configs/rubble.yaml -t 0.02
```

### 第 6、7 步 — 渲染与评测

```bash
python render.py  -c configs/rubble.yaml --train_eval_split --eval_only
python metrics.py -c configs/rubble.yaml --train_eval_split --eval_only
```

### 一键运行

```bash
bash scripts/run_pipeline_example.sh configs/rubble.yaml
```

将 `configs/rubble.yaml` 替换为 `configs/` 目录下其他配置文件即可切换场景。

---

## 数据准备

流程需要 COLMAP 风格场景目录：

```
<scene_dir>/
  images/
  sparse/0/
    cameras.bin / cameras.txt
    images.bin  / images.txt
    points3D.bin / points3D.txt
```

含训练/评测分割的数据集需在 `train/` 同级目录准备 `val/`，并使用 `--train_eval_split`。

| 数据集 | 对应配置 |
|---|---|
| Mill-19 | rubble、building |
| UrbanScene3D | residence、sciart |
| MatrixCity | mc_street、mc_aerial |

---

## 引用

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

## 许可证

本项目仅供非商业研究用途，详见 [LICENSE](LICENSE)。`scene/`、`utils/`、`gaussian_renderer/` 等目录下继承自 [3D Gaussian Splatting（INRIA）](https://github.com/graphdeco-inria/gaussian-splatting) 的部分代码适用其原始许可协议。
