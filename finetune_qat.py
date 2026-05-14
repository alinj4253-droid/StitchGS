#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
from tqdm import tqdm
from random import randint
import gc
from argparse import ArgumentParser
import numpy as np
import random

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.scene_loader import Scene, SceneDataset
from scene.cameras import get_render_camera

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, q_min, q_max):
        scale = (q_max - q_min) / 255.0
        scale[scale == 0] = 1e-6
        input_q = (input - q_min) / scale
        input_q = input_q.round().clamp(0, 255)
        input_dq = input_q * scale + q_min
        return input_dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class QuantAwareGaussianModel(GaussianModel):
    def __init__(self, sh_degree):
        super().__init__(sh_degree)

    def get_quantized_features_rest(self):
        f_rest = self._features_rest
        with torch.no_grad():
            flat = f_rest.reshape(f_rest.shape[0], -1)
            d_min = flat.min(dim=0)[0].reshape(1, 15, 3)
            d_max = flat.max(dim=0)[0].reshape(1, 15, 3)
        return FakeQuantizeFunction.apply(f_rest, d_min, d_max)

    def manual_prune(self, mask):
        keep_mask = ~mask
        self._xyz = torch.nn.Parameter(self._xyz[keep_mask])
        self._features_dc = torch.nn.Parameter(self._features_dc[keep_mask])
        self._features_rest = torch.nn.Parameter(self._features_rest[keep_mask])
        self._opacity = torch.nn.Parameter(self._opacity[keep_mask])
        self._scaling = torch.nn.Parameter(self._scaling[keep_mask])
        self._rotation = torch.nn.Parameter(self._rotation[keep_mask])

        if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.shape[0] == mask.shape[0]:
            self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        if hasattr(self, 'denom') and self.denom.shape[0] == mask.shape[0]:
            self.denom = self.denom[keep_mask]
        if hasattr(self, 'max_radii2D') and self.max_radii2D.shape[0] == mask.shape[0]:
            self.max_radii2D = self.max_radii2D[keep_mask]

def finetune(args):
    set_seed(args.seed)
    print(f"[Seed] {args.seed}")

    gaussians = QuantAwareGaussianModel(sh_degree=3)

    print(f"[Init] Loading merged PLY from {args.merged_ply}...")
    if not os.path.exists(args.merged_ply):
        raise FileNotFoundError(f"Merged PLY not found: {args.merged_ply}")
    gaussians.load_ply(args.merged_ply)

    n_before = gaussians.get_xyz.shape[0]
    print(f"[Info] Initial points: {n_before}")

    OPACITY_THRESHOLD = -4.595
    opacity_mask = gaussians._opacity.squeeze() < OPACITY_THRESHOLD
    n_pruned_op = opacity_mask.sum().item()

    if n_pruned_op > 0:
        print(f"[Smart Clean] Removing {n_pruned_op} invisible points (Opacity < 0.05)...")
        gaussians.manual_prune(opacity_mask)
        n_current = gaussians.get_xyz.shape[0]
        print(f"   -> Points reduced to: {n_current}")

    scale_mask = torch.max(gaussians.get_scaling, dim=1).values > 50.0
    n_pruned_scale = scale_mask.sum().item()
    if n_pruned_scale > 0:
        print(f"[Smart Clean] Removing {n_pruned_scale} giant outlier points...")
        gaussians.manual_prune(scale_mask)

    HARD_LIMIT = 13000000
    n_current = gaussians.get_xyz.shape[0]

    if n_current > HARD_LIMIT:
        print(f"[Safety] Points {n_current} still > {HARD_LIMIT}. Performing random downsample...")
        n_to_remove = n_current - HARD_LIMIT
        random_mask = torch.zeros(n_current, dtype=torch.bool, device="cuda")
        indices = torch.randperm(n_current)[:n_to_remove]
        random_mask[indices] = True
        gaussians.manual_prune(random_mask)
        print(f"[Safety] Downsampled to {HARD_LIMIT} points.")

    torch.cuda.empty_cache()

    print(f"[Init] Loading scene metadata from {args.source_path}...")
    scene = Scene(args.source_path, evaluate=False, scene_scale=1.0)

    TRAIN_SCALE = 0.5
    print(f"[Init] Using image scale {TRAIN_SCALE} for VRAM safety.")

    train_views_info = [scene.views_info[vid] for vid in scene.train_views_id]
    train_dataset = SceneDataset(train_views_info, image_scale=TRAIN_SCALE, scene_scale=1.0, preload=False)
    print(f"[Init] Training images: {len(train_dataset)}")

    print("[Init] Setting up QAT Optimizer (SGD)...")
    gaussians._xyz.requires_grad = False
    gaussians._scaling.requires_grad = False
    gaussians._rotation.requires_grad = False
    gaussians._features_dc.requires_grad = True
    gaussians._features_rest.requires_grad = True
    gaussians._opacity.requires_grad = True

    param_groups = [
        {'params': [gaussians._features_dc], 'lr': 0.0025, 'name': 'f_dc'},
        {'params': [gaussians._features_rest], 'lr': 0.00025, 'name': 'f_rest'},
        {'params': [gaussians._opacity], 'lr': 0.05, 'name': 'opacity'}
    ]

    optimizer = torch.optim.SGD(param_groups, lr=0.0, momentum=0.9)
    gaussians.optimizer = optimizer

    iter_start = 0
    iter_end = args.iterations
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(iter_start, iter_end), desc="Finetuning (QAT)")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    qat_start_iter = int(args.iterations * 0.1)

    for iteration in range(iter_start, iter_end):
        try:
            idx = randint(0, len(train_dataset) - 1)
            data_packet = train_dataset[idx]

            extrinsic = data_packet["extrinsic"].cuda()
            intrinsic = data_packet["intrinsic"].cuda()
            gt_image = data_packet["image"].cuda()
            h, w = data_packet["image_height"], data_packet["image_width"]

            viewpoint_cam = get_render_camera(h, w, extrinsic, intrinsic)

            original_f_rest = gaussians._features_rest

            if iteration >= qat_start_iter:
                quantized_f_rest = gaussians.get_quantized_features_rest()
                gaussians._features_rest = quantized_f_rest

            render_pkg = render(viewpoint_cam, gaussians, args, background)
            image = render_pkg["render"]

            gaussians._features_rest = original_f_rest

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gaussians._opacity.data = torch.clamp(gaussians._opacity.data, max=100.0)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                status = "QAT:ON" if iteration >= qat_start_iter else "QAT:OFF"
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.4f}", "Mode": status, "Pts": f"{gaussians.get_xyz.shape[0]}"})

            if iteration % 100 == 0:
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"\n[Warning] OOM at iter {iteration}. Clearing cache and skipping step.")
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            continue

    progress_bar.close()

    save_path = os.path.join(args.output_path, "point_cloud_finetuned_final.ply")
    print(f"[Done] Saving QAT model to {save_path}...")
    gaussians.save_ply(save_path)

if __name__ == "__main__":
    import yaml
    parser = ArgumentParser(description="Quantization-Aware Finetuning")
    parser.add_argument("--config", "-c", type=str, default=None, help="config file; infers -s and -o when omitted")
    parser.add_argument("--source_path", "-s", type=str, default=None)
    parser.add_argument("--merged_ply", "-m", type=str, default=None, help="merged PLY; defaults to <output_path>/point_cloud_merged_seed{seed}.ply")
    parser.add_argument("--output_path", "-o", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--white_background", action="store_true")

    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            _cfg = yaml.safe_load(f)
        if args.source_path is None:
            args.source_path = _cfg["scene_dirpath"]
        if args.output_path is None:
            args.output_path = _cfg["output_dirpath"]

    if args.source_path is None or args.output_path is None:
        parser.error("provide --config / -c or both --source_path / -s and --output_path / -o")

    if args.merged_ply is None:
        args.merged_ply = os.path.join(args.output_path, f"point_cloud_merged_seed{args.seed}.ply")

    finetune(args)
