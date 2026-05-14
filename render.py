#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import torch
import argparse
import easydict
import numpy as np
from tqdm import tqdm
from utils.image_utils import read_image, save_image
from gaussian_renderer import render
from scene.cameras import get_render_camera
from scene.gaussian_model import GaussianModel
from scene.scene_loader import Scene, SceneDataset
from scene.colmap_loader import read_colmap_views_info


def custom_load_ply_adaptive(self, path):
    print(f"[AdaptiveLoader] Loading model: {path}")

    meta_path = path + ".params.npz"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing params: {meta_path}")
    meta = np.load(meta_path)
    q_min, q_range = meta['q_min'], meta['q_range']

    with open(path, 'rb') as f:
        content = f.read()
        header_end_idx = content.find(b'end_header\n') + len(b'end_header\n')

    header = content[:header_end_idx].decode('utf-8')
    for line in header.split('\n'):
        if line.startswith('element vertex'):
            num_points = int(line.split()[-1])
            break

    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for i in range(3):
        dtype_list.append((f'f_dc_{i}', 'f2'))
    for i in range(45):
        dtype_list.append((f'f_rest_{i}', 'u1'))
    dtype_list.append(('opacity', 'f2'))
    for i in range(3):
        dtype_list.append((f'scale_{i}', 'f2'))
    for i in range(4):
        dtype_list.append((f'rot_{i}', 'f2'))

    data = np.frombuffer(content[header_end_idx:], dtype=dtype_list, count=num_points)

    xyz = np.stack((data['x'], data['y'], data['z']), axis=1).astype(np.float32)
    f_dc = np.stack([data[f'f_dc_{i}'] for i in range(3)], axis=1).reshape(-1, 1, 3).astype(np.float32)

    f_rest_int = np.stack([data[f'f_rest_{i}'] for i in range(45)], axis=1).astype(np.float32)
    f_rest_float = (f_rest_int / 255.0) * q_range + q_min


    zero_mask = np.mean(np.abs(f_rest_float), axis=1) < 0.05
    pruned_count = np.sum(zero_mask)
    print(f"[AdaptiveLoader] Pruned SH-rest points: {pruned_count} / {num_points} ({pruned_count/num_points*100:.1f}%).")

    f_rest_correct = f_rest_float.reshape(-1, 3, 15)

    opacities = data['opacity'].copy()[..., np.newaxis].astype(np.float32)
    scale = np.stack([data[f'scale_{i}'] for i in range(3)], axis=1).astype(np.float32)
    rotation = np.stack([data[f'rot_{i}'] for i in range(4)], axis=1).astype(np.float32)

    self._xyz = torch.nn.Parameter(torch.tensor(xyz, device="cuda").requires_grad_(False))
    self._features_dc = torch.nn.Parameter(torch.tensor(f_dc, device="cuda").contiguous().requires_grad_(False))
    self._features_rest = torch.nn.Parameter(torch.tensor(f_rest_correct, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
    self._opacity = torch.nn.Parameter(torch.tensor(opacities, device="cuda").requires_grad_(False))
    self._scaling = torch.nn.Parameter(torch.tensor(scale, device="cuda").requires_grad_(False))
    self._rotation = torch.nn.Parameter(torch.tensor(rotation, device="cuda").requires_grad_(False))
    self.active_sh_degree = self.max_sh_degree


def color_correct(img: np.ndarray, ref: np.ndarray, num_iters: int = 5, eps: float = 0.5 / 255):
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError("Channels must match")
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))
    mask0 = is_unclipped(img_mat)
    for _ in range(num_iters):
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])
        a_mat.append(img_mat)
        a_mat.append(np.ones_like(img_mat[:, :1]))
        a_mat = np.concatenate(a_mat, axis=-1)
        warp = []
        for c in range(num_channels):
            b = ref_mat[:, c]
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = np.where(mask[:, None], a_mat, 0)
            mb = np.where(mask, b, 0)
            w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            warp.append(w)
        warp = np.stack(warp, axis=-1)
        img_mat = np.clip(np.matmul(a_mat, warp), 0, 1)
    return np.reshape(img_mat, img.shape)


def render_loop(cfg, scene_gaussian, image_rendered_dirpath, views_info_list, color_correction=False, save_depth=True):
    os.makedirs(image_rendered_dirpath, exist_ok=True)
    depth_rendered_dirpath = os.path.join(os.path.dirname(image_rendered_dirpath), 'rendered_depth')
    if save_depth:
        os.makedirs(depth_rendered_dirpath, exist_ok=True)

    device = torch.device("cuda")
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    bg = torch.rand((3), device=device) if cfg.random_background else background

    dataset = SceneDataset(views_info_list, cfg.image_scale, cfg.scene_scale)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    for _, view_info in enumerate(tqdm(dataloader, desc="Rendering")):
        extrinsic = view_info["extrinsic"].squeeze(0).to(device)
        intrinsic = view_info["intrinsic"].squeeze(0).to(device)
        h, w = view_info["image_height"].item(), view_info["image_width"].item()
        camera_render = get_render_camera(h, w, extrinsic, intrinsic)

        render_pkg = render(camera_render, scene_gaussian, cfg, bg)

        image_rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
        image_rendered = image_rendered.permute(1, 2, 0).cpu().numpy()

        if color_correction:
            image_ref = read_image(view_info["image_filepath"][0], image_scale=cfg.image_scale)
            image_rendered = color_correct(image_rendered, image_ref)

        image_name = os.path.basename(view_info["image_filepath"][0]).replace("jpg", "png")


        if save_depth and ('depth' in render_pkg):
            depth = render_pkg['depth'].detach().float().squeeze()
            depth = depth.cpu().numpy().astype(np.float32)
            depth_name = os.path.splitext(image_name)[0] + '.npy'
            np.save(os.path.join(depth_rendered_dirpath, depth_name), depth)

        save_image(image_rendered, os.path.join(image_rendered_dirpath, image_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None, help="config file; infers --optimized_path when -o is omitted")
    parser.add_argument("--optimized_path", "-o", type=str, default=None)
    parser.add_argument("--train_eval_split", action="store_true")
    parser.add_argument("--eval_only", action="store_true", help="when train_eval_split, only render eval split")
    parser.add_argument("--no_save_depth", action="store_true", help="do not save depth maps (npy) for seam metrics")
    args = parser.parse_args()

    if args.optimized_path is None:
        if args.config is None:
            parser.error("provide --config / -c or --optimized_path / -o")
        with open(args.config) as f:
            args.optimized_path = yaml.load(f, Loader=yaml.FullLoader)["output_dirpath"]

    config_filepath = os.path.join(args.optimized_path, "config.yaml")
    if not os.path.exists(config_filepath):
        raise FileNotFoundError("Config missing")

    with open(config_filepath, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = easydict.EasyDict(cfg)
    cfg.output_dirpath = args.optimized_path

    GaussianModel.load_ply = custom_load_ply_adaptive

    with torch.no_grad():
        scene_gaussian = GaussianModel(sh_degree=cfg.sh_degree)
        ply_path = os.path.join(cfg.output_dirpath, "point_cloud_quantized.ply")
        scene_gaussian.load_ply(ply_path)

        save_depth = (not args.no_save_depth)

        if args.train_eval_split:

            eval_views_info, _, __ = read_colmap_views_info(
                cfg.scene_dirpath.replace("train", "val"),
                False,
                cfg.scene_scale
            )
            render_loop(
                cfg, scene_gaussian,
                os.path.join(cfg.output_dirpath, "render", "eval", "rendered"),
                list(eval_views_info.values()),
                True,
                save_depth=save_depth
            )


            if not args.eval_only:
                train_views_info, _, __ = read_colmap_views_info(
                    cfg.scene_dirpath,
                    False,
                    cfg.scene_scale
                )
                render_loop(
                    cfg, scene_gaussian,
                    os.path.join(cfg.output_dirpath, "render", "train", "rendered"),
                    list(train_views_info.values()),
                    True,
                    save_depth=save_depth
                )
        else:
            scene = Scene(cfg.scene_dirpath, evaluate=cfg.evaluate, scene_scale=cfg.scene_scale)
            render_loop(
                cfg, scene_gaussian,
                os.path.join(cfg.output_dirpath, "render", "eval", "rendered"),
                [scene.views_info[vid] for vid in scene.eval_views_id],
                True,
                save_depth=save_depth
            )

    print("[Done] Rendering Finished.")
