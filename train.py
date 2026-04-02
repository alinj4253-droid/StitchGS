import os
import cv2
import json
import time
import math
import torch
import shutil
import random
import argparse
import sys
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader, Sampler

from gaussian_renderer import render
from utils.image_utils import psnr
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim, src2ref, loss_reproj
from scene.cameras import get_render_camera
from scene.gaussian_model import GaussianModel
from scene.scene_loader import SceneDataset, Scene
from utils.utils import parse_cfg, cal_local_cam_extent, save_cfg, read_pcdfile

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False





class OrderedIndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)





def _project_to_plane(c, vertical_axis="z"):
    if vertical_axis == "z": return np.array([c[0], c[1]])
    elif vertical_axis == "y": return np.array([c[0], c[2]])
    elif vertical_axis == "x": return np.array([c[1], c[2]])
    else: return np.array([c[0], c[1]])

def _camera_center_from_extrinsic(extr):
    W2C = extr
    C2W = np.linalg.inv(W2C)
    return C2W[:3, 3]

def _block_center_from_bbx_expand(bbx_expand):
    return np.asarray(bbx_expand).mean(axis=0)

def _make_balanced_order(views_info_list, block_bbx_expand, iterations, batch_size,
                         vertical_axis="z", angle_bins=8, dist_bins=3,
                         mix_prob=0.0, warmup_ratio=1.0, seed=42):
    rng = random.Random(seed)
    center3d = _block_center_from_bbx_expand(block_bbx_expand)
    center2d = _project_to_plane(center3d, vertical_axis)

    cam2d = []
    for v in views_info_list:
        c = _camera_center_from_extrinsic(v.extrinsic)
        p = _project_to_plane(c, vertical_axis)
        cam2d.append(p)
    cam2d = np.stack(cam2d, 0)

    vec = cam2d - center2d[None, :]
    angles = (np.arctan2(vec[:, 1], vec[:, 0]) + 2 * math.pi) % (2 * math.pi)
    dists = np.linalg.norm(vec, axis=1)

    lo, hi = np.percentile(dists, 5), np.percentile(dists, 95)
    if hi <= lo: lo, hi = dists.min(), dists.max() + 1e-6
    d_norm = np.clip((dists - lo) / (hi - lo), 0.0, 1.0)

    angle_ids = np.clip(np.floor(angles / (2 * math.pi / angle_bins)).astype(int), 0, angle_bins - 1)
    dist_ids = np.clip(np.floor(d_norm * dist_bins - 1e-8).astype(int), 0, dist_bins - 1)

    buckets = {}
    for idx in range(len(views_info_list)):
        key = (int(angle_ids[idx]), int(dist_ids[idx]))
        buckets.setdefault(key, []).append(idx)

    for key in buckets:
        rng.shuffle(buckets[key])

    keys = sorted(buckets.keys(), key=lambda x: (x[0], x[1]))
    K = len(keys)
    bucket_ptr = {k: 0 for k in keys}

    total = iterations * batch_size
    order_idx_list = []

    def pop_from_bucket(k):
        arr = buckets[k]
        if not arr: return None
        p = bucket_ptr[k]
        out = arr[p]
        bucket_ptr[k] = (p + 1) % len(arr)
        return out

    stride = 1
    for s in (1, 3, 5, 7, 9, 11):
        if K == 0 or math.gcd(s, K) == 1:
            stride = s
            break

    warmup_total = int(total * warmup_ratio)
    i = 0
    while i < warmup_total:
        base = (i // batch_size) % K
        for b in range(batch_size):
            if i >= warmup_total: break
            k = keys[(base + b * stride) % K]
            if mix_prob > 0.0 and rng.random() < mix_prob:
                ridx = rng.randrange(len(views_info_list))
                order_idx_list.append(ridx)
            else:
                idx = pop_from_bucket(k)
                if idx is None: idx = rng.randrange(len(views_info_list))
                order_idx_list.append(idx)
            i += 1

    while i < total:
        if mix_prob >= 1.0 or K == 0:
            ridx = rng.randrange(len(views_info_list))
            order_idx_list.append(ridx)
        else:
            if rng.random() < max(mix_prob, 0.5):
                ridx = rng.randrange(len(views_info_list))
                order_idx_list.append(ridx)
            else:
                k = keys[(i // batch_size + i) % K]
                idx = pop_from_bucket(k)
                if idx is None: idx = rng.randrange(len(views_info_list))
                order_idx_list.append(idx)
        i += 1

    return order_idx_list




def update_training_stats(output_dir, block_id, time_taken, final_pts, peak_vram_gb):

    stats_path = os.path.join(output_dir, "experiment_stats.json")


    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {"training_stats": []}

    if "training_stats" not in stats:
        stats["training_stats"] = []


    record = {
        "block_id": block_id,
        "time_sec": time_taken,
        "num_pts": final_pts,
        "vram_gb": peak_vram_gb
    }


    existing_idx = -1
    for i, item in enumerate(stats["training_stats"]):
        if item["block_id"] == block_id:
            existing_idx = i
            break

    if existing_idx >= 0:
        stats["training_stats"][existing_idx] = record
    else:
        stats["training_stats"].append(record)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[Paper Data] Block {block_id} stats saved: Time={time_taken:.1f}s, Pts={final_pts}, VRAM={peak_vram_gb:.1f}GB")





def reconstruct(cfg, block_id, block_bbx_expand, views_info_list, init_pcd, eval_views_info=None, device=torch.device("cuda")):
    print(f"\n{'='*20} Start Reconstructing Block {block_id} {'='*20}")
    print(f"[Info] Num block views (Unique): {len(views_info_list)}")

    point_cloud_path = os.path.join(cfg.output_dirpath, "point_cloud")


    torch.cuda.reset_peak_memory_stats()


    local_gaussian = GaussianModel(sh_degree=cfg.sh_degree)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    bg = torch.rand((3), device=device) if cfg.random_background else background

    num_views = len(views_info_list)
    cfg.position_lr_max_steps = cfg.iterations
    if cfg.densify_until_iter is None:
        cfg.densify_until_iter = cfg.iterations // 2

    if cfg.opacity_reset_interval is None:
        cfg.opacity_reset_interval = max(cfg.iterations // 10, 3000)


    save_cfg(cfg, block_id)


    angle_bins = getattr(cfg, "balance_angle_bins", 8)
    dist_bins = getattr(cfg, "balance_dist_bins", 3)
    mix_prob  = getattr(cfg, "balance_mix_prob", 0.0)
    warmup    = getattr(cfg, "balance_warmup_ratio", 1.0)
    seed      = getattr(cfg, "seed", 42)
    vertical_axis = getattr(cfg, "vertical_axis", "z")

    ordered_indices = _make_balanced_order(
        views_info_list=views_info_list,
        block_bbx_expand=block_bbx_expand if block_bbx_expand is not None else np.zeros((8,3)),
        iterations=cfg.iterations,
        batch_size=cfg.batch_size,
        vertical_axis=vertical_axis,
        angle_bins=angle_bins,
        dist_bins=dist_bins,
        mix_prob=mix_prob,
        warmup_ratio=warmup,
        seed=seed
    )


    block_preload_setting = 'cpu'
    print(f"[Info] Dataset preload setting: {block_preload_setting} (Optimized: Only loading unique views)")
    print(f"[IO-Wait] Loading {len(views_info_list)} unique images into RAM...")
    sys.stdout.flush()

    loading_start = time.time()
    scene_dataset = SceneDataset(
        views_info_list,
        cfg.image_scale,
        cfg.scene_scale,
        len(views_info_list),
        preload=block_preload_setting
    )
    loading_end = time.time()
    print(f"[IO-Done] Loaded {len(scene_dataset)} images in {loading_end - loading_start:.2f}s.")

    if eval_views_info is not None:
        eval_dataset = SceneDataset(eval_views_info, cfg.image_scale, cfg.scene_scale)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    sampler = OrderedIndexSampler(ordered_indices)

    scene_dataloader = DataLoader(
        scene_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )


    scene_extent = cal_local_cam_extent(views_info_list)
    print(f"[Info] Scene extent radius: {scene_extent:.4f}")
    local_gaussian.create_from_pcd(init_pcd, scene_extent)
    local_gaussian.training_setup(cfg)

    depth_l1_weight = get_expon_lr_func(cfg.depth_l1_weight_init, cfg.depth_l1_weight_final, max_steps=cfg.iterations)
    reproj_l1_weight = get_expon_lr_func(cfg.reproj_l1_weight_init, cfg.reproj_l1_weight_final, max_steps=cfg.iterations)

    start_time = time.time()
    progress_bar = tqdm(scene_dataloader, desc=f"Block {block_id} Train", total=len(ordered_indices)//cfg.batch_size)


    for iter_idx, view_info in enumerate(progress_bar):
        iteration = iter_idx + 1
        local_gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            local_gaussian.oneupSHdegree()

        batch_sample_num = view_info["extrinsic"].shape[0]

        total_loss = 0.0

        for sample_idx in range(batch_sample_num):
            extrinsic = view_info["extrinsic"][sample_idx].to(device)
            intrinsic = view_info["intrinsic"][sample_idx].to(device)
            image_height = view_info["image_height"][sample_idx].item()
            image_width  = view_info["image_width"][sample_idx].item()
            image_gt     = view_info["image"][sample_idx].to(device)

            camera_render = get_render_camera(image_height, image_width, extrinsic, intrinsic)
            render_pkg = render(camera_render, local_gaussian, cfg, bg)
            image_rendered = render_pkg["render"]

            l1_loss_photo = l1_loss(image_rendered, image_gt)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image_rendered.unsqueeze(0), image_gt.unsqueeze(0))
            else:
                ssim_value = ssim(image_rendered, image_gt)

            loss_photo = (1.0 - cfg.lambda_dssim) * l1_loss_photo + cfg.lambda_dssim * (1.0 - ssim_value)

            loss_scaling = local_gaussian.get_scaling.prod(dim=1).mean()
            loss = loss_photo + 0.01 * loss_scaling

            if cfg.depth_inv_loss and "depth_inv" in view_info and not isinstance(view_info["depth_inv"][sample_idx], str):
                depth_rendered_inv = render_pkg["depth"].squeeze(0)
                depth_gt_inv = view_info["depth_inv"][sample_idx].to(device)
                l1_loss_depth = torch.abs(depth_gt_inv - depth_rendered_inv).mean()
                loss += depth_l1_weight(iteration) * l1_loss_depth

            if getattr(cfg, "pseudo_loss", False) and iteration > getattr(cfg, "pseudo_loss_start", 0):
                depth_rendered = (1.0 / (render_pkg["depth"] + 1e-8)).squeeze(0)
                disturb = torch.tensor(
                    (0.05 * image_width * torch.median(depth_rendered) / intrinsic[0, 0], 0.0, 0.0), device=device
                )
                dummy_camera = get_render_camera(image_height, image_width, extrinsic, intrinsic, disturb=disturb)
                dummy_render_pkg = render(dummy_camera, local_gaussian, cfg, bg)
                dummy_rendered = torch.clamp(dummy_render_pkg["render"], 0.0, 1.0)
                dummy_depth_rendered = (1.0 / (dummy_render_pkg["depth"] + 1e-8)).squeeze(0)
                reprojected_depth, reprojected_image = src2ref(
                    camera_render.intrinsic, camera_render.extrinsic, depth_rendered,
                    dummy_camera.intrinsic, dummy_camera.extrinsic, dummy_depth_rendered, dummy_rendered
                )
                loss_reproj_photo = loss_reproj(reprojected_depth, reprojected_image, image_gt)
                loss += reproj_l1_weight(iteration) * loss_reproj_photo

            loss.backward()
            total_loss += loss.item()

        with torch.no_grad():
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            if iteration < cfg.densify_until_iter:
                local_gaussian.max_radii2D[visibility_filter] = torch.max(
                    local_gaussian.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                local_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > cfg.densify_from_iter and iteration % cfg.densification_interval == 0:
                    size_threshold = 500 if iteration > cfg.opacity_reset_interval else None
                    block_bbx_ = block_bbx_expand if cfg.densify_only_in_block else None
                    local_gaussian.densify_and_prune(
                        cfg.densify_grad_threshold, cfg.min_opacity, scene_extent, size_threshold, block_bbx_
                    )

                if iteration % cfg.opacity_reset_interval == 0 and iteration > cfg.densify_from_iter:
                    local_gaussian.reset_opacity()

            local_gaussian.optimizer.step()
            local_gaussian.optimizer.zero_grad(set_to_none=True)

        if iteration % 100 == 0:
            mem_use = torch.cuda.max_memory_allocated() / (1024 ** 3)
            num_pts = local_gaussian.get_xyz.shape[0]
            progress_bar.set_postfix({
                "Loss": f"{total_loss:.4f}",
                "Pts": f"{num_pts//1000}k",
                "Mem": f"{mem_use:.1f}G"
            })
            if iteration % 500 == 0:
                tqdm.write(f"[Iter {iteration}] Loss: {total_loss:.4f} | Pts: {num_pts} | Mem: {mem_use:.2f}G")

        if iteration % 2000 == 0:
            psnr_train_acc = 0.0
            num_samples = 5
            actual_samples = min(num_samples, len(scene_dataset))
            rand_indices = random.sample(range(len(scene_dataset)), actual_samples)

            subset = Subset(scene_dataset, rand_indices)
            temp_loader = DataLoader(subset, batch_size=1, shuffle=False)

            cnt = 0
            for v_info in temp_loader:
                ext = v_info["extrinsic"][0].to(device)
                intr = v_info["intrinsic"][0].to(device)
                h, w = v_info["image_height"][0].item(), v_info["image_width"][0].item()
                gt = v_info["image"][0].to(device)

                cam = get_render_camera(h, w, ext, intr)
                r_pkg = render(cam, local_gaussian, cfg, bg)
                psnr_train_acc += psnr(r_pkg["render"], gt).mean().item()
                cnt += 1

            avg_train_psnr = psnr_train_acc / max(cnt, 1)
            tqdm.write(f"[EVAL] Iter {iteration} | Train PSNR (Rand {cnt}): {avg_train_psnr:.4f}")

            if eval_views_info is not None:
                psnr_eval_acc = 0.0
                cnt_e = 0
                for _, v_info3 in enumerate(eval_dataloader):
                    ext3 = v_info3["extrinsic"].squeeze(0).to(device)
                    intr3 = v_info3["intrinsic"].squeeze(0).to(device)
                    h3, w3 = v_info3["image_height"].item(), v_info3["image_width"].item()
                    gt3 = v_info3["image"].squeeze(0).to(device)

                    cam3 = get_render_camera(h3, w3, ext3, intr3)
                    r_pkg3 = render(cam3, local_gaussian, cfg, bg)
                    psnr_eval_acc += psnr(r_pkg3["render"], gt3).mean().item()
                    cnt_e += 1

                tqdm.write(f"[EVAL] Iter {iteration} | Val PSNR (All {cnt_e}): {psnr_eval_acc/max(cnt_e, 1):.4f}")

        if iteration == cfg.iterations:
            out_dir = os.path.join(point_cloud_path, str(block_id))
            os.makedirs(out_dir, exist_ok=True)
            ply_path = os.path.join(out_dir, f"point_cloud_{iteration:03d}.ply")
            local_gaussian.save_ply(ply_path)
            tqdm.write(f"[Save] Saved point cloud to: {ply_path}")


    final_pts = local_gaussian.get_xyz.shape[0]
    elapsed_time = time.time() - start_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"Block {block_id} finished. Final Points: {final_pts}")


    update_training_stats(cfg.output_dirpath, block_id, elapsed_time, final_pts, peak_mem_gb)


    with open(os.path.join(cfg.output_dirpath, "time_consumption.txt"), "a") as f:
        f.write(f"block_id: {block_id}, pts: {final_pts}, time: {elapsed_time:.4f}s\n")

    del local_gaussian
    del scene_dataset
    del scene_dataloader
    del sampler
    if eval_views_info is not None:
        del eval_dataset
        del eval_dataloader

    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Clean] Memory released for Block {block_id}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction Process of View-based Gaussian Splatting.")
    parser.add_argument("--config", "-c", type=str, default="./configs/rubble.yaml", help="config filepath")
    parser.add_argument("--scene_dirpath", "-s", type=str, default=None, help="scene data dirpath")
    parser.add_argument("--output_dirpath", "-o", type=str, default=None, help="optimized result output dirpath")
    parser.add_argument("--block_ids", "-b", nargs="+", type=int, default=None)
    args = parser.parse_args()

    cfg = parse_cfg(args)

    original_preload_setting = getattr(cfg, "preload", None)
    print(f"[Init] Original preload setting: {original_preload_setting}")
    print("[Init] Loading global Scene metadata (preload temporarily disabled)...")

    cfg.preload = None
    scene = Scene(cfg.scene_dirpath, evaluate=cfg.evaluate, scene_scale=cfg.scene_scale)
    cfg.preload = original_preload_setting

    os.makedirs(cfg.output_dirpath, exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.output_dirpath, "config.yaml"))

    blocks_info_jsonpath = os.path.join(cfg.output_dirpath, "blocks_info.json")
    with open(blocks_info_jsonpath, "r") as json_file:
        blocks_info = json.load(json_file)

    num_blocks = blocks_info["num_blocks"]
    block_ids = args.block_ids if args.block_ids is not None else range(0, num_blocks)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"\n[Start] Processing {len(block_ids)} blocks: {list(block_ids)}")

    for block_id in block_ids:
        block_info = blocks_info[str(block_id)]
        pcd_filepath = block_info["block_pcd_filepath"]
        block_bbx_expand = np.array(block_info["bbx_expand"])

        print(f"Reading init PCD: {pcd_filepath}")
        pcd = read_pcdfile(pcd_filepath)

        views_info_list = [scene.views_info[vid] for vid in block_info["views_id"]]
        eval_views_info = None

        reconstruct(cfg, block_id, block_bbx_expand, views_info_list, pcd, eval_views_info, device=device)

        torch.cuda.empty_cache()
