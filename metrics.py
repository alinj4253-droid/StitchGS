import os
import yaml
import json
import torch
import argparse
import easydict
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim
import torchvision.transforms.functional as tf
from scene.colmap_loader import read_colmap_views_info


def _to_cuda_mat(x):
    if isinstance(x, torch.Tensor):
        return x.float().cuda()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().cuda()
    return torch.tensor(x, dtype=torch.float32, device="cuda")


def _safe_get(vobj, key, default=None):
    if isinstance(vobj, dict):
        return vobj.get(key, default)
    if hasattr(vobj, key):
        val = getattr(vobj, key)
        return val if val is not None else default
    try:
        return vobj[key]
    except Exception:
        return default


def _nanmean(xs):
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if np.isnan(xf) or np.isinf(xf):
            continue
        vals.append(xf)
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _sobel_grad_mag(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 2.0, 1.0]], device=device).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(x, kx, padding=1)
    gy = torch.nn.functional.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _load_blocks_obb_params(blocks_info_path: str, device: torch.device):
    with open(blocks_info_path, "r") as f:
        info = json.load(f)

    blocks_list = None
    if isinstance(info, list):
        blocks_list = info
    elif isinstance(info, dict):
        for key in ["blocks", "blocks_info", "block_infos", "partitions", "partition", "data"]:
            if key in info and isinstance(info[key], (list, dict)):
                blocks = info[key]
                if isinstance(blocks, dict):
                    try:
                        keys = sorted(blocks.keys(), key=lambda x: int(x))
                    except Exception:
                        keys = sorted(blocks.keys())
                    blocks_list = [blocks[k] for k in keys]
                else:
                    blocks_list = blocks
                break

        if blocks_list is None:
            id_keys = [k for k in info.keys() if k != "num_blocks"]
            numeric_like = 0
            for k in id_keys:
                try:
                    int(k)
                    numeric_like += 1
                except Exception:
                    pass
            if len(id_keys) > 0 and numeric_like / len(id_keys) > 0.8:
                keys = sorted(id_keys, key=lambda x: int(x))
                blocks_list = [info[k] for k in keys]

    if blocks_list is None or not isinstance(blocks_list, list) or len(blocks_list) == 0:
        raise KeyError(f"Unrecognized blocks_info.json schema: {list(info.keys()) if isinstance(info, dict) else type(info)}")

    def pick_corners(b):
        if not isinstance(b, dict):
            return None
        for k in ["bbx_expand", "bbx", "corners", "obb", "bbox", "obb_corners"]:
            if k in b:
                return b[k]
        for kk in ["meta", "data", "info"]:
            if kk in b and isinstance(b[kk], dict):
                for k in ["bbx_expand", "bbx", "corners", "obb", "bbox", "obb_corners"]:
                    if k in b[kk]:
                        return b[kk][k]
        return None

    centers, extents, rot_mats = [], [], []
    skipped = 0

    for b in blocks_list:
        corners = pick_corners(b)
        if corners is None:
            skipped += 1
            continue
        bbx = np.array(corners, dtype=np.float32)
        if bbx.size == 24:
            bbx = bbx.reshape(8, 3)
        if bbx.shape != (8, 3):
            skipped += 1
            continue

        center = bbx.mean(axis=0)
        vec_x = bbx[1] - bbx[0]
        vec_y = bbx[4] - bbx[0]
        vec_z = bbx[3] - bbx[0]

        ex = (np.linalg.norm(vec_x) / 2.0)
        ey = (np.linalg.norm(vec_y) / 2.0)
        ez = (np.linalg.norm(vec_z) / 2.0)

        ax = vec_x / (np.linalg.norm(vec_x) + 1e-12)
        ay = vec_y / (np.linalg.norm(vec_y) + 1e-12)
        az = vec_z / (np.linalg.norm(vec_z) + 1e-12)

        rot = np.stack([ax, ay, az], axis=1)
        centers.append(center)
        extents.append([ex, ey, ez])
        rot_mats.append(rot)

    if len(centers) == 0:
        raise RuntimeError(f"Failed to parse OBB corners from {blocks_info_path}. skipped={skipped}")

    centers = torch.tensor(np.stack(centers, axis=0), device=device, dtype=torch.float32)
    extents = torch.tensor(np.stack(extents, axis=0), device=device, dtype=torch.float32)
    rot_mats = torch.tensor(np.stack(rot_mats, axis=0), device=device, dtype=torch.float32)
    print(f"[Seam] Loaded OBBs: {centers.shape[0]} blocks (skipped {skipped}).")
    return centers, extents, rot_mats


def _backproject_depth_to_world(depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor):
    H, W = depth.shape
    valid = depth > 0

    ys, xs = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)
    Kinv = torch.inverse(intrinsic)
    cam = torch.einsum("ij,hwj->hwi", Kinv, pix) * depth[..., None]

    if extrinsic.shape == (3, 4):
        bottom = torch.tensor([[0, 0, 0, 1]], device=depth.device, dtype=extrinsic.dtype)
        extr4 = torch.cat([extrinsic, bottom], dim=0)
    else:
        extr4 = extrinsic

    c2w = torch.inverse(extr4)
    cam_h = torch.cat([cam, torch.ones((H, W, 1), device=depth.device, dtype=cam.dtype)], dim=-1)
    world_h = torch.einsum("ij,hwj->hwi", c2w, cam_h)
    Xw = world_h[..., :3]
    return Xw, valid


def _compute_seam_mask_from_obb(Xw, valid, centers, extents, rot_mats, band_hi=1.1, band_lo=1.0, chunk=200000):
    device = Xw.device
    H, W, _ = Xw.shape
    seam_mask = torch.zeros((H, W), device=device, dtype=torch.bool)

    idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return seam_mask

    pts = Xw.reshape(-1, 3)[idx]
    rot_t = rot_mats.transpose(1, 2).contiguous()
    eps = 1e-12

    N = pts.shape[0]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        p = pts[s:e]
        v = p[None, :, :] - centers[:, None, :]
        local = torch.einsum("bij,bmj->bmi", rot_t, v)
        normed = torch.abs(local) / (extents[:, None, :] + eps)
        d = normed.max(dim=-1).values
        in_band = (d > band_lo) & (d <= band_hi)
        m = in_band.any(dim=0)
        if m.any():
            seam_mask.reshape(-1)[idx[s:e][m]] = True

    return seam_mask


def _masked_psnr(diff: torch.Tensor, mask: torch.Tensor) -> float:
    mse_map = (diff * diff).mean(dim=1).squeeze(0)
    if mask.sum() == 0:
        return None
    mse = mse_map[mask].mean().clamp(min=1e-12)
    psnr_val = 10.0 * torch.log10(1.0 / mse)
    return psnr_val.item()


def _load_depth_gt(depth_dir: str, image_name: str):
    base = os.path.splitext(image_name)[0]
    candidates = [
        os.path.join(depth_dir, base + ".png"),
        os.path.join(depth_dir, base + ".jpg"),
        os.path.join(depth_dir, base + ".jpeg"),
        os.path.join(depth_dir, base + ".tiff"),
        os.path.join(depth_dir, base + ".tif"),
        os.path.join(depth_dir, base + ".npy"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        return None
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        return d
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _align_pred_depth_to_gt(depth_pred: torch.Tensor, depth_gt: torch.Tensor):
    valid = (depth_gt > 0) & (depth_pred > 0)
    if valid.sum() < 16:
        return depth_pred, 1.0
    ratio = depth_gt[valid] / depth_pred[valid].clamp(min=1e-6)
    s = ratio.median().clamp(min=1e-6, max=1e6)
    return depth_pred * s, float(s.item())


def load_experiment_stats(output_dir):
    stats_path = os.path.join(output_dir, "experiment_stats.json")
    if not os.path.exists(stats_path):
        print(f"[Warning] Stats file not found at {stats_path}. Paper metrics will be incomplete.")
        return None
    with open(stats_path, "r") as f:
        return json.load(f)


def generate_paper_report(output_dir, eval_metrics, exp_stats):
    if exp_stats is None:
        return

    n_blocks = exp_stats.get("N_blocks", 0)
    n_views_list = exp_stats.get("N_views_list", [])
    training_stats = exp_stats.get("training_stats", [])

    n_views_mean = np.mean(n_views_list) if n_views_list else 0
    n_views_max = np.max(n_views_list) if n_views_list else 0

    pts_list = [item["num_pts"] for item in training_stats] if training_stats else []
    n_pts_mean = np.mean(pts_list) if pts_list else 0
    n_pts_max = np.max(pts_list) if pts_list else 0

    time_list = [item["time_sec"] for item in training_stats] if training_stats else []
    t_opt_max = (np.max(time_list) / 60.0) if time_list else 0
    t_opt_total = (np.sum(time_list) / 60.0) if time_list else 0

    vram_list = [item.get("vram_gb", 0) for item in training_stats] if training_stats else []
    vram_max = np.max(vram_list) if vram_list else 0

    report = []
    report.append("==================================================")
    report.append("          BlockGaussian Paper Metrics Report      ")
    report.append("==================================================")
    report.append(f"Scene Path: {output_dir}")
    report.append("")
    report.append("[Quality Metrics] (Eval Set)")
    report.append(f"  PSNR:  {eval_metrics['PSNR']:.4f}")
    report.append(f"  SSIM:  {eval_metrics['SSIM']:.4f}")
    report.append(f"  LPIPS: {eval_metrics['LPIPS']:.4f}")

    seam_keys = ["Seam_PSNR", "BPD_L1", "nBGJ", "rBDD", "Seam_Coverage"]
    if all(k in eval_metrics for k in seam_keys) and eval_metrics.get("Seam_Coverage", None) is not None:
        report.append("")
        report.append("[Seam Metrics] (Eval Set, Seam Band Only)")
        report.append(f"  Seam_PSNR:      {eval_metrics['Seam_PSNR']:.4f}")
        report.append(f"  BPD_L1:         {eval_metrics['BPD_L1']:.6f}")
        report.append(f"  nBGJ:           {eval_metrics['nBGJ']:.4f}")
        report.append(f"  rBDD:           {eval_metrics['rBDD']:.4f}")
        report.append(f"  Seam_Coverage:  {eval_metrics['Seam_Coverage']*100:.2f}%")

    depth_keys = ["Seam_Depth_MAE", "Seam_Depth_RMSE", "nSeam_Depth_MAE"]
    if all(k in eval_metrics for k in depth_keys):
        report.append("")
        report.append("[Seam Depth Metrics] (Pred vs GT depth_any, Seam Band Only)")
        report.append(f"  Seam_Depth_MAE:   {eval_metrics['Seam_Depth_MAE']:.6f}")
        report.append(f"  Seam_Depth_RMSE:  {eval_metrics['Seam_Depth_RMSE']:.6f}")
        report.append(f"  nSeam_Depth_MAE:  {eval_metrics['nSeam_Depth_MAE']:.4f}")

    report.append("")
    report.append("[Partition Statistics]")
    report.append(f"  N_blocks:      {n_blocks}")
    report.append(f"  N_views (mean): {n_views_mean:.1f}")
    report.append(f"  N_views (max):  {n_views_max}")

    report.append("")
    report.append("[Training Statistics]")
    report.append(f"  N_pts (mean):   {n_pts_mean/1000000:.2f} M")
    report.append(f"  N_pts (max):    {n_pts_max/1000000:.2f} M")
    report.append(f"  t_opt (max):    {t_opt_max:.1f} min")
    report.append(f"  t_opt (total):  {t_opt_total:.1f} min")
    report.append(f"  VRAM (peak):    {vram_max:.2f} GB")
    report.append("==================================================")

    print("\n".join(report))
    save_path = os.path.join(output_dir, "Paper_Metrics.txt")
    with open(save_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n[Success] Full report saved to: {save_path}")


def evaluate(
    scene_dirpath,
    output_dirpath,
    split,
    scene_scale,
    result_dirname="render",
    train_eval_split=False,
    enable_seam_metrics=True,
    seam_band_lo=1.0,
    seam_band_hi=1.1,
    enable_gt_depth_metrics=True,
    gt_depth_subdir="depth_any",
):
    render_dirpath = os.path.join(output_dirpath, result_dirname, split, "rendered")
    if not os.path.isdir(render_dirpath):
        raise FileNotFoundError(f"Rendered images dir not found: {render_dirpath}")

    if train_eval_split and split == "eval":
        gt_dirpath = os.path.join(scene_dirpath.replace("train", "val"), "images")
        cam_dirpath = scene_dirpath.replace("train", "val")
        gt_depth_dirpath = os.path.join(scene_dirpath.replace("train", "val"), gt_depth_subdir)
    else:
        gt_dirpath = os.path.join(scene_dirpath, "images")
        cam_dirpath = scene_dirpath
        gt_depth_dirpath = os.path.join(scene_dirpath, gt_depth_subdir)

    per_view_metrics = {}
    ssims, psnrs, lpipss = [], [], []

    depth_pred_dirpath = os.path.join(output_dirpath, result_dirname, split, "rendered_depth")
    blocks_info_path = os.path.join(output_dirpath, "blocks_info.json")

    seam_ready = enable_seam_metrics and os.path.isdir(depth_pred_dirpath) and os.path.exists(blocks_info_path)
    if enable_seam_metrics and not seam_ready:
        print("[Warning] Seam metrics requested, but rendered_depth/ or blocks_info.json not found. Seam metrics will be skipped.")
        seam_ready = False

    gt_depth_ready = enable_gt_depth_metrics and os.path.isdir(gt_depth_dirpath)
    if enable_gt_depth_metrics and not gt_depth_ready:
        print(f"[Warning] GT depth metrics requested, but GT depth dir not found: {gt_depth_dirpath}. GT depth metrics will be skipped.")
        gt_depth_ready = False




    views_by_base = {}
    centers = extents = rot_mats = None
    if seam_ready:
        views_info, _, __ = read_colmap_views_info(cam_dirpath, False, scene_scale)
        for _, v in views_info.items():
            fp = _safe_get(v, "image_filepath", None)
            if fp is None:
                fp = _safe_get(v, "image_path", None)
            if fp is None:
                fp = _safe_get(v, "filepath", None)
            if isinstance(fp, (list, tuple)):
                fp = fp[0]
            if fp is None:
                continue
            base = os.path.splitext(os.path.basename(fp))[0]
            views_by_base[base] = v

        device = torch.device("cuda")
        centers, extents, rot_mats = _load_blocks_obb_params(blocks_info_path, device=device)

    seam_psnrs, bpds, nbgjs, rbdds, seam_covs = [], [], [], [], []
    seam_d_mae, seam_d_rmse, nseam_d_mae = [], [], []

    image_names = sorted(os.listdir(render_dirpath))
    for image_name in tqdm(image_names, desc="Evaluating"):
        render_path = os.path.join(render_dirpath, image_name)
        if not os.path.isfile(render_path):
            continue

        image_render = Image.open(render_path)
        base = os.path.splitext(image_name)[0]


        gt_name = image_name.replace("png", "jpg")
        if not os.path.exists(os.path.join(gt_dirpath, gt_name)):
            gt_name = image_name
        image_gt = Image.open(os.path.join(gt_dirpath, gt_name))

        if image_render.width != image_gt.width or image_render.height != image_gt.height:
            image_gt = image_gt.resize((image_render.width, image_render.height))

        image_render_t = tf.to_tensor(image_render).unsqueeze(0)[:, :3, :, :].cuda()
        image_gt_t = tf.to_tensor(image_gt).unsqueeze(0)[:, :3, :, :].cuda()

        ssims.append(ssim(image_render_t, image_gt_t))
        psnrs.append(psnr(image_render_t, image_gt_t))
        lpipss.append(lpips(image_render_t, image_gt_t, net_type='vgg'))

        per_view_metrics[image_name] = {
            "PSNR": psnrs[-1].item(),
            "SSIM": ssims[-1].item(),
            "LPIPS": lpipss[-1].item()
        }

        if not (seam_ready and (base in views_by_base)):
            continue

        vinfo = views_by_base[base]

        depth_pred_path = os.path.join(depth_pred_dirpath, base + ".npy")
        if not os.path.exists(depth_pred_path):
            continue

        depth_pred_np = np.load(depth_pred_path).astype(np.float32)
        depth_pred = torch.from_numpy(depth_pred_np).float().cuda()

        intrinsic = _to_cuda_mat(getattr(vinfo, "intrinsic"))
        extrinsic = _to_cuda_mat(getattr(vinfo, "extrinsic"))

        depth_for_mask = None
        depth_gt = None
        if gt_depth_ready:
            gt_d = _load_depth_gt(gt_depth_dirpath, image_name)
            if gt_d is not None:
                if gt_d.shape[0] != depth_pred_np.shape[0] or gt_d.shape[1] != depth_pred_np.shape[1]:
                    gt_img = Image.fromarray(gt_d)
                    gt_img = gt_img.resize((depth_pred_np.shape[1], depth_pred_np.shape[0]), resample=Image.NEAREST)
                    gt_d = np.array(gt_img).astype(np.float32)
                depth_gt = torch.from_numpy(gt_d.astype(np.float32)).float().cuda()
                depth_for_mask = depth_gt

        if depth_for_mask is None:
            depth_for_mask = depth_pred

        Xw, valid = _backproject_depth_to_world(depth_for_mask, intrinsic, extrinsic)
        seam_mask = _compute_seam_mask_from_obb(
            Xw, valid, centers, extents, rot_mats,
            band_hi=seam_band_hi, band_lo=seam_band_lo
        )

        seam_pixels = int(seam_mask.sum().item())
        total_pixels = int(depth_pred.numel())
        seam_cov = seam_pixels / max(total_pixels, 1)

        per_view_metrics[image_name].update({
            "Seam_Coverage": float(seam_cov),
            "Seam_Pixels": int(seam_pixels),
        })

        if seam_pixels == 0 or seam_pixels == total_pixels:
            per_view_metrics[image_name].update({
                "Seam_PSNR": None,
                "BPD_L1": None,
                "nBGJ": None,
                "rBDD": None,
            })
            continue

        diff = image_render_t - image_gt_t
        l1_map = diff.abs().mean(dim=1).squeeze(0)
        bpd = float(l1_map[seam_mask].mean().item())
        seam_psnr = _masked_psnr(diff, seam_mask)

        lum = (0.299 * image_render_t[:, 0:1] + 0.587 * image_render_t[:, 1:2] + 0.114 * image_render_t[:, 2:3])
        grad = _sobel_grad_mag(lum).squeeze(0).squeeze(0)
        g_seam = grad[seam_mask].mean()
        g_non = grad[~seam_mask].mean()
        if torch.isnan(g_non) or (g_non.abs() < 1e-12):
            nbgj = None
        else:
            nbgj = float((g_seam / g_non).item())

        depth_4 = depth_pred.view(1, 1, depth_pred.shape[0], depth_pred.shape[1])
        dgrad = _sobel_grad_mag(depth_4).squeeze(0).squeeze(0)
        dg_seam = dgrad[seam_mask].mean()
        dg_non = dgrad[~seam_mask].mean()
        if torch.isnan(dg_non) or (dg_non.abs() < 1e-12):
            rbdd = None
        else:
            rbdd = float((dg_seam / dg_non).item())

        per_view_metrics[image_name].update({
            "Seam_PSNR": float(seam_psnr) if seam_psnr is not None else None,
            "BPD_L1": float(bpd),
            "nBGJ": nbgj,
            "rBDD": rbdd,
        })

        if seam_psnr is not None:
            seam_psnrs.append(seam_psnr)
        bpds.append(bpd)
        seam_covs.append(seam_cov)
        if nbgj is not None:
            nbgjs.append(nbgj)
        if rbdd is not None:
            rbdds.append(rbdd)

        if gt_depth_ready and (depth_gt is not None):
            depth_pred_aligned, _ = _align_pred_depth_to_gt(depth_pred, depth_gt)

            valid2 = (depth_gt > 0) & (depth_pred_aligned > 0) & seam_mask
            valid_non = (depth_gt > 0) & (depth_pred_aligned > 0) & (~seam_mask)

            if valid2.sum() > 0 and valid_non.sum() > 0:
                derr = (depth_pred_aligned - depth_gt).abs()
                mae_seam = float(derr[valid2].mean().item())
                rmse_seam = float(torch.sqrt(((depth_pred_aligned - depth_gt) ** 2)[valid2].mean().clamp(min=1e-12)).item())
                mae_non = float(derr[valid_non].mean().item())
                nmae = float((mae_seam / max(mae_non, 1e-12)))

                seam_d_mae.append(mae_seam)
                seam_d_rmse.append(rmse_seam)
                nseam_d_mae.append(nmae)

                per_view_metrics[image_name].update({
                    "Seam_Depth_MAE": mae_seam,
                    "Seam_Depth_RMSE": rmse_seam,
                    "nSeam_Depth_MAE": nmae,
                })

    scene_metrics = {
        "PSNR": float(torch.tensor(psnrs).mean().item()) if len(psnrs) else None,
        "SSIM": float(torch.tensor(ssims).mean().item()) if len(ssims) else None,
        "LPIPS": float(torch.tensor(lpipss).mean().item()) if len(lpipss) else None
    }

    if seam_ready:
        m_seam_psnr = _nanmean(seam_psnrs)
        m_bpd = _nanmean(bpds)
        m_nbgj = _nanmean(nbgjs)
        m_rbdd = _nanmean(rbdds)
        m_cov = _nanmean(seam_covs)

        if m_seam_psnr is not None:
            scene_metrics["Seam_PSNR"] = m_seam_psnr
        if m_bpd is not None:
            scene_metrics["BPD_L1"] = m_bpd
        if m_nbgj is not None:
            scene_metrics["nBGJ"] = m_nbgj
        if m_rbdd is not None:
            scene_metrics["rBDD"] = m_rbdd
        if m_cov is not None:
            scene_metrics["Seam_Coverage"] = m_cov

    if gt_depth_ready:
        m_mae = _nanmean(seam_d_mae)
        m_rmse = _nanmean(seam_d_rmse)
        m_nmae = _nanmean(nseam_d_mae)
        if m_mae is not None:
            scene_metrics["Seam_Depth_MAE"] = m_mae
        if m_rmse is not None:
            scene_metrics["Seam_Depth_RMSE"] = m_rmse
        if m_nmae is not None:
            scene_metrics["nSeam_Depth_MAE"] = m_nmae

    os.makedirs(os.path.join(output_dirpath, result_dirname, split), exist_ok=True)
    with open(os.path.join(output_dirpath, result_dirname, split, "result.json"), "w") as file:
        json.dump(scene_metrics, file, indent=True)
    with open(os.path.join(output_dirpath, result_dirname, split, "per_view.json"), "w") as file:
        json.dump(per_view_metrics, file, indent=True)

    return scene_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metrics calculation script parameters.")
    parser.add_argument("--optimized_path", "-o", type=str, required=True, help="optimized scene dirpath")
    parser.add_argument("--train_eval_split", action="store_true", help="train and eval is stored sperately")
    parser.add_argument("--eval_only", action="store_true", help="only evaluate eval split")

    parser.add_argument("--disable_seam_metrics", action="store_true")
    parser.add_argument("--seam_band_lo", type=float, default=1.0)
    parser.add_argument("--seam_band_hi", type=float, default=1.1)

    parser.add_argument("--disable_gt_depth_metrics", action="store_true")
    parser.add_argument("--gt_depth_subdir", type=str, default="depth_any")

    args = parser.parse_args()

    config_filepath = os.path.join(args.optimized_path, "config.yaml")
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Config missing: {config_filepath}")

    with open(config_filepath, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = easydict.EasyDict(cfg)
    cfg.output_dirpath = args.optimized_path

    enable_seam = (not args.disable_seam_metrics)
    enable_gt_depth = (not args.disable_gt_depth_metrics)

    print(f"Evaluating: {args.optimized_path}")

    eval_view_metrics = evaluate(
        cfg.scene_dirpath,
        cfg.output_dirpath,
        "eval",
        scene_scale=cfg.scene_scale,
        result_dirname="render",
        train_eval_split=args.train_eval_split,
        enable_seam_metrics=enable_seam,
        seam_band_lo=args.seam_band_lo,
        seam_band_hi=args.seam_band_hi,
        enable_gt_depth_metrics=enable_gt_depth,
        gt_depth_subdir=args.gt_depth_subdir,
    )
    print("eval_view metrics: ", eval_view_metrics)

    if not args.eval_only:
        train_view_metrics = evaluate(
            cfg.scene_dirpath,
            cfg.output_dirpath,
            "train",
            scene_scale=cfg.scene_scale,
            result_dirname="render",
            train_eval_split=args.train_eval_split,
            enable_seam_metrics=enable_seam,
            seam_band_lo=args.seam_band_lo,
            seam_band_hi=args.seam_band_hi,
            enable_gt_depth_metrics=enable_gt_depth,
            gt_depth_subdir=args.gt_depth_subdir,
        )
        print("train_view metrics: ", train_view_metrics)

    exp_stats = load_experiment_stats(args.optimized_path)
    generate_paper_report(args.optimized_path, eval_view_metrics, exp_stats)
