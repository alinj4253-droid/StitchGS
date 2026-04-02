# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import parse_cfg, read_pcdfile
from scene.scene_loader import Scene
from utils.general_utils import storePly
from scene.colmap_loader import (
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text,
    read_points3D_binary_, read_points3D_text_
)



def plot_rectangle(rectangle: np.ndarray, color=None, text=None):

    rectangle = np.asarray(rectangle)
    if color is None:
        color = np.random.rand(3)
    closed = np.vstack([rectangle, rectangle[0]])
    plt.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.2)
    plt.plot(closed[:, 0], closed[:, 1], color=color, alpha=0.7)
    if text is not None:
        cx = np.mean(rectangle[:, 0])
        cy = np.mean(rectangle[:, 1])
        plt.text(cx, cy, text, fontsize=10, color='blue')

def minimum_area_bounding_rectangle(polygon_xy: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    pts = np.asarray(polygon_xy, dtype=np.float64)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    best_area = np.inf
    best_rect = None
    for i in range(len(hull_pts)):
        p1 = hull_pts[i]
        p2 = hull_pts[(i + 1) % len(hull_pts)]
        edge = p2 - p1
        theta = np.arctan2(edge[1], edge[0])
        c = np.cos(-theta)
        s = np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        rot = hull_pts @ R.T
        min_xy = rot.min(axis=0)
        max_xy = rot.max(axis=0)
        area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
        if area < best_area:
            best_area = area
            rect = np.array([
                [min_xy[0], min_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
                [min_xy[0], max_xy[1]],
            ])
            best_rect = rect @ R
    return best_rect

def expand_rectangle(rect_xy: np.ndarray, ratio: float) -> np.ndarray:
    rect_xy = np.asarray(rect_xy, dtype=np.float64)
    c = rect_xy.mean(axis=0, keepdims=True)
    return c + (rect_xy - c) * (1.0 + float(ratio))

def cross2(u, v):
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

def points_in_rotated_rectangle(points_xy: np.ndarray, rect_xy: np.ndarray) -> np.ndarray:
    P = np.asarray(points_xy, dtype=np.float64)
    R = np.asarray(rect_xy, dtype=np.float64)
    A, B, C, D = R[0], R[1], R[2], R[3]
    AB = (B - A)[None, :]
    BC = (C - B)[None, :]
    CD = (D - C)[None, :]
    DA = (A - D)[None, :]

    AP = P - A[None, :]
    BP = P - B[None, :]
    CP = P - C[None, :]
    DP = P - D[None, :]

    s1 = cross2(np.repeat(AB, len(P), axis=0), AP)
    s2 = cross2(np.repeat(BC, len(P), axis=0), BP)
    s3 = cross2(np.repeat(CD, len(P), axis=0), CP)
    s4 = cross2(np.repeat(DA, len(P), axis=0), DP)

    inside_pos = (s1 >= 0) & (s2 >= 0) & (s3 >= 0) & (s4 >= 0)
    inside_neg = (s1 <= 0) & (s2 <= 0) & (s3 <= 0) & (s4 <= 0)
    return inside_pos | inside_neg

def divide_rect_along_longer_edge(rect_xy: np.ndarray):
    rect = np.asarray(rect_xy, dtype=np.float64)
    e1_len = np.linalg.norm(rect[1] - rect[0])
    e2_len = np.linalg.norm(rect[3] - rect[0])

    if e1_len >= e2_len:
        m1 = (rect[0] + rect[1]) / 2
        m2 = (rect[2] + rect[3]) / 2
        sub1 = np.array([rect[0], m1, m2, rect[3]])
        sub2 = np.array([m1, rect[1], rect[2], m2])
    else:
        m1 = (rect[0] + rect[3]) / 2
        m2 = (rect[1] + rect[2]) / 2
        sub1 = np.array([rect[0], rect[1], m2, m1])
        sub2 = np.array([m1, m2, rect[2], rect[3]])
    return sub1, sub2

def select_scene_area(pts_2d, cams_2d):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Select ROI (Click to add points; Press Enter to finish)")
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c='k', s=0.3, alpha=0.05)
    ax.scatter(cams_2d[:, 0], cams_2d[:, 1], c='r', s=1.0, alpha=0.8)
    poly = []

    (line,) = ax.plot([], [], 'yo-')
    def on_click(e):
        if e.inaxes != ax: return
        poly.append([e.xdata, e.ydata])
        xs, ys = zip(*poly)
        line.set_data(xs, ys)
        fig.canvas.draw()

    def on_key(e):
        if e.key == 'enter' and len(poly) >= 3:
            poly.append(poly[0])
            xs, ys = zip(*poly)
            line.set_data(xs, ys)
            fig.canvas.draw()
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.axis('equal'); ax.axis('off')
    plt.show()
    if len(poly) < 3:
        raise RuntimeError("ROI polygon not defined. Please click at least 3 points and press Enter.")
    return np.array(poly[:-1], dtype=np.float64)

def divide_condition(rect_xy, condition_vars):
    pts_2d_init, num_points_thresh = condition_vars
    mask = points_in_rotated_rectangle(pts_2d_init, rect_xy)
    cnt = int(mask.sum())
    print(f"block init point number: {cnt}")
    return cnt <= int(num_points_thresh)

def recursive_split(rect_xy, out_list, depth, max_depth, condition_vars):
    stop = divide_condition(rect_xy, condition_vars)
    if stop or depth >= max_depth:
        out_list.append(rect_xy)
    else:
        r1, r2 = divide_rect_along_longer_edge(rect_xy)
        recursive_split(r1, out_list, depth + 1, max_depth, condition_vars)
        recursive_split(r2, out_list, depth + 1, max_depth, condition_vars)

def fetch_local_pcd(scene_dirpath, block_pcd_filepath, views_id, scene_scale=1.0):
    assert os.path.exists(os.path.join(scene_dirpath, "sparse/0")),        "sparse folder not found (expect COLMAP style at scene_dirpath/sparse/0)."
    bin_path = os.path.join(scene_dirpath, "sparse/0/points3D.bin")
    txt_path = os.path.join(scene_dirpath, "sparse/0/points3D.txt")
    try:
        points3D = read_points3D_binary_(bin_path, scene_scale)
    except Exception:
        points3D = read_points3D_text_(txt_path, scene_scale)

    try:
        images_info = read_extrinsics_binary(os.path.join(scene_dirpath, "sparse/0/images.bin"))
    except Exception:
        images_info = read_extrinsics_text(os.path.join(scene_dirpath, "sparse/0/images.txt"))

    pts_ids_total = set()
    for vid in views_id:
        if vid in images_info:
            extr = images_info[vid]
            ids = extr.point3D_ids[extr.point3D_ids != -1]
            pts_ids_total.update(ids.tolist())

    pts_ids_total = list(pts_ids_total)
    if len(pts_ids_total) == 0:
        return 0

    xyzs = np.array([points3D[pid].xyz for pid in pts_ids_total], dtype=np.float32)
    rgbs = np.array([points3D[pid].rgb for pid in pts_ids_total], dtype=np.float32)

    os.makedirs(os.path.dirname(block_pcd_filepath), exist_ok=True)
    storePly(block_pcd_filepath, xyzs, rgbs)
    return len(pts_ids_total)


def save_partition_stats(output_dir, blocks_info):

    stats_path = os.path.join(output_dir, "experiment_stats.json")


    num_blocks = blocks_info["num_blocks"]
    n_views_list = []

    for i in range(num_blocks):
        if i in blocks_info or str(i) in blocks_info:

            key = i if i in blocks_info else str(i)
            n_views_list.append(blocks_info[key]["num_views"])


    stats = {
        "N_blocks": num_blocks,
        "N_views_list": n_views_list,

        "training_stats": [],
        "metrics": {}
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[Paper Data] Partition stats saved to {stats_path}")



def scene_partion(cfg, vertical_axis="z"):
    axis_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis]

    scene = Scene(cfg.scene_dirpath, evaluate=cfg.evaluate, scene_scale=cfg.scene_scale)
    cam_centers_3d = [np.linalg.inv(scene.views_info[vid].extrinsic)[:3, 3] for vid in scene.train_views_id]
    cam_centers_3d = np.stack(cam_centers_3d, axis=0)

    pts_bin = os.path.join(cfg.scene_dirpath, "sparse/0/points3D.bin")
    try:
        points3D = read_points3D_binary_(pts_bin, scale=cfg.scene_scale)
    except Exception:
        points3D = read_points3D_text_(os.path.join(cfg.scene_dirpath, "sparse/0/points3D.txt"), scale=cfg.scene_scale)

    all_ids = list(points3D.keys())
    pts_xyz = np.stack([points3D[i].xyz for i in all_ids], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pts_xyz_f = np.asarray(pcd.points)

    if vertical_axis == "z":
        cams_2d = cam_centers_3d[:, [0, 1]]
        pts_2d = pts_xyz_f[:, [0, 1]]
        pts_2d_init = pts_xyz[:, [0, 1]]
    elif vertical_axis == "y":
        cams_2d = cam_centers_3d[:, [0, 2]]
        pts_2d = pts_xyz_f[:, [0, 2]]
        pts_2d_init = pts_xyz[:, [0, 2]]
    elif vertical_axis == "x":
        cams_2d = cam_centers_3d[:, [1, 2]]
        pts_2d = pts_xyz_f[:, [1, 2]]
        pts_2d_init = pts_xyz[:, [1, 2]]
    else:
        raise ValueError("vertical_axis should be one of ['x','y','z'].")

    polygon_pts = select_scene_area(pts_2d, cams_2d)
    roi_rect = minimum_area_bounding_rectangle(polygon_pts)

    plt.figure("ROI Region", figsize=(8, 8))
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='k', s=0.1, alpha=0.05)
    plt.scatter(cams_2d[:, 0], cams_2d[:, 1], c='r', s=0.5, alpha=0.8)
    plot_rectangle(roi_rect, color=[0, 1, 0], text="ROI")
    plt.axis('equal'); plt.axis('off')
    os.makedirs(cfg.output_dirpath, exist_ok=True)
    plt.savefig(os.path.join(cfg.output_dirpath, "ROI_region.png"), dpi=200)
    plt.show()

    block_rects = []
    cond_vars = [pts_2d_init, cfg.num_points_thresh]
    print()
    recursive_split(roi_rect, block_rects, 0, int(cfg.max_tree_depth), cond_vars)
    print(f"candidate block number (recursive): {len(block_rects)}")

    if len(block_rects) == 0:
        print("[fallback] no blocks created by recursion; build 3x3 grid over ROI.")
        R = roi_rect
        A, B, C, D = R[0], R[1], R[2], R[3]
        blocks = []
        for i in range(3):
            for j in range(3):
                p1 = A + i * (B - A) / 3 + j * (D - A) / 3
                p2 = A + (i + 1) * (B - A) / 3 + j * (D - A) / 3
                p3 = A + (i + 1) * (B - A) / 3 + (j + 1) * (D - A) / 3
                p4 = A + i * (B - A) / 3 + (j + 1) * (D - A) / 3
                blocks.append(np.stack([p1, p2, p3, p4], axis=0))
        block_rects = blocks

    try:
        images_info = read_extrinsics_binary(os.path.join(cfg.scene_dirpath, "sparse/0/images.bin"))
    except Exception:
        images_info = read_extrinsics_text(os.path.join(cfg.scene_dirpath, "sparse/0/images.txt"))

    id2idx = {pid: k for k, pid in enumerate(all_ids)}
    pts_xy_all = pts_2d_init

    image_to_ptidxset = {}
    for img_id, im in images_info.items():
        ids = im.point3D_ids
        ids = ids[ids != -1]
        if len(ids) == 0:
            image_to_ptidxset[img_id] = set()
        else:
            valid = [id2idx[i] for i in ids if i in id2idx]
            image_to_ptidxset[img_id] = set(valid)

    cover_ratio_view_thr = float(getattr(cfg, "cover_ratio_thresh", 0.2))
    cover_ratio_block_min = 0.05
    target_views = int(getattr(cfg, "target_views_per_block", 0))
    min_views_per_block = int(getattr(cfg, "min_views_per_block", 20))

    plt.figure("Scene Partition Blocks", figsize=(8, 8))
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='k', s=0.1, alpha=0.3)
    plt.scatter(cams_2d[:, 0], cams_2d[:, 1], c='r', s=0.5, alpha=0.8)

    blocks_info = {}
    valid_block_count = 0

    for block_idx, rect in enumerate(block_rects):
        print("###" * 20)
        print(f"Candidate block idx: {block_idx}")

        rect_expand = expand_rectangle(rect, ratio=float(cfg.expand_ratio))
        mask = points_in_rotated_rectangle(pts_xy_all, rect_expand)
        num_pts_in_block = int(mask.sum())
        if num_pts_in_block < 2000:
            print("Too little sparse points in this block, filtered.")
            continue

        pts_block = pts_xyz[mask]
        vmin = np.percentile(pts_block[:, axis_idx], 1)
        vmax = np.percentile(pts_block[:, axis_idx], 99)
        pad = (vmax - vmin) * 0.2
        vmin -= pad
        vmax += pad

        bbx = np.zeros((8, 3), dtype=np.float64)
        bbx[:4, :2] = rect
        bbx[:4, 2] = vmin
        bbx[4:, :2] = rect
        bbx[4:, 2] = vmax

        bbx_expand = np.zeros((8, 3), dtype=np.float64)
        bbx_expand[:4, :2] = rect_expand
        bbx_expand[:4, 2] = vmin
        bbx_expand[4:, :2] = rect_expand
        bbx_expand[4:, 2] = vmax

        if vertical_axis == "x":
            bbx[:, [0, 1, 2]] = bbx[:, [2, 0, 1]]
            bbx_expand[:, [0, 1, 2]] = bbx_expand[:, [2, 0, 1]]
        elif vertical_axis == "y":
            bbx[:, [0, 1, 2]] = bbx[:, [0, 2, 1]]
            bbx_expand[:, [0, 1, 2]] = bbx_expand[:, [0, 2, 1]]

        block_pt_indices = np.nonzero(mask)[0]
        block_pt_set = set(block_pt_indices)

        candidate_ids = []
        for img_id, ptset in image_to_ptidxset.items():
            if not ptset:
                continue
            inter = ptset & block_pt_set
            if not inter:
                continue
            cover_view = len(inter) / max(len(ptset), 1)
            cover_block = len(inter) / max(len(block_pt_set), 1)
            if (cover_view >= cover_ratio_view_thr) or (cover_block >= cover_ratio_block_min):
                candidate_ids.append((img_id, cover_view, cover_block))

        if len(candidate_ids) < min_views_per_block:
            print(f"Too few candidate views ({len(candidate_ids)}) for this block, filtered.")
            continue

        selected_ids = [cid for (cid, _, __) in candidate_ids]
        coverage = 1.0

        print(f"candidates: {len(candidate_ids)}, selected: {len(selected_ids)}, coverage: {coverage:.3f}, num pts in block: {num_pts_in_block}")

        if target_views > 0 and len(selected_ids) > target_views:
            scored = sorted(candidate_ids, key=lambda t: (0.7 * t[2] + 0.3 * t[1]), reverse=True)
            selected_ids = [x[0] for x in scored[:target_views]]
            print(f"[soft budget enabled] capped to target_views={target_views}: selected={len(selected_ids)}")

        block_pcd_path = os.path.join(cfg.output_dirpath, "block_init_pcd", f"block_{valid_block_count:03d}_init_pcd.ply")
        num_init_pts = fetch_local_pcd(cfg.scene_dirpath, block_pcd_path, selected_ids, cfg.scene_scale)

        print(f"Selected, block_id: {valid_block_count}, block sparse pcd num: {num_pts_in_block}, "
              f"initial point num: {num_init_pts}, num views: {len(selected_ids)}")

        plot_rectangle(rect, color=np.random.rand(3))

        blocks_info[valid_block_count] = {
            "bbx": bbx.tolist(),
            "bbx_expand": bbx_expand.tolist(),
            "views_id": list(map(int, selected_ids)),
            "num_views": int(len(selected_ids)),
            "num_pts": int(num_pts_in_block),
            "scene_scale": float(cfg.scene_scale),
            "block_pcd_filepath": block_pcd_path
        }
        valid_block_count += 1

    print(f"Scene partion finished, valid block num: {valid_block_count}")

    blocks_info["num_blocks"] = valid_block_count
    with open(os.path.join(cfg.output_dirpath, "blocks_info.json"), "w") as f:
        json.dump(blocks_info, f, indent=4)


    save_partition_stats(cfg.output_dirpath, blocks_info)

    plt.axis('equal'); plt.axis('off')
    plt.savefig(os.path.join(cfg.output_dirpath, "Partition_Results.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene partition (manual ROI, no camera drop by default)")
    parser.add_argument("--config", "-c", type=str, default="./configs/rubble.yaml", help="config filepath")
    parser.add_argument("--scene_dirpath", "-s", type=str, default=None, help="scene data dirpath")
    parser.add_argument("--output_dirpath", "-o", type=str, default=None, help="optimized result output dirpath")
    args = parser.parse_args()

    cfg = parse_cfg(args)
    os.makedirs(cfg.output_dirpath, exist_ok=True)

    scene_partion(cfg, vertical_axis=cfg.vertical_axis)
