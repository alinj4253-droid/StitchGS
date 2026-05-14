import os
import json
import yaml
import argparse
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R_tool
from scipy.spatial import cKDTree
import random

def update_rots(raw_rots, T):
    r_align = T[:3, :3]
    q_align = R_tool.from_matrix(r_align).as_quat()
    q_old = R_tool.from_quat(np.stack([raw_rots[:,1], raw_rots[:,2], raw_rots[:,3], raw_rots[:,0]], axis=1))
    q_new = R_tool.from_quat(q_align) * q_old
    res = q_new.as_quat()
    return np.stack([res[:,3], res[:,0], res[:,1], res[:,2]], axis=1).astype(np.float32)

def get_obb_params_cuda(bbx):
    corners = np.array(bbx)
    center = torch.tensor(np.mean(corners, axis=0), device='cuda', dtype=torch.float32)
    vec_x = torch.tensor(corners[1] - corners[0], device='cuda', dtype=torch.float32)
    vec_y = torch.tensor(corners[4] - corners[0], device='cuda', dtype=torch.float32)
    vec_z = torch.tensor(corners[3] - corners[0], device='cuda', dtype=torch.float32)
    extents = torch.tensor([vec_x.norm()/2, vec_y.norm()/2, vec_z.norm()/2], device='cuda')
    rot_mat = torch.stack([vec_x/vec_x.norm(), vec_y/vec_y.norm(), vec_z/vec_z.norm()], dim=1)
    return center, extents, rot_mat

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None, help="config file; infers --optimized_path when -o is omitted")
    parser.add_argument("--optimized_path", "-o", type=str, default=None)
    parser.add_argument("--enable_icp", action="store_true")
    parser.add_argument("--sharpness", type=float, default=30.0)
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.optimized_path is None:
        if args.config is None:
            parser.error("provide --config / -c or --optimized_path / -o")
        with open(args.config) as f:
            args.optimized_path = yaml.safe_load(f)["output_dirpath"]

    set_seed(args.seed)

    opt_dir = Path(args.optimized_path)
    with open(opt_dir / "blocks_info.json", 'r') as f:
        info = json.load(f)
    num_blocks = info['num_blocks']

    processed_blocks_dir = opt_dir / f"processed_blocks_seed{args.seed}"
    processed_blocks_dir.mkdir(exist_ok=True)

    blocks = []
    print(f"[Merge] Seed={args.seed} | Loading blocks (k={args.k_neighbors})...")
    for i in range(num_blocks):
        p = sorted(list((opt_dir / "point_cloud" / str(i)).glob("*.ply")))[-1]
        ply = PlyData.read(p)
        v = ply['vertex']
        xyz = np.stack((v['x'], v['y'], v['z']), axis=1).astype(np.float32)
        opac = v['opacity'].astype(np.float32)
        scales = np.exp(np.stack([v[f'scale_{k}'] for k in range(3)], axis=1).astype(np.float32))
        quality = torch.tensor(opac.squeeze() / (np.mean(scales, axis=1) + 1e-6), device='cuda')

        blocks.append({
            'xyz': xyz,
            'quality': quality,
            'raw': v.data,
            'rot': np.stack([v[f'rot_{k}'] for k in range(4)], axis=1).astype(np.float32),
            'bbx': info[str(i)]['bbx']
        })

    print(f"[Merge] Building KDTree for {num_blocks} blocks...")
    trees = [cKDTree(b['xyz']) for b in blocks]
    print("[Merge] KDTree ready.")

    if args.enable_icp:
        for i in range(1, num_blocks):
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(blocks[i]['xyz'][::10])
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(blocks[0]['xyz'][::10])
            reg = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, 0.1, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            if reg.fitness > 0.05:
                blocks[i]['xyz'] = (np.dot(blocks[i]['xyz'], reg.transformation[:3, :3].T) + reg.transformation[:3, 3]).astype(np.float32)
                blocks[i]['rot'] = update_rots(blocks[i]['rot'], reg.transformation)

    obb_params = [get_obb_params_cuda(b['bbx']) for b in blocks]
    final_vertices = []

    for i in range(num_blocks):
        print(f"Processing Block {i}/{num_blocks}...")
        xyz_t = torch.tensor(blocks[i]['xyz'], device='cuda')
        qual_t = blocks[i]['quality']
        center_i, extent_i, rot_i = obb_params[i]

        with torch.no_grad():
            d_self = torch.max(torch.abs(torch.matmul(xyz_t - center_i, rot_i)) / (extent_i + 1e-6), dim=1)[0]
            mask_keep = (d_self <= 1.0)

            idx_exp = torch.nonzero((d_self > 1.0) & (d_self <= 1.1)).squeeze()

            if idx_exp.numel() > 0:
                margin_points_cuda = xyz_t[idx_exp]
                margin_points_cpu = margin_points_cuda.cpu().numpy()

                my_score = (1.1 - d_self[idx_exp]) * qual_t[idx_exp]

                max_neighbor_score = torch.zeros_like(my_score)

                for j in range(num_blocks):
                    if i == j:
                        continue

                    c_j, e_j, r_j = obb_params[j]
                    d_j = torch.max(torch.abs(torch.matmul(margin_points_cuda - c_j, r_j)) / (e_j + 1e-6), dim=1)[0]


                    _, neighbor_indices = trees[j].query(margin_points_cpu, k=args.k_neighbors, workers=1)

                    neighbor_indices_cuda = torch.from_numpy(neighbor_indices).cuda()
                    neighbor_quals_k = blocks[j]['quality'][neighbor_indices_cuda]

                    if args.k_neighbors > 1:
                        neighbor_qual_avg = torch.mean(neighbor_quals_k, dim=1)
                    else:
                        neighbor_qual_avg = neighbor_quals_k

                    n_score = (1.1 - d_j).clamp(min=0) * neighbor_qual_avg
                    max_neighbor_score = torch.max(max_neighbor_score, n_score)

                diff = my_score - max_neighbor_score
                prob = torch.sigmoid(diff * args.sharpness)
                mask_keep[idx_exp] = torch.rand_like(prob) < prob

        keep_idx = torch.nonzero(mask_keep).squeeze().cpu().numpy()
        v_data = blocks[i]['raw'].copy()
        v_data['x'] = blocks[i]['xyz'][:, 0]
        v_data['y'] = blocks[i]['xyz'][:, 1]
        v_data['z'] = blocks[i]['xyz'][:, 2]
        for k in range(4):
            v_data[f'rot_{k}'] = blocks[i]['rot'][:, k]

        kept_vertices = v_data[keep_idx]
        final_vertices.append(kept_vertices)

        block_out_path = processed_blocks_dir / f"block_{i}_processed.ply"
        PlyData([PlyElement.describe(kept_vertices, 'vertex')], text=False).write(str(block_out_path))
        print(f"  -> Saved Block {i} to {block_out_path} (Points: {len(kept_vertices)})")

    merged = np.concatenate(final_vertices, axis=0)
    out_path = opt_dir / f"point_cloud_merged_seed{args.seed}.ply"
    PlyData([PlyElement.describe(merged, 'vertex')], text=False).write(str(out_path))
    print(f"\n[Success] Merge done. Seed={args.seed} | Points: {len(merged)}")
    print(f"[Success] Merged PLY: {out_path}")
    print(f"[Success] Processed blocks dir: {processed_blocks_dir}")

if __name__ == "__main__":
    main()
