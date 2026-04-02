#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from plyfile import PlyData

def load_ply_generic(path):
    print(f"[Load] Reading {path}...")
    plydata = PlyData.read(path)
    v = plydata.elements[0]

    def get_prop(prefix, count):
        names = [f"{prefix}_{i}" for i in range(count)]
        valid_names = [n for n in names if n in v.data.dtype.names]
        if len(valid_names) == 0: return None
        return np.stack([v[n] for n in valid_names], axis=1)

    xyz = np.stack((v['x'], v['y'], v['z']), axis=1)
    opacities = v['opacity'][..., np.newaxis] if 'opacity' in v.data.dtype.names else np.ones((xyz.shape[0], 1))
    f_dc = get_prop("f_dc", 3)

    all_props = v.data.dtype.names
    rest_props = [p for p in all_props if p.startswith("f_rest_")]
    f_rest = get_prop("f_rest", len(rest_props))

    scale = get_prop("scale", 3)
    rotation = get_prop("rot", 4)
    return xyz, f_dc, f_rest, opacities, scale, rotation

def adaptive_sh_pruning(f_rest, threshold=0.05):

    print("[Adaptive] Analyzing Texture Complexity...")



    energy = np.mean(np.abs(f_rest), axis=1)


    mask_diffuse = energy < threshold
    count_diffuse = np.sum(mask_diffuse)
    ratio = count_diffuse / f_rest.shape[0]

    print(f"[Adaptive] Diffuse Points (SH=0): {count_diffuse} ({ratio*100:.1f}%)")
    print(f"[Adaptive] Glossy Points  (SH=3): {f_rest.shape[0] - count_diffuse}")





    f_rest[mask_diffuse] = 0.0

    return f_rest

def save_adaptive_ply(path, xyz, f_dc, f_rest, opacities, scale, rotation):
    num_points = xyz.shape[0]


    xyz = xyz.astype(np.float32)
    f_dc = f_dc.astype(np.float16)
    opacities = opacities.astype(np.float16)
    scale = scale.astype(np.float16)
    rotation = rotation.astype(np.float16)


    print("[Quantization] Quantizing SH_Rest to Int8...")

    f_rest_min = np.min(f_rest, axis=0)
    f_rest_max = np.max(f_rest, axis=0)
    f_rest_range = f_rest_max - f_rest_min
    f_rest_range[f_rest_range == 0] = 1e-6

    f_rest_q = ((f_rest - f_rest_min) / f_rest_range * 255.0).astype(np.uint8)


    meta_path = path + ".params.npz"
    np.savez(meta_path, q_min=f_rest_min, q_range=f_rest_range)


    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {num_points}\n"
    header += "property float x\nproperty float y\nproperty float z\n"

    for i in range(f_dc.shape[1]): header += f"property half f_dc_{i}\n"
    for i in range(f_rest.shape[1]): header += f"property uint8 f_rest_{i}\n"
    header += "property half opacity\n"
    for i in range(scale.shape[1]): header += f"property half scale_{i}\n"
    for i in range(rotation.shape[1]): header += f"property half rot_{i}\n"
    header += "end_header\n"

    print(f"[Save] Writing binary data...")
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))

        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        for i in range(3): dtype_list.append((f'f_dc_{i}', 'f2'))
        for i in range(45): dtype_list.append((f'f_rest_{i}', 'u1'))
        dtype_list.append(('opacity', 'f2'))
        for i in range(3): dtype_list.append((f'scale_{i}', 'f2'))
        for i in range(4): dtype_list.append((f'rot_{i}', 'f2'))

        data = np.empty(num_points, dtype=dtype_list)
        data['x'] = xyz[:, 0]; data['y'] = xyz[:, 1]; data['z'] = xyz[:, 2]
        for i in range(3): data[f'f_dc_{i}'] = f_dc[:, i]
        for i in range(45): data[f'f_rest_{i}'] = f_rest_q[:, i]
        data['opacity'] = opacities.squeeze()
        for i in range(3): data[f'scale_{i}'] = scale[:, i]
        for i in range(4): data[f'rot_{i}'] = rotation[:, i]

        data.tofile(f)
    print(f"[Save] Done: {path}")

def main():
    parser = argparse.ArgumentParser(description="Adaptive SH Pruning & Quantization")
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)

    parser.add_argument("--threshold", "-t", type=float, default=0.02)
    args = parser.parse_args()

    xyz, f_dc, f_rest, opacities, scale, rotation = load_ply_generic(args.input)


    scales_prod = np.prod(scale, axis=1)
    mask = (scales_prod <= np.percentile(scales_prod, 99.9))
    xyz, f_dc, f_rest, opacities, scale, rotation =        xyz[mask], f_dc[mask], f_rest[mask], opacities[mask], scale[mask], rotation[mask]

    print(f"[Prune] Points: {xyz.shape[0]}")


    f_rest = adaptive_sh_pruning(f_rest, threshold=args.threshold)


    save_adaptive_ply(args.output, xyz, f_dc, f_rest, opacities, scale, rotation)

if __name__ == "__main__":
    main()
