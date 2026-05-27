"""
04_stage1_pcd_v5.py — 使用正确的颜色映射函数

使用 utils_kinect.make_aligned_color 进行正确的深度→颜色映射
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import glob
import argparse
import time
import numpy as np

try:
    import cv2
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import load_calib, get_intrinsics, get_stereo, make_aligned_color


def detect_background_walls(depth, margin=50):
    """检测背景墙壁深度"""
    h, w = depth.shape
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        return 2000.0

    # 取边缘区域
    edge_depths = []
    for region in [depth[:margin, :], depth[-margin:, :],
                   depth[:, :margin], depth[:, -margin:]]:
        valid = region[region > 0]
        if len(valid) > 0:
            edge_depths.extend(valid.tolist())

    if not edge_depths:
        return 2000.0

    edge_depths = np.array(edge_depths)
    hist, bin_edges = np.histogram(edge_depths, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_idx = np.argmax(hist)
    peak_depth = bin_centers[peak_idx]

    # 返回峰值 - 2*标准差 作为截断深度
    peak_mask = (edge_depths > peak_depth - 200) & (edge_depths < peak_depth + 200)
    std = np.std(edge_depths[peak_mask]) if peak_mask.any() else 100

    return float(peak_depth - 2 * std)


def make_colored_pcd_v5(depth_path, color_path, K_depth, K_color, dist_color,
                        R, T, depth_trunc=None, z_min=500.0):
    """使用正确的颜色映射生成彩色点云"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path)
    if depth is None or color is None:
        return None, "图像读取失败"
    if depth.dtype != np.uint16:
        return None, f"深度图不是 uint16 (实际 {depth.dtype})"

    # 自动检测背景墙壁深度
    if depth_trunc is None:
        depth_trunc = detect_background_walls(depth)
        depth_trunc = max(depth_trunc, 1500.0)  # 最小 1500mm

    # 使用正确的映射函数生成对齐颜色图
    aligned_color = make_aligned_color(depth, color, K_depth, K_color, dist_color, R, T)

    # 生成点云
    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # 过滤：只去除深度范围外的点，不去除地板
    mask = (Z > z_min) & (Z < depth_trunc)

    n_valid = int(np.count_nonzero(mask))
    if n_valid < 100:
        return None, f"有效点太少 ({n_valid})"

    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    bgr = aligned_color[mask]
    rgb = bgr[:, ::-1] / 255.0  # BGR → RGB

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    stats = {
        "n_points": n_valid,
        "depth_min_mm": float(Z[mask].min()),
        "depth_max_mm": float(Z[mask].max()),
        "depth_median_mm": float(np.median(Z[mask])),
        "y_min_mm": float(Y[mask].min()),
        "y_max_mm": float(Y[mask].max()),
        "depth_trunc_used": float(depth_trunc),
    }
    return (pcd, stats), None


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 正确颜色映射")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--calib", required=True, help="calibration.json")
    parser.add_argument("--n", type=int, default=36, help="均匀抽取 N 帧")
    parser.add_argument("--every", type=int, default=0, help="或每 N 帧取一帧")
    parser.add_argument("--all", action="store_true", help="生成所有帧")
    parser.add_argument("--depth-trunc", type=float, default=None,
                        help="深度截断 mm（不指定则自动检测）")
    parser.add_argument("--z-min", type=float, default=500.0, help="最小深度 mm")
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 1: 正确颜色映射")
    print("=" * 64)

    # 加载标定
    calib = load_calib(args.calib)
    K_d, _, _ = get_intrinsics(calib, "depth")
    K_c, dist_c, _ = get_intrinsics(calib, "color")
    R, T = get_stereo(calib)

    print(f"深度内参: fx={K_d[0,0]:.2f} fy={K_d[1,1]:.2f}")
    print(f"颜色内参: fx={K_c[0,0]:.2f} fy={K_c[1,1]:.2f}")
    print(f"颜色畸变: {dist_c}")

    depth_dir = os.path.join(args.input, "depth")
    color_dir = os.path.join(args.input, "color")
    if not os.path.isdir(color_dir):
        print(f"✗ 缺少 color/ 目录")
        sys.exit(1)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    all_ids = sorted([int(os.path.splitext(os.path.basename(f))[0])
                      for f in depth_files])
    if not all_ids:
        print(f"✗ {depth_dir} 没有 PNG")
        sys.exit(1)

    if args.all:
        sel_ids = all_ids
    elif args.every > 0:
        sel_ids = all_ids[::args.every]
    else:
        if len(all_ids) <= args.n:
            sel_ids = all_ids
        else:
            idx = np.linspace(0, len(all_ids) - 1, args.n, dtype=int)
            sel_ids = [all_ids[i] for i in idx]

    print(f"总帧数: {len(all_ids)}, 选取: {len(sel_ids)} 帧")
    print(f"深度范围: {args.z_min:.0f} ~ {'自动检测' if args.depth_trunc is None else f'{args.depth_trunc:.0f}'} mm")
    print()

    pcds_dir = os.path.join(args.input, "pcds")
    os.makedirs(pcds_dir, exist_ok=True)

    index = []
    failed = []
    t0 = time.time()
    for k, fid in enumerate(sel_ids):
        d_path = os.path.join(depth_dir, f"{fid:04d}.png")
        c_path = os.path.join(color_dir, f"{fid:04d}.png")

        result, err = make_colored_pcd_v5(
            d_path, c_path, K_d, K_c, dist_c, R, T,
            depth_trunc=args.depth_trunc,
            z_min=args.z_min
        )
        if result is None:
            failed.append({"id": fid, "error": err})
            print(f"  ✗ #{fid:04d}  {err}")
            continue
        pcd, stats = result

        out_path = os.path.join(pcds_dir, f"{fid:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)

        index.append({"id": fid, "stats": stats})

        if (k + 1) % 5 == 0 or k == len(sel_ids) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(sel_ids) - k - 1)
            print(f"  [{k+1:3d}/{len(sel_ids)}] #{fid:04d}  "
                  f"点数 {stats['n_points']:6,}  "
                  f"深度截断 {stats['depth_trunc_used']:.0f}mm  "
                  f"ETA {eta:.0f}s")

    print()
    print(f"✓ 完成: {len(index)} 帧成功, {len(failed)} 帧失败  "
          f"耗时 {time.time()-t0:.1f}s")

    index_path = os.path.join(pcds_dir, "pcd_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            "frames": index,
            "failed": failed,
            "K_depth": K_d.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {index_path}")

    if index:
        n_pts = [e["stats"]["n_points"] for e in index]
        print()
        print("统计:")
        print(f"  点数: 中位 {int(np.median(n_pts)):,}  "
              f"min {min(n_pts):,}  max {max(n_pts):,}")

    print()
    print("下一步：")
    print(f"  1. 用 MeshLab/CloudCompare 打开 {pcds_dir}/0000.ply 检查质量")
    print(f"  2. 如果 OK：python 05_stage2_segment.py --input {args.input}")


if __name__ == "__main__":
    main()
