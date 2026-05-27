"""
04_stage1_pcd_v4.py — 使用原始颜色图

使用原始颜色图（1920x1080）而不是对齐图，避免边缘黑色问题
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
from utils_kinect import load_calib, get_intrinsics


def detect_background_walls(depth, K_depth, margin=50):
    """检测背景墙壁"""
    h, w = depth.shape
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        return None

    edge_masks = {
        'top': depth[:margin, :],
        'bottom': depth[-margin:, :],
        'left': depth[:, :margin],
        'right': depth[:, -margin:]
    }

    edge_depths = []
    for region in edge_masks.values():
        valid = region[region > 0]
        if len(valid) > 0:
            edge_depths.extend(valid.tolist())

    if not edge_depths:
        return None

    edge_depths = np.array(edge_depths)
    hist, bin_edges = np.histogram(edge_depths, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peak_idx = np.argmax(hist)
    peak_depth = bin_centers[peak_idx]

    peak_mask = (edge_depths > peak_depth - 200) & (edge_depths < peak_depth + 200)
    if peak_mask.any():
        std = np.std(edge_depths[peak_mask])
    else:
        std = 100

    return {
        "peak_depth": float(peak_depth),
        "std": float(std),
        "suggested_trunc": float(peak_depth - 2 * std),
    }


def map_color_to_depth(depth, color, K_depth, K_color, R, T):
    """
    将原始颜色图映射到深度图坐标系

    参数:
        depth: 深度图 (H_d, W_d)
        color: 原始颜色图 (H_c, W_c, 3)
        K_depth: 深度相机内参 (3, 3)
        K_color: 颜色相机内参 (3, 3)
        R: 旋转矩阵 (3, 3) - 从深度坐标系到颜色坐标系
        T: 平移向量 (3, 1) - 从深度坐标系到颜色坐标系

    返回:
        aligned_color: 对齐后的颜色图 (H_d, W_d, 3)
    """
    h_d, w_d = depth.shape
    aligned_color = np.zeros((h_d, w_d, 3), dtype=np.uint8)

    # 创建深度图的像素坐标网格
    u_d, v_d = np.meshgrid(np.arange(w_d), np.arange(h_d))

    # 将深度图像素坐标转换为深度相机的 3D 坐标
    fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]
    cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]

    Z_d = depth.astype(np.float64)
    X_d = (u_d - cx_d) * Z_d / fx_d
    Y_d = (v_d - cy_d) * Z_d / fy_d

    # 将深度坐标系的 3D 点转换到颜色坐标系
    # P_color = R * P_depth + T
    points_depth = np.stack([X_d, Y_d, Z_d], axis=-1)  # (H_d, W_d, 3)
    points_depth_flat = points_depth.reshape(-1, 3)  # (H_d*W_d, 3)

    # 应用旋转和平移
    points_color = (R @ points_depth_flat.T + T).T  # (H_d*W_d, 3)

    # 将颜色坐标系的 3D 点投影到颜色图像平面
    fx_c, fy_c = K_color[0, 0], K_color[1, 1]
    cx_c, cy_c = K_color[0, 2], K_color[1, 2]

    X_c, Y_c, Z_c = points_color[:, 0], points_color[:, 1], points_color[:, 2]

    # 避免除以零
    Z_c_safe = np.where(Z_c > 0, Z_c, 1.0)

    u_c = (fx_c * X_c / Z_c_safe + cx_c).astype(np.int64)
    v_c = (fy_c * Y_c / Z_c_safe + cy_c).astype(np.int64)

    # 检查投影是否在颜色图像范围内
    h_c, w_c = color.shape[:2]
    valid_mask = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c) & (Z_c > 0)

    # 将颜色映射到深度图坐标系
    aligned_flat = aligned_color.reshape(-1, 3)
    color_flat = color.reshape(-1, 3)

    # 获取有效的颜色坐标
    valid_indices = np.where(valid_mask)[0]
    valid_u = u_c[valid_mask]
    valid_v = v_c[valid_mask]

    # 映射颜色
    color_indices = valid_v * w_c + valid_u
    aligned_flat[valid_indices] = color_flat[color_indices]

    return aligned_color


def make_colored_pcd_v4(depth_path, color_path, K_depth, K_color, R, T,
                        depth_trunc=None, z_min=500.0, y_floor=30.0,
                        auto_detect=True):
    """使用原始颜色图生成彩色点云"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path)
    if depth is None or color is None:
        return None, "图像读取失败"
    if depth.dtype != np.uint16:
        return None, f"深度图不是 uint16 (实际 {depth.dtype})"

    # 自动检测背景墙壁
    bg_analysis = None
    if auto_detect and depth_trunc is None:
        bg_analysis = detect_background_walls(depth, K_depth)
        if bg_analysis:
            depth_trunc = bg_analysis["suggested_trunc"] + 100
        else:
            depth_trunc = 2200.0

    if depth_trunc is None:
        depth_trunc = 2200.0

    # 将颜色图映射到深度图坐标系
    aligned_color = map_color_to_depth(depth, color, K_depth, K_color, R, T)

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # 过滤条件
    mask = (Z > z_min) & (Z < depth_trunc) & (Y > y_floor)

    n_valid = int(np.count_nonzero(mask))
    if n_valid < 100:
        return None, f"有效点太少 ({n_valid})"

    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    bgr = aligned_color[mask]
    rgb = bgr[:, ::-1] / 255.0  # BGR → RGB, 0-1

    # 检查并修复黑色点
    black_mask = (rgb[:, 0] == 0) & (rgb[:, 1] == 0) & (rgb[:, 2] == 0)
    if black_mask.any():
        # 使用周围点的平均颜色，或者使用灰色
        rgb[black_mask] = [0.5, 0.5, 0.5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    stats = {
        "n_points": n_valid,
        "n_black_fixed": int(black_mask.sum()),
        "depth_min_mm": float(Z[mask].min()),
        "depth_max_mm": float(Z[mask].max()),
        "depth_median_mm": float(np.median(Z[mask])),
        "valid_ratio": n_valid / (h * w),
        "y_min_mm": float(Y[mask].min()),
        "y_max_mm": float(Y[mask].max()),
        "y_floor_threshold": float(y_floor),
        "depth_trunc_used": float(depth_trunc),
        "bg_analysis": bg_analysis,
    }
    return (pcd, stats), None


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 使用原始颜色图")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--calib", required=True, help="calibration.json")
    parser.add_argument("--n", type=int, default=36, help="均匀抽取 N 帧")
    parser.add_argument("--every", type=int, default=0, help="或每 N 帧取一帧")
    parser.add_argument("--all", action="store_true", help="生成所有帧")
    parser.add_argument("--depth-trunc", type=float, default=None,
                        help="深度截断 mm（不指定则自动检测+100）")
    parser.add_argument("--z-min", type=float, default=500.0, help="最小深度 mm")
    parser.add_argument("--y-floor", type=float, default=30.0,
                        help="地板 Y 坐标阈值 mm（默认 30）")
    parser.add_argument("--no-auto", action="store_true", help="禁用自动检测")
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 1: 使用原始颜色图")
    print("=" * 64)

    # 加载标定
    calib = load_calib(args.calib)
    K_d, _, _ = get_intrinsics(calib, "depth")
    K_c, _, _ = get_intrinsics(calib, "color")

    # 获取立体标定参数
    R = np.array(calib.get("R", np.eye(3)))
    T = np.array(calib.get("T", np.zeros(3))).reshape(3, 1)

    print(f"深度内参: fx={K_d[0,0]:.2f} fy={K_d[1,1]:.2f} "
          f"cx={K_d[0,2]:.2f} cy={K_d[1,2]:.2f}")
    print(f"颜色内参: fx={K_c[0,0]:.2f} fy={K_c[1,1]:.2f} "
          f"cx={K_c[0,2]:.2f} cy={K_c[1,2]:.2f}")

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
    print(f"深度范围: {args.z_min:.0f} ~ {'自动检测+100' if args.depth_trunc is None else f'{args.depth_trunc:.0f}'} mm")
    print(f"地板阈值: Y > {args.y_floor:.0f} mm")
    print(f"颜色来源: 原始颜色图（1920x1080）")
    print()

    pcds_dir = os.path.join(args.input, "pcds")
    os.makedirs(pcds_dir, exist_ok=True)

    index = []
    failed = []
    t0 = time.time()
    for k, fid in enumerate(sel_ids):
        d_path = os.path.join(depth_dir, f"{fid:04d}.png")
        c_path = os.path.join(color_dir, f"{fid:04d}.png")

        result, err = make_colored_pcd_v4(
            d_path, c_path, K_d, K_c, R, T,
            depth_trunc=args.depth_trunc,
            z_min=args.z_min,
            y_floor=args.y_floor,
            auto_detect=not args.no_auto
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
                  f"黑色修复 {stats['n_black_fixed']:4,}  "
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
            "K_color": K_c.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {index_path}")

    if index:
        n_pts = [e["stats"]["n_points"] for e in index]
        n_black = [e["stats"]["n_black_fixed"] for e in index]
        print()
        print("统计:")
        print(f"  点数: 中位 {int(np.median(n_pts)):,}  "
              f"min {min(n_pts):,}  max {max(n_pts):,}")
        print(f"  黑色修复: 中位 {int(np.median(n_black)):,}  "
              f"min {min(n_black):,}  max {max(n_black):,}")

    print()
    print("下一步：")
    print(f"  1. 用 MeshLab/CloudCompare 打开 {pcds_dir}/0000.ply 检查质量")
    print(f"  2. 如果 OK：python 05_stage2_segment.py --input {args.input}")


if __name__ == "__main__":
    main()
