"""
04_stage1_pcd_smart.py — 智能背景去除的单帧点云生成

自动分析深度分布，识别并去除地面和背景墙壁
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


def analyze_depth_distribution(depth, sample_size=5):
    """分析深度分布，找出背景墙壁的深度范围"""
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        return None

    # 计算深度直方图
    hist, bin_edges = np.histogram(valid_depth, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 找出峰值（背景墙壁通常是最大的峰值）
    # 只考虑 1000mm 以上的深度（排除盆栽主体）
    mask = bin_centers > 1000
    if not mask.any():
        return None

    hist_filtered = hist[mask]
    centers_filtered = bin_centers[mask]

    # 找出最大的峰值
    peak_idx = np.argmax(hist_filtered)
    peak_depth = centers_filtered[peak_idx]
    peak_count = hist_filtered[peak_idx]

    # 计算峰值周围的范围（标准差）
    peak_mask = (bin_centers > peak_depth - 200) & (bin_centers < peak_depth + 200)
    if peak_mask.any():
        std = np.std(valid_depth[(valid_depth > peak_depth - 200) & (valid_depth < peak_depth + 200)])
    else:
        std = 100

    return {
        "peak_depth": float(peak_depth),
        "peak_count": int(peak_count),
        "std": float(std),
        "suggested_trunc": float(peak_depth - 2 * std)  # 建议的截断深度
    }


def make_colored_pcd_smart(depth_path, aligned_path, K_depth, depth_trunc=None,
                           z_min=500.0, y_min=-200.0, auto_detect=True):
    """从 depth + aligned RGB 生成彩色点云，智能去除地面和背景"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    aligned = cv2.imread(aligned_path)
    if depth is None or aligned is None:
        return None, "图像读取失败"
    if depth.dtype != np.uint16:
        return None, f"深度图不是 uint16 (实际 {depth.dtype})"

    # 自动检测背景深度
    bg_analysis = None
    if auto_detect and depth_trunc is None:
        bg_analysis = analyze_depth_distribution(depth)
        if bg_analysis:
            depth_trunc = bg_analysis["suggested_trunc"]
            print(f"  自动检测背景深度: {bg_analysis['peak_depth']:.0f}mm, "
                  f"建议截断: {depth_trunc:.0f}mm")
        else:
            depth_trunc = 2000.0  # 默认值

    if depth_trunc is None:
        depth_trunc = 2000.0

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # 过滤：深度范围 + 地板 + 近处杂点
    mask = (Z > z_min) & (Z < depth_trunc) & (Y > y_min)
    n_valid = int(np.count_nonzero(mask))
    if n_valid < 100:
        return None, f"有效点太少 ({n_valid})"

    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    bgr = aligned[mask]
    rgb = bgr[:, ::-1] / 255.0  # BGR → RGB, 0-1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    stats = {
        "n_points": n_valid,
        "depth_min_mm": float(Z[mask].min()),
        "depth_max_mm": float(Z[mask].max()),
        "depth_median_mm": float(np.median(Z[mask])),
        "valid_ratio": n_valid / (h * w),
        "y_min_mm": float(Y[mask].min()),
        "y_max_mm": float(Y[mask].max()),
        "depth_trunc_used": float(depth_trunc),
        "bg_analysis": bg_analysis,
    }
    return (pcd, stats), None


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 智能背景去除的单帧点云生成")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--calib", required=True, help="calibration.json")
    parser.add_argument("--n", type=int, default=36, help="均匀抽取 N 帧（默认 36）")
    parser.add_argument("--every", type=int, default=0, help="或每 N 帧取一帧（覆盖 --n）")
    parser.add_argument("--all", action="store_true", help="生成所有帧（覆盖 --n 和 --every）")
    parser.add_argument("--depth-trunc", type=float, default=None,
                        help="深度截断 mm（不指定则自动检测）")
    parser.add_argument("--z-min", type=float, default=500.0, help="最小深度 mm（默认 500）")
    parser.add_argument("--y-min", type=float, default=-200.0, help="最小 Y 高度 mm（默认 -200）")
    parser.add_argument("--no-auto", action="store_true", help="禁用自动背景检测")
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 1: 智能背景去除的单帧点云生成")
    print("=" * 64)

    # 标定
    calib = load_calib(args.calib)
    K_d, _, _ = get_intrinsics(calib, "depth")
    print(f"IR 内参: fx={K_d[0,0]:.2f} fy={K_d[1,1]:.2f} "
          f"cx={K_d[0,2]:.2f} cy={K_d[1,2]:.2f}")

    # 找帧
    depth_dir = os.path.join(args.input, "depth")
    aligned_dir = os.path.join(args.input, "aligned")
    if not os.path.isdir(aligned_dir):
        print(f"✗ 缺少 aligned/ 目录。要求用 03_capture_scan.py 采集的数据")
        sys.exit(1)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    all_ids = sorted([int(os.path.splitext(os.path.basename(f))[0])
                      for f in depth_files])
    if not all_ids:
        print(f"✗ {depth_dir} 没有 PNG")
        sys.exit(1)

    # 抽帧
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
    print(f"Y 最小值: {args.y_min:.0f} mm（去除地板）")
    print()

    # 输出目录
    pcds_dir = os.path.join(args.input, "pcds")
    os.makedirs(pcds_dir, exist_ok=True)

    # 生成
    index = []
    failed = []
    t0 = time.time()
    for k, fid in enumerate(sel_ids):
        d_path = os.path.join(depth_dir, f"{fid:04d}.png")
        a_path = os.path.join(aligned_dir, f"{fid:04d}.png")

        result, err = make_colored_pcd_smart(
            d_path, a_path, K_d,
            depth_trunc=args.depth_trunc,
            z_min=args.z_min,
            y_min=args.y_min,
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
                  f"深度截断 {stats['depth_trunc_used']:.0f}mm  "
                  f"ETA {eta:.0f}s")

    print()
    print(f"✓ 完成: {len(index)} 帧成功, {len(failed)} 帧失败  "
          f"耗时 {time.time()-t0:.1f}s")

    # 保存索引
    index_path = os.path.join(pcds_dir, "pcd_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            "frames": index,
            "failed": failed,
            "K_depth": K_d.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {index_path}")

    # 总结
    if index:
        n_pts = [e["stats"]["n_points"] for e in index]
        d_trunc = [e["stats"]["depth_trunc_used"] for e in index]
        print()
        print("统计:")
        print(f"  点数: 中位 {int(np.median(n_pts)):,}  "
              f"min {min(n_pts):,}  max {max(n_pts):,}")
        print(f"  深度截断: 中位 {np.median(d_trunc):.0f}mm  "
              f"min {min(d_trunc):.0f}  max {max(d_trunc):.0f}")

    print()
    print("下一步：")
    print(f"  1. 用 MeshLab/CloudCompare 打开 {pcds_dir}/0000.ply 检查质量")
    print(f"  2. 如果 OK：python 05_stage2_segment.py --input {args.input}")


if __name__ == "__main__":
    main()
