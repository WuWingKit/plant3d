"""分析深度图分布，找出地面和背景墙壁的深度范围"""
import cv2
import numpy as np
import glob
import os

depth_dir = "capture_xxx/depth"
depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:5]

for f in depth_files:
    depth = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if depth is None:
        continue

    print(f"\n=== {os.path.basename(f)} ===")
    print(f"形状: {depth.shape}, 类型: {depth.dtype}")
    print(f"深度范围: {depth.min()}-{depth.max()} mm")

    # 统计不同深度区间的像素数
    bins = [0, 500, 1000, 1500, 2000, 2200, 2500, 3000, 5000]
    hist, _ = np.histogram(depth[depth > 0], bins=bins)
    total = hist.sum()
    print(f"\n深度分布 (总有效像素: {total}):")
    for i in range(len(bins)-1):
        pct = hist[i] / total * 100
        print(f"  {bins[i]:4d}-{bins[i+1]:4d}mm: {hist[i]:6d} 像素 ({pct:5.1f}%)")

    # 分析 Y 坐标与深度的关系
    h, w = depth.shape
    # 假设 Kinect 内参
    fx, fy = 365.456, 365.456
    cx, cy = 254.878, 205.395

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    Y = (v - cy) * Z / fy

    # 统计不同 Y 区间的深度
    print(f"\nY 坐标分布:")
    y_bins = [-500, -200, 0, 200, 400, 600, 800, 1000, 1500]
    for i in range(len(y_bins)-1):
        mask = (Y >= y_bins[i]) & (Y < y_bins[i+1]) & (Z > 0)
        if mask.any():
            depths = Z[mask]
            print(f"  Y={y_bins[i]:4d}-{y_bins[i+1]:4d}mm: {mask.sum():6d} 像素, "
                  f"深度中位数: {np.median(depths):.0f}mm, "
                  f"范围: {depths.min():.0f}-{depths.max():.0f}mm")
