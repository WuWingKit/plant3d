"""分析点云中地板的分布"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:3]

for f in pcd_files:
    pcd = o3d.io.read_point_cloud(f)
    pts = np.asarray(pcd.points)

    print(f"\n=== {os.path.basename(f)} ===")
    print(f"总点数: {len(pts)}")
    print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")

    # 分析 Y 坐标分布
    y_bins = [-200, -100, -50, 0, 50, 100, 200, 400, 600, 800, 1000]
    hist, _ = np.histogram(pts[:, 1], bins=y_bins)
    total = len(pts)

    print(f"\nY 坐标分布:")
    for i in range(len(y_bins)-1):
        pct = hist[i] / total * 100
        if hist[i] > 0:
            # 计算该区域的深度范围
            mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
            depths = pts[mask, 2]
            print(f"  Y={y_bins[i]:4d}~{y_bins[i+1]:4d}mm: {hist[i]:6d} 点 ({pct:5.1f}%), "
                  f"深度: {depths.min():.0f}~{depths.max():.0f}mm, 中位: {np.median(depths):.0f}mm")

    # 分析最低点的特征
    lowest_y = np.percentile(pts[:, 1], 5)
    lowest_mask = pts[:, 1] < lowest_y
    lowest_pts = pts[lowest_mask]

    print(f"\n最低 5% 点 (Y < {lowest_y:.1f}mm):")
    print(f"  点数: {len(lowest_pts)}")
    print(f"  深度范围: {lowest_pts[:, 2].min():.0f}~{lowest_pts[:, 2].max():.0f}mm")
    print(f"  X 范围: {lowest_pts[:, 0].min():.0f}~{lowest_pts[:, 0].max():.0f}mm")
