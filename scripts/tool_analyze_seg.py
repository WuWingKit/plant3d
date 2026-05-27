"""分析裁剪后的点云"""
import open3d as o3d
import numpy as np
import glob
import os

seg_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(seg_dir, "*.ply")))[:5]

for f in pcd_files:
    pcd = o3d.io.read_point_cloud(f)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    print(f"\n=== {os.path.basename(f)} ===")
    print(f"点数: {len(pts)}")
    print(f"X 范围: {pts[:, 0].min():.0f} ~ {pts[:, 0].max():.0f} mm")
    print(f"Y 范围: {pts[:, 1].min():.0f} ~ {pts[:, 1].max():.0f} mm")
    print(f"Z 范围: {pts[:, 2].min():.0f} ~ {pts[:, 2].max():.0f} mm")

    if colors is not None:
        print(f"颜色范围: R={colors[:, 0].min():.2f}~{colors[:, 0].max():.2f}, "
              f"G={colors[:, 1].min():.2f}~{colors[:, 1].max():.2f}, "
              f"B={colors[:, 2].min():.2f}~{colors[:, 2].max():.2f}")

        # 分析颜色分布
        y_bins = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 10)
        print(f"\nY 坐标颜色分布:")
        for i in range(len(y_bins)-1):
            mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
            if mask.any():
                avg_color = colors[mask].mean(axis=0)
                print(f"  Y={y_bins[i]:.0f}~{y_bins[i+1]:.0f}mm: {mask.sum():4d} 点, "
                      f"颜色: R={avg_color[0]:.2f} G={avg_color[1]:.2f} B={avg_color[2]:.2f}")
