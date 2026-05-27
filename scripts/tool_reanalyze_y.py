"""重新分析 Y 轴方向"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:5]

print(f"=== 重新分析 Y 轴方向 ===")
print(f"分析前 5 帧")

for i in range(min(5, len(pcd_files))):
    pcd = o3d.io.read_point_cloud(pcd_files[i])
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    print(f"\n帧 {i}:")
    print(f"  点数: {len(pts)}")
    print(f"  Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")

    # 分析 Y 坐标和颜色的关系
    if colors is not None:
        # 找出绿色点（植物）
        green_mask = (colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]) & (colors[:, 1] > 0.3)
        if green_mask.any():
            green_y = pts[green_mask, 1]
            print(f"  绿色点（植物）Y 范围: {green_y.min():.1f} ~ {green_y.max():.1f} mm")

        # 找出黑色点（可能是底座或转盘）
        black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)
        if black_mask.any():
            black_y = pts[black_mask, 1]
            print(f"  黑色点 Y 范围: {black_y.min():.1f} ~ {black_y.max():.1f} mm")

        # 找出白色点（花盆）
        white_mask = (colors[:, 0] > 0.7) & (colors[:, 1] > 0.7) & (colors[:, 2] > 0.7)
        if white_mask.any():
            white_y = pts[white_mask, 1]
            print(f"  白色点（花盆）Y 范围: {white_y.min():.1f} ~ {white_y.max():.1f} mm")

    # 分析 Y 坐标分布
    print(f"  Y 坐标分布:")
    y_bins = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 6)
    for j in range(len(y_bins)-1):
        mask = (pts[:, 1] >= y_bins[j]) & (pts[:, 1] < y_bins[j+1])
        if mask.any():
            n_points = mask.sum()
            avg_color = colors[mask].mean(axis=0) if colors is not None else None
            color_str = f"R={avg_color[0]:.2f} G={avg_color[1]:.2f} B={avg_color[2]:.2f}" if avg_color is not None else ""
            print(f"    Y={y_bins[j]:6.0f} ~ {y_bins[j+1]:6.0f}mm: {n_points:5d} 点 {color_str}")
