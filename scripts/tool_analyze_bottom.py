"""分析底部位置"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds"
pcd_file = os.path.join(pcds_dir, "0000.ply")

pcd = o3d.io.read_point_cloud(pcd_file)
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"=== 分析底部位置 ===")
print(f"总点数: {len(pts)}")
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")

# 用户说底部在 Y = 149mm 左右
bottom_y = 149.0
print(f"\n用户指定的底部 Y = {bottom_y}mm")

# 分析 Y = 149mm 附近的区域
margin = 20
mask = (pts[:, 1] >= bottom_y - margin) & (pts[:, 1] <= bottom_y + margin)
if mask.any():
    region_pts = pts[mask]
    region_colors = colors[mask]
    print(f"\nY = {bottom_y-margin} ~ {bottom_y+margin}mm 区域:")
    print(f"  点数: {mask.sum()}")
    print(f"  平均颜色: R={region_colors[:, 0].mean():.2f} G={region_colors[:, 1].mean():.2f} B={region_colors[:, 2].mean():.2f}")

# 分析 Y 坐标分布
print(f"\n=== Y 坐标分布（底部附近）===")
y_bins = np.linspace(bottom_y - 100, bottom_y + 100, 10)
hist, _ = np.histogram(pts[:, 1], bins=y_bins)
for i in range(len(y_bins)-1):
    if hist[i] > 0:
        mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
        avg_color = colors[mask].mean(axis=0)
        print(f"  Y={y_bins[i]:6.0f} ~ {y_bins[i+1]:6.0f}mm: {hist[i]:5d} 点, "
              f"颜色: R={avg_color[0]:.2f} G={avg_color[1]:.2f} B={avg_color[2]:.2f}")

# 找出黑色区域（底座）
print(f"\n=== 黑色区域（底座）===")
black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)
if black_mask.any():
    black_pts = pts[black_mask]
    print(f"黑色点数量: {black_mask.sum()}")
    print(f"黑色点 Y 范围: {black_pts[:, 1].min():.1f} ~ {black_pts[:, 1].max():.1f} mm")

    # 分析黑色点在 Y = 149mm 附近的比例
    black_in_region = (black_pts[:, 1] >= bottom_y - 50) & (black_pts[:, 1] <= bottom_y + 50)
    print(f"Y = {bottom_y-50} ~ {bottom_y+50}mm 区域的黑色点: {black_in_region.sum()}")
