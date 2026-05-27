"""找出黑色底座的位置"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds"
pcd_file = os.path.join(pcds_dir, "0000.ply")

pcd = o3d.io.read_point_cloud(pcd_file)
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"=== 找出黑色底座的位置 ===")
print(f"总点数: {len(pts)}")
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")

# 找出黑色点
black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)
print(f"\n黑色点数量: {black_mask.sum()} ({black_mask.sum()/len(pts)*100:.1f}%)")

if black_mask.any():
    black_pts = pts[black_mask]
    print(f"黑色点 Y 范围: {black_pts[:, 1].min():.1f} ~ {black_pts[:, 1].max():.1f} mm")

    # 分析黑色点的 Y 分布
    print(f"\n黑色点 Y 分布:")
    y_bins = np.linspace(black_pts[:, 1].min(), black_pts[:, 1].max(), 10)
    hist, _ = np.histogram(black_pts[:, 1], bins=y_bins)
    for i in range(len(y_bins)-1):
        if hist[i] > 0:
            print(f"  Y={y_bins[i]:6.0f} ~ {y_bins[i+1]:6.0f}mm: {hist[i]:5d} 点")

# 找出绿色点（植物）
green_mask = (colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]) & (colors[:, 1] > 0.3)
if green_mask.any():
    green_pts = pts[green_mask]
    print(f"\n绿色点（植物）Y 范围: {green_pts[:, 1].min():.1f} ~ {green_pts[:, 1].max():.1f} mm")

# 找出白色点（花盆）
white_mask = (colors[:, 0] > 0.7) & (colors[:, 1] > 0.7) & (colors[:, 2] > 0.7)
if white_mask.any():
    white_pts = pts[white_mask]
    print(f"\n白色点（花盆）Y 范围: {white_pts[:, 1].min():.1f} ~ {white_pts[:, 1].max():.1f} mm")

# 分析 Y 坐标分布
print(f"\n=== Y 坐标分布 ===")
y_bins = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 15)
hist, _ = np.histogram(pts[:, 1], bins=y_bins)
for i in range(len(y_bins)-1):
    if hist[i] > 0:
        mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
        avg_color = colors[mask].mean(axis=0)
        print(f"  Y={y_bins[i]:6.0f} ~ {y_bins[i+1]:6.0f}mm: {hist[i]:5d} 点, "
              f"颜色: R={avg_color[0]:.2f} G={avg_color[1]:.2f} B={avg_color[2]:.2f}")
