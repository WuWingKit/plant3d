"""重新分析原始点云"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds"
pcd_file = os.path.join(pcds_dir, "0000.ply")

pcd = o3d.io.read_point_cloud(pcd_file)
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"=== 原始点云分析 ===")
print(f"总点数: {len(pts)}")
print(f"Y 范围: {pts[:, 1].min():.0f} ~ {pts[:, 1].max():.0f} mm")

# 分析 Y 坐标分布
print(f"\n=== Y 坐标分布 ===")
y_bins = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 15)
hist, _ = np.histogram(pts[:, 1], bins=y_bins)
for i in range(len(y_bins)-1):
    if hist[i] > 0:
        mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
        avg_color = colors[mask].mean(axis=0)
        # 判断颜色类型
        r, g, b = avg_color
        if r < 0.2 and g < 0.2 and b < 0.2:
            color_type = "黑色"
        elif g > r and g > b and g > 0.3:
            color_type = "绿色"
        elif r > 0.7 and g > 0.7 and b > 0.7:
            color_type = "白色"
        elif b > r and b > g and b > 0.3:
            color_type = "蓝色"
        else:
            color_type = "其他"
        print(f"  Y={y_bins[i]:6.0f}~{y_bins[i+1]:6.0f}mm: {hist[i]:5d} 点, "
              f"颜色: R={r:.2f} G={g:.2f} B={b:.2f} ({color_type})")

# 分析绿色点（植物）
print(f"\n=== 绿色点分析 ===")
green_mask = (colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]) & (colors[:, 1] > 0.2)
if green_mask.any():
    green_pts = pts[green_mask]
    print(f"绿色点数: {green_mask.sum()} ({green_mask.sum()/len(pts)*100:.1f}%)")
    print(f"绿色区域 Y 范围: {green_pts[:, 1].min():.0f} ~ {green_pts[:, 1].max():.0f} mm")
else:
    print("没有找到绿色点")

# 分析黑色点（转盘）
print(f"\n=== 黑色点分析 ===")
black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)
if black_mask.any():
    black_pts = pts[black_mask]
    print(f"黑色点数: {black_mask.sum()} ({black_mask.sum()/len(pts)*100:.1f}%)")
    print(f"黑色区域 Y 范围: {black_pts[:, 1].min():.0f} ~ {black_pts[:, 1].max():.0f} mm")
else:
    print("没有找到黑色点")
