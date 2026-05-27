"""分析点云空间分布，确定裁剪范围"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds"
pcd_file = os.path.join(pcds_dir, "0000.ply")

pcd = o3d.io.read_point_cloud(pcd_file)
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"=== 分析 {pcd_file} ===")
print(f"总点数: {len(pts)}")
print(f"X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
print(f"Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

# 分析 Y 坐标分布（从上到下）
print(f"\n=== Y 坐标分布（从上到下）===")
y_bins = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 20)
hist, _ = np.histogram(pts[:, 1], bins=y_bins)
for i in range(len(y_bins)-1):
    pct = hist[i] / len(pts) * 100
    if hist[i] > 0:
        mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
        avg_color = colors[mask].mean(axis=0)
        print(f"  Y={y_bins[i]:6.0f}~{y_bins[i+1]:6.0f}mm: {hist[i]:5d} 点 ({pct:4.1f}%), "
              f"平均颜色: R={avg_color[0]:.2f} G={avg_color[1]:.2f} B={avg_color[2]:.2f}")

# 分析颜色分布，找出黑色区域（转盘）
print(f"\n=== 黑色区域分析（转盘）===")
# 黑色点：RGB 都小于 0.2
black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)
if black_mask.any():
    black_pts = pts[black_mask]
    print(f"黑色点数: {black_mask.sum()} ({black_mask.sum()/len(pts)*100:.1f}%)")
    print(f"黑色区域 Y 范围: {black_pts[:, 1].min():.0f} ~ {black_pts[:, 1].max():.0f} mm")
    print(f"黑色区域 Z 范围: {black_pts[:, 2].min():.0f} ~ {black_pts[:, 2].max():.0f} mm")

    # 分析黑色区域的 Y 分布
    y_black_bins = np.linspace(black_pts[:, 1].min(), black_pts[:, 1].max(), 10)
    hist_black, _ = np.histogram(black_pts[:, 1], bins=y_black_bins)
    print(f"\n黑色区域 Y 分布:")
    for i in range(len(y_black_bins)-1):
        if hist_black[i] > 0:
            print(f"  Y={y_black_bins[i]:6.0f}~{y_black_bins[i+1]:6.0f}mm: {hist_black[i]:5d} 点")

# 分析绿色区域（植物）
print(f"\n=== 绿色区域分析（植物）===")
# 绿色点：G > R 且 G > B
green_mask = (colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]) & (colors[:, 1] > 0.3)
if green_mask.any():
    green_pts = pts[green_mask]
    print(f"绿色点数: {green_mask.sum()} ({green_mask.sum()/len(pts)*100:.1f}%)")
    print(f"绿色区域 Y 范围: {green_pts[:, 1].min():.0f} ~ {green_pts[:, 1].max():.0f} mm")

# 分析白色区域（花盆）
print(f"\n=== 白色区域分析（花盆）===")
# 白色点：RGB 都大于 0.7
white_mask = (colors[:, 0] > 0.7) & (colors[:, 1] > 0.7) & (colors[:, 2] > 0.7)
if white_mask.any():
    white_pts = pts[white_mask]
    print(f"白色点数: {white_mask.sum()} ({white_mask.sum()/len(pts)*100:.1f}%)")
    print(f"白色区域 Y 范围: {white_pts[:, 1].min():.0f} ~ {white_pts[:, 1].max():.0f} mm")
