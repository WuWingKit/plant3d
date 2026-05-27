"""分析点云问题：地板残留、花盆缺失、边缘黑色"""
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
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
print(f"X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
print(f"Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

# 分析 Y 坐标分布
print(f"\n=== Y 坐标分布 ===")
y_bins = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
hist, _ = np.histogram(pts[:, 1], bins=y_bins)
for i in range(len(y_bins)-1):
    pct = hist[i] / len(pts) * 100
    if hist[i] > 0:
        mask = (pts[:, 1] >= y_bins[i]) & (pts[:, 1] < y_bins[i+1])
        print(f"  Y={y_bins[i]:3d}~{y_bins[i+1]:3d}mm: {hist[i]:5d} 点 ({pct:4.1f}%)")

# 分析颜色分布（检查黑色边缘）
print(f"\n=== 颜色分析 ===")
print(f"颜色范围: R={colors[:, 0].min():.3f}~{colors[:, 0].max():.3f}, "
      f"G={colors[:, 1].min():.3f}~{colors[:, 1].max():.3f}, "
      f"B={colors[:, 2].min():.3f}~{colors[:, 2].max():.3f}")

# 检查黑色点（颜色接近 0）
black_mask = (colors[:, 0] < 0.1) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
print(f"黑色点 (RGB < 0.1): {black_mask.sum()} ({black_mask.sum()/len(pts)*100:.1f}%)")

# 分析边缘区域
print(f"\n=== 边缘分析 ===")
x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

# 左右边缘
margin = 50  # mm
left_mask = pts[:, 0] < x_min + margin
right_mask = pts[:, 0] > x_max - margin
print(f"左边缘 (X < {x_min + margin:.0f}): {left_mask.sum()} 点")
print(f"右边缘 (X > {x_max - margin:.0f}): {right_mask.sum()} 点")

# 检查边缘点的颜色
if left_mask.any():
    left_colors = colors[left_mask]
    print(f"  左边缘平均颜色: R={left_colors[:, 0].mean():.3f}, "
          f"G={left_colors[:, 1].mean():.3f}, B={left_colors[:, 2].mean():.3f}")
if right_mask.any():
    right_colors = colors[right_mask]
    print(f"  右边缘平均颜色: R={right_colors[:, 0].mean():.3f}, "
          f"G={right_colors[:, 1].mean():.3f}, B={right_colors[:, 2].mean():.3f}")

# 分析最低点（可能是地板残留）
print(f"\n=== 最低点分析 ===")
lowest_y = np.percentile(pts[:, 1], 5)
lowest_mask = pts[:, 1] < lowest_y
lowest_pts = pts[lowest_mask]
lowest_colors = colors[lowest_mask]
print(f"最低 5% 点 (Y < {lowest_y:.1f}mm): {len(lowest_pts)} 点")
print(f"  深度范围: {lowest_pts[:, 2].min():.0f}~{lowest_pts[:, 2].max():.0f}mm")
print(f"  平均颜色: R={lowest_colors[:, 0].mean():.3f}, "
      f"G={lowest_colors[:, 1].mean():.3f}, B={lowest_colors[:, 2].mean():.3f}")

# 分析最高点（检查花盆是否缺失）
print(f"\n=== 最高点分析 ===")
highest_y = np.percentile(pts[:, 1], 95)
highest_mask = pts[:, 1] > highest_y
highest_pts = pts[highest_mask]
print(f"最高 5% 点 (Y > {highest_y:.1f}mm): {len(highest_pts)} 点")
print(f"  深度范围: {highest_pts[:, 2].min():.0f}~{highest_pts[:, 2].max():.0f}mm")
