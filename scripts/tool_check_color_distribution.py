"""检查点云的颜色分布"""
import open3d as o3d
import numpy as np
import os

merged_path = "capture_xxx/output_v2/merged_color_coded.ply"
pcd = o3d.io.read_point_cloud(merged_path)
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"=== 点云颜色分布分析 ===")
print(f"总点数: {len(pts)}")

# 计算 XZ 平面上的角度
xz_pts = pts[:, [0, 2]]
xz_centroid = xz_pts.mean(axis=0)
angles = np.arctan2(xz_pts[:, 1] - xz_centroid[1], xz_pts[:, 0] - xz_centroid[0])
angles_deg = np.degrees(angles)

print(f"\nXZ 平面上的角度分布:")
print(f"  范围: {angles_deg.min():.1f}° ~ {angles_deg.max():.1f}°")
print(f"  总跨度: {angles_deg.max() - angles_deg.min():.1f}°")

# 分析每个角度区间的颜色
print(f"\n每个角度区间的颜色:")
n_bins = 12  # 每 30° 一个区间
bin_edges = np.linspace(-180, 180, n_bins + 1)
for i in range(n_bins):
    mask = (angles_deg >= bin_edges[i]) & (angles_deg < bin_edges[i+1])
    if mask.any():
        avg_color = colors[mask].mean(axis=0)
        n_points = mask.sum()
        print(f"  {bin_edges[i]:6.1f}° ~ {bin_edges[i+1]:6.1f}°: "
              f"{n_points:6d} 点, 颜色: R={avg_color[0]:.2f} G={avg_color[1]:.2f} B={avg_color[2]:.2f}")

# 检查是否有明显的颜色边界（说明配准有问题）
print(f"\n检查颜色边界:")
for i in range(n_bins):
    mask1 = (angles_deg >= bin_edges[i]) & (angles_deg < bin_edges[i+1])
    mask2 = (angles_deg >= bin_edges[(i+1) % n_bins]) & (angles_deg < bin_edges[(i+2) % n_bins])
    if mask1.any() and mask2.any():
        avg_color1 = colors[mask1].mean(axis=0)
        avg_color2 = colors[mask2].mean(axis=0)
        color_diff = np.linalg.norm(avg_color1 - avg_color2)
        if color_diff > 0.3:
            print(f"  角度 {bin_edges[i]:.1f}° 和 {bin_edges[(i+1) % n_bins]:.1f}° 之间颜色差异大: {color_diff:.2f}")
