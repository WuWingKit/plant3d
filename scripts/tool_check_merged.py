"""检查合并后的点云"""
import open3d as o3d
import numpy as np
import os

merged_path = "capture_xxx/output_v2/merged_clean.ply"
pcd = o3d.io.read_point_cloud(merged_path)
pts = np.asarray(pcd.points)

print(f"=== 合并后的点云分析 ===")
print(f"总点数: {len(pts)}")
print(f"X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
print(f"Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

# 计算质心
centroid = pts.mean(axis=0)
print(f"\n质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

# 计算点云在 XZ 平面上的分布
xz_pts = pts[:, [0, 2]]
xz_centroid = xz_pts.mean(axis=0)
xz_distances = np.sqrt(np.sum((xz_pts - xz_centroid)**2, axis=1))
print(f"\nXZ 平面上的分布:")
print(f"  质心: ({xz_centroid[0]:.1f}, {xz_centroid[1]:.1f})")
print(f"  平均距离: {xz_distances.mean():.1f}mm")
print(f"  标准差: {xz_distances.std():.1f}mm")
print(f"  最小距离: {xz_distances.min():.1f}mm")
print(f"  最大距离: {xz_distances.max():.1f}mm")

# 分析 XZ 平面上的角度分布
angles = np.arctan2(xz_pts[:, 1] - xz_centroid[1], xz_pts[:, 0] - xz_centroid[0])
angles_deg = np.degrees(angles)
print(f"\nXZ 平面上的角度分布:")
print(f"  范围: {angles_deg.min():.1f}° ~ {angles_deg.max():.1f}°")
print(f"  总跨度: {angles_deg.max() - angles_deg.min():.1f}°")

# 计算点云在 XZ 平面上的主方向
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(xz_pts)
print(f"\nXZ 平面上的主方向:")
print(f"  第一主成分: {pca.components_[0]}")
print(f"  第二主成分: {pca.components_[1]}")
print(f"  解释方差比: {pca.explained_variance_ratio_}")
