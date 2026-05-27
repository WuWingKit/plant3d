"""检查点云的空间闭合"""
import open3d as o3d
import numpy as np
import os

merged_path = "capture_xxx/output_v2/merged_clean.ply"
pcd = o3d.io.read_point_cloud(merged_path)
pts = np.asarray(pcd.points)

print(f"=== 点云空间闭合分析 ===")
print(f"总点数: {len(pts)}")
print(f"X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
print(f"Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

# 计算质心
centroid = pts.mean(axis=0)
print(f"\n质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

# 计算 XZ 平面上的分布
xz_pts = pts[:, [0, 2]]
xz_centroid = xz_pts.mean(axis=0)
xz_distances = np.sqrt(np.sum((xz_pts - xz_centroid)**2, axis=1))
print(f"\nXZ 平面上的分布:")
print(f"  质心: ({xz_centroid[0]:.1f}, {xz_centroid[1]:.1f})")
print(f"  平均距离: {xz_distances.mean():.1f}mm")
print(f"  标准差: {xz_distances.std():.1f}mm")

# 检查是否有空洞
print(f"\n检查是否有空洞:")
# 将 XZ 平面分成 12 个扇区（每 30° 一个）
n_sectors = 12
angles = np.arctan2(xz_pts[:, 1] - xz_centroid[1], xz_pts[:, 0] - xz_centroid[0])
angles_deg = np.degrees(angles)

for i in range(n_sectors):
    angle_min = -180 + i * 30
    angle_max = angle_min + 30
    mask = (angles_deg >= angle_min) & (angles_deg < angle_max)
    n_points = mask.sum()
    if n_points < 1000:
        print(f"  扇区 {angle_min:4.0f}° ~ {angle_max:4.0f}°: {n_points:6d} 点 (可能有空洞)")
    else:
        print(f"  扇区 {angle_min:4.0f}° ~ {angle_max:4.0f}°: {n_points:6d} 点")

# 检查点云是否是旋转体
print(f"\n检查点云是否是旋转体:")
# 计算每个角度的距离
for i in range(n_sectors):
    angle_min = -180 + i * 30
    angle_max = angle_min + 30
    mask = (angles_deg >= angle_min) & (angles_deg < angle_max)
    if mask.any():
        distances = xz_distances[mask]
        print(f"  扇区 {angle_min:4.0f}° ~ {angle_max:4.0f}°: "
              f"平均距离={distances.mean():.1f}mm, 标准差={distances.std():.1f}mm")
