"""分析完整质心轨迹"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))

print(f"=== 完整质心轨迹分析 ===")
print(f"总帧数: {len(pcd_files)}")

# 计算所有帧的质心
centroids = []
for f in pcd_files:
    pcd = o3d.io.read_point_cloud(f)
    pts = np.asarray(pcd.points)
    centroids.append(pts.mean(axis=0))

centroids = np.array(centroids)

print(f"\n质心范围:")
print(f"  X: {centroids[:, 0].min():.1f} ~ {centroids[:, 0].max():.1f} mm")
print(f"  Y: {centroids[:, 1].min():.1f} ~ {centroids[:, 1].max():.1f} mm")
print(f"  Z: {centroids[:, 2].min():.1f} ~ {centroids[:, 2].max():.1f} mm")

# 计算 XZ 平面上的角度
angles = np.arctan2(centroids[:, 2], centroids[:, 0])
angles_deg = np.degrees(angles)

print(f"\nXZ 平面角度:")
print(f"  范围: {angles_deg.min():.1f}° ~ {angles_deg.max():.1f}°")
print(f"  总跨度: {angles_deg.max() - angles_deg.min():.1f}°")

# 计算角度变化
print(f"\n角度变化（每 50 帧）:")
for i in range(0, len(angles_deg), 50):
    if i < len(angles_deg) - 1:
        angle_diff = angles_deg[i+1] - angles_deg[i]
        print(f"  帧 {i:3d} -> {i+1:3d}: {angle_diff:.2f}°")

# 计算总角度变化
total_angle = angles_deg[-1] - angles_deg[0]
print(f"\n总角度变化: {total_angle:.1f}°")

# 分析质心在 XZ 平面上的轨迹
print(f"\n质心在 XZ 平面上的轨迹（每 50 帧）:")
for i in range(0, len(centroids), 50):
    x, y, z = centroids[i]
    angle = angles_deg[i]
    print(f"  帧 {i:3d}: ({x:.1f}, {z:.1f}) 角度={angle:.1f}°")

# 计算质心在 XZ 平面上的距离（相对于原点）
distances = np.sqrt(centroids[:, 0]**2 + centroids[:, 2]**2)
print(f"\n质心到原点的距离:")
print(f"  范围: {distances.min():.1f} ~ {distances.max():.1f} mm")
print(f"  平均: {distances.mean():.1f} mm")

# 分析质心轨迹是否是圆形
print(f"\n质心轨迹分析:")
print(f"  如果转盘旋转轴在原点，质心轨迹应该是圆形")
print(f"  实际质心到原点的距离范围: {distances.min():.1f} ~ {distances.max():.1f} mm")
print(f"  距离变化: {distances.max() - distances.min():.1f} mm")
