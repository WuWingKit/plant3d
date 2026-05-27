"""检查原始点云中相邻帧的旋转"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:20]

print(f"=== 检查原始点云旋转 ===")
print(f"分析前 20 帧")

# 分析第一帧和最后一帧
pcd_first = o3d.io.read_point_cloud(pcd_files[0])
pcd_last = o3d.io.read_point_cloud(pcd_files[-1])

pts_first = np.asarray(pcd_first.points)
pts_last = np.asarray(pcd_last.points)

print(f"\n第一帧:")
print(f"  点数: {len(pts_first)}")
print(f"  X 范围: {pts_first[:, 0].min():.1f} ~ {pts_first[:, 0].max():.1f} mm")
print(f"  Y 范围: {pts_first[:, 1].min():.1f} ~ {pts_first[:, 1].max():.1f} mm")
print(f"  Z 范围: {pts_first[:, 2].min():.1f} ~ {pts_first[:, 2].max():.1f} mm")

print(f"\n最后一帧:")
print(f"  点数: {len(pts_last)}")
print(f"  X 范围: {pts_last[:, 0].min():.1f} ~ {pts_last[:, 0].max():.1f} mm")
print(f"  Y 范围: {pts_last[:, 1].min():.1f} ~ {pts_last[:, 1].max():.1f} mm")
print(f"  Z 范围: {pts_last[:, 2].min():.1f} ~ {pts_last[:, 2].max():.1f} mm")

# 计算质心
centroid_first = pts_first.mean(axis=0)
centroid_last = pts_last.mean(axis=0)

print(f"\n质心:")
print(f"  第一帧: ({centroid_first[0]:.1f}, {centroid_first[1]:.1f}, {centroid_first[2]:.1f})")
print(f"  最后帧: ({centroid_last[0]:.1f}, {centroid_last[1]:.1f}, {centroid_last[2]:.1f})")
print(f"  距离: {np.linalg.norm(centroid_last - centroid_first):.1f}mm")

# 计算 XZ 平面上的角度
angle_first = np.degrees(np.arctan2(centroid_first[2], centroid_first[0]))
angle_last = np.degrees(np.arctan2(centroid_last[2], centroid_last[0]))
angle_diff = angle_last - angle_first

print(f"\nXZ 平面角度:")
print(f"  第一帧: {angle_first:.1f}°")
print(f"  最后帧: {angle_last:.1f}°")
print(f"  差值: {angle_diff:.1f}°")

# 分析相邻帧的质心变化
print(f"\n相邻帧质心变化（每 10 帧）:")
for i in range(0, min(20, len(pcd_files)) - 1, 10):
    pcd1 = o3d.io.read_point_cloud(pcd_files[i])
    pcd2 = o3d.io.read_point_cloud(pcd_files[i + 1])
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    centroid1 = pts1.mean(axis=0)
    centroid2 = pts2.mean(axis=0)
    dist = np.linalg.norm(centroid2 - centroid1)
    angle1 = np.degrees(np.arctan2(centroid1[2], centroid1[0]))
    angle2 = np.degrees(np.arctan2(centroid2[2], centroid2[0]))
    angle_diff = angle2 - angle1
    print(f"  帧 {i:3d} -> {i+1:3d}: 距离={dist:.1f}mm, 角度差={angle_diff:.2f}°")
