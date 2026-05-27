"""分析相邻帧之间的旋转"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:10]

print(f"=== 相邻帧旋转分析 ===")
print(f"分析前 10 帧")

for i in range(len(pcd_files) - 1):
    # 加载两帧
    pcd1 = o3d.io.read_point_cloud(pcd_files[i])
    pcd2 = o3d.io.read_point_cloud(pcd_files[i + 1])

    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)

    # 计算质心
    centroid1 = pts1.mean(axis=0)
    centroid2 = pts2.mean(axis=0)

    # 计算质心之间的距离
    dist = np.linalg.norm(centroid2 - centroid1)

    # 计算质心在 XZ 平面上的角度
    angle1 = np.arctan2(centroid1[2], centroid1[0])
    angle2 = np.arctan2(centroid2[2], centroid2[0])
    angle_diff = np.degrees(angle2 - angle1)

    print(f"\n帧 {i:3d} -> {i+1:3d}:")
    print(f"  质心1: ({centroid1[0]:.1f}, {centroid1[1]:.1f}, {centroid1[2]:.1f})")
    print(f"  质心2: ({centroid2[0]:.1f}, {centroid2[1]:.1f}, {centroid2[2]:.1f})")
    print(f"  质心距离: {dist:.1f}mm")
    print(f"  XZ 平面角度差: {angle_diff:.2f}°")
