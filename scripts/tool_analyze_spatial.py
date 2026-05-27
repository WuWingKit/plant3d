"""分析点云的空间分布"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:10]

print(f"=== 分析点云的空间分布 ===")
print(f"分析前 10 帧")

# 分析每一帧的空间分布
for i in range(min(10, len(pcd_files))):
    pcd = o3d.io.read_point_cloud(pcd_files[i])
    pts = np.asarray(pcd.points)

    print(f"\n帧 {i}:")
    print(f"  点数: {len(pts)}")
    print(f"  X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
    print(f"  Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
    print(f"  Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

    # 计算质心
    centroid = pts.mean(axis=0)
    print(f"  质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    # 计算 XZ 平面上的角度
    angle = np.degrees(np.arctan2(centroid[2], centroid[0]))
    print(f"  XZ 平面角度: {angle:.1f}°")

    # 计算点云在 XZ 平面上的分布
    xz_pts = pts[:, [0, 2]]
    xz_centroid = xz_pts.mean(axis=0)
    xz_distances = np.sqrt(np.sum((xz_pts - xz_centroid)**2, axis=1))
    print(f"  XZ 平面上的分布: 平均距离={xz_distances.mean():.1f}mm, 标准差={xz_distances.std():.1f}mm")
