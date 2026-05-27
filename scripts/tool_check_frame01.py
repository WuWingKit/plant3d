"""检查帧 0 和帧 1 的点云"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:5]

print(f"=== 检查帧 0 和帧 1 的点云 ===")

# 加载帧 0 和帧 1
pcd0 = o3d.io.read_point_cloud(pcd_files[0])
pcd1 = o3d.io.read_point_cloud(pcd_files[1])

pts0 = np.asarray(pcd0.points)
pts1 = np.asarray(pcd1.points)

print(f"\n帧 0:")
print(f"  点数: {len(pts0)}")
print(f"  X 范围: {pts0[:, 0].min():.1f} ~ {pts0[:, 0].max():.1f} mm")
print(f"  Y 范围: {pts0[:, 1].min():.1f} ~ {pts0[:, 1].max():.1f} mm")
print(f"  Z 范围: {pts0[:, 2].min():.1f} ~ {pts0[:, 2].max():.1f} mm")

print(f"\n帧 1:")
print(f"  点数: {len(pts1)}")
print(f"  X 范围: {pts1[:, 0].min():.1f} ~ {pts1[:, 0].max():.1f} mm")
print(f"  Y 范围: {pts1[:, 1].min():.1f} ~ {pts1[:, 1].max():.1f} mm")
print(f"  Z 范围: {pts1[:, 2].min():.1f} ~ {pts1[:, 2].max():.1f} mm")

# 计算质心
centroid0 = pts0.mean(axis=0)
centroid1 = pts1.mean(axis=0)

print(f"\n质心:")
print(f"  帧 0: ({centroid0[0]:.1f}, {centroid0[1]:.1f}, {centroid0[2]:.1f})")
print(f"  帧 1: ({centroid1[0]:.1f}, {centroid1[1]:.1f}, {centroid1[2]:.1f})")
print(f"  距离: {np.linalg.norm(centroid1 - centroid0):.1f}mm")

# 计算 XZ 平面上的角度
angle0 = np.degrees(np.arctan2(centroid0[2], centroid0[0]))
angle1 = np.degrees(np.arctan2(centroid1[2], centroid1[0]))
angle_diff = angle1 - angle0
print(f"\nXZ 平面角度:")
print(f"  帧 0: {angle0:.1f}°")
print(f"  帧 1: {angle1:.1f}°")
print(f"  差值: {angle_diff:.1f}°")

# 使用 ICP 计算变换
voxel = 8.0
pcd0_ds = pcd0.voxel_down_sample(voxel)
pcd1_ds = pcd1.voxel_down_sample(voxel)

# 估计法向量
pcd0_ds.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=30))
pcd1_ds.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=30))

# ICP 配准
threshold = voxel * 2
result = o3d.pipelines.registration.registration_icp(
    pcd1_ds, pcd0_ds, threshold, np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
)

T_matrix = result.transformation
print(f"\nICP 配准结果:")
print(f"  fitness: {result.fitness:.3f}")
print(f"  RMSE: {result.inlier_rmse:.3f}")

# 提取旋转和平移
R_matrix = T_matrix[:3, :3]
t_vector = T_matrix[:3, 3]

# 计算旋转角度
angle = np.arccos(np.clip((np.trace(R_matrix) - 1) / 2, -1, 1))
angle_deg = np.degrees(angle)

print(f"  旋转角度: {angle_deg:.2f}°")
print(f"  平移: ({t_vector[0]:.1f}, {t_vector[1]:.1f}, {t_vector[2]:.1f})mm")
