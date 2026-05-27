"""检查配准算法计算的变换矩阵"""
import open3d as o3d
import numpy as np
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_kinect import load_calib, get_intrinsics, get_stereo

# 加载标定
calib = load_calib("calib_imgs/calibration.json")
K_d, _, _ = get_intrinsics(calib, "depth")
K_c, dist_c, _ = get_intrinsics(calib, "color")
R, T = get_stereo(calib)

print(f"=== 检查配准算法 ===")

# 加载两帧点云
pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:5]

for i in range(len(pcd_files) - 1):
    pcd1 = o3d.io.read_point_cloud(pcd_files[i])
    pcd2 = o3d.io.read_point_cloud(pcd_files[i + 1])

    # 计算质心
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    centroid1 = pts1.mean(axis=0)
    centroid2 = pts2.mean(axis=0)

    print(f"\n帧 {i} -> {i+1}:")
    print(f"  质心1: ({centroid1[0]:.1f}, {centroid1[1]:.1f}, {centroid1[2]:.1f})")
    print(f"  质心2: ({centroid2[0]:.1f}, {centroid2[1]:.1f}, {centroid2[2]:.1f})")
    print(f"  质心距离: {np.linalg.norm(centroid2 - centroid1):.1f}mm")

    # 计算 XZ 平面上的角度
    angle1 = np.degrees(np.arctan2(centroid1[2], centroid1[0]))
    angle2 = np.degrees(np.arctan2(centroid2[2], centroid2[0]))
    angle_diff = angle2 - angle1
    print(f"  XZ 平面角度差: {angle_diff:.2f}°")

    # 使用 Open3D 的 ICP 计算变换
    # 降采样
    voxel = 8.0
    pcd1_ds = pcd1.voxel_down_sample(voxel)
    pcd2_ds = pcd2.voxel_down_sample(voxel)

    # 估计法向量
    pcd1_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=30))
    pcd2_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=30))

    # ICP 配准
    threshold = voxel * 2
    result = o3d.pipelines.registration.registration_icp(
        pcd2_ds, pcd1_ds, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    T_matrix = result.transformation
    print(f"  ICP fitness: {result.fitness:.3f}")
    print(f"  ICP RMSE: {result.inlier_rmse:.3f}")

    # 提取旋转和平移
    R_matrix = T_matrix[:3, :3]
    t_vector = T_matrix[:3, 3]

    # 计算旋转角度
    angle = np.arccos(np.clip((np.trace(R_matrix) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)

    print(f"  旋转角度: {angle_deg:.2f}°")
    print(f"  平移: ({t_vector[0]:.1f}, {t_vector[1]:.1f}, {t_vector[2]:.1f})mm")
