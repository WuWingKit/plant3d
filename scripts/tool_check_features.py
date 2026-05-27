"""检查点云特征"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))[:5]

print(f"=== 检查点云特征 ===")
print(f"分析前 5 帧")

for i in range(min(5, len(pcd_files))):
    pcd = o3d.io.read_point_cloud(pcd_files[i])
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    print(f"\n帧 {i}:")
    print(f"  点数: {len(pts)}")
    print(f"  X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
    print(f"  Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
    print(f"  Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

    # 计算质心
    centroid = pts.mean(axis=0)
    print(f"  质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    # 计算点云在 XZ 平面上的分布
    xz_pts = pts[:, [0, 2]]
    xz_centroid = xz_pts.mean(axis=0)
    xz_distances = np.sqrt(np.sum((xz_pts - xz_centroid)**2, axis=1))
    print(f"  XZ 平面上的分布: 平均距离={xz_distances.mean():.1f}mm, 标准差={xz_distances.std():.1f}mm")

    # 检查点云是否有明显的特征（如边缘、角点）
    # 计算点云的法向量
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    normals = np.asarray(pcd.normals)

    # 计算法向量的变化
    normal_changes = np.diff(normals, axis=0)
    normal_change_magnitudes = np.linalg.norm(normal_changes, axis=1)
    print(f"  法向量变化: 平均={normal_change_magnitudes.mean():.3f}, 标准差={normal_change_magnitudes.std():.3f}")

    # 检查颜色分布
    if colors is not None:
        print(f"  颜色范围: R={colors[:, 0].min():.2f}~{colors[:, 0].max():.2f}, "
              f"G={colors[:, 1].min():.2f}~{colors[:, 1].max():.2f}, "
              f"B={colors[:, 2].min():.2f}~{colors[:, 2].max():.2f}")

        # 计算颜色的变化
        color_changes = np.diff(colors, axis=0)
        color_change_magnitudes = np.linalg.norm(color_changes, axis=1)
        print(f"  颜色变化: 平均={color_change_magnitudes.mean():.3f}, 标准差={color_change_magnitudes.std():.3f}")
