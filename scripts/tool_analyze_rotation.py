"""分析旋转情况"""
import open3d as o3d
import numpy as np
import json
import os

# 读取位姿
poses_path = "capture_xxx/output_v2/poses.json"
with open(poses_path) as f:
    poses = json.load(f)

print(f"=== 位姿分析 ===")
print(f"总帧数: {len(poses)}")

# 提取位移
translations = []
for p in poses:
    T = np.array(p["pose"])
    translations.append(T[:3, 3])
translations = np.array(translations)

print(f"\n位移范围:")
print(f"  X: {translations[:, 0].min():.1f} ~ {translations[:, 0].max():.1f} mm")
print(f"  Y: {translations[:, 1].min():.1f} ~ {translations[:, 1].max():.1f} mm")
print(f"  Z: {translations[:, 2].min():.1f} ~ {translations[:, 2].max():.1f} mm")

# 计算相邻帧的位移差异
print(f"\n相邻帧位移差异:")
for i in range(0, len(translations), 30):
    if i < len(translations) - 1:
        diff = translations[i+1] - translations[i]
        dist = np.linalg.norm(diff)
        print(f"  帧 {i:3d} -> {i+1:3d}: 位移 {dist:.1f}mm")

# 分析旋转角度
print(f"\n旋转分析:")
# 提取旋转矩阵，计算累积旋转角度
rotations = []
for p in poses:
    T = np.array(p["pose"])
    R = T[:3, :3]
    rotations.append(R)

# 计算相邻帧之间的旋转角度
angles = []
for i in range(len(rotations) - 1):
    R_diff = rotations[i+1] @ rotations[i].T
    # 旋转矩阵转角度
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    angles.append(np.degrees(angle))

angles = np.array(angles)
print(f"相邻帧旋转角度:")
print(f"  平均: {angles.mean():.2f}°")
print(f"  最小: {angles.min():.2f}°")
print(f"  最大: {angles.max():.2f}°")
print(f"  总和: {angles.sum():.1f}°")

# 读取合并点云
merged_path = "capture_xxx/output_v2/merged_clean.ply"
pcd = o3d.io.read_point_cloud(merged_path)
pts = np.asarray(pcd.points)

print(f"\n=== 合并点云分析 ===")
print(f"总点数: {len(pts)}")
print(f"X 范围: {pts[:, 0].min():.1f} ~ {pts[:, 0].max():.1f} mm")
print(f"Y 范围: {pts[:, 1].min():.1f} ~ {pts[:, 1].max():.1f} mm")
print(f"Z 范围: {pts[:, 2].min():.1f} ~ {pts[:, 2].max():.1f} mm")

# 计算点云的主方向
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(pts)
print(f"\n主成分分析:")
print(f"  第一主成分: {pca.components_[0]}")
print(f"  第二主成分: {pca.components_[1]}")
print(f"  第三主成分: {pca.components_[2]}")
print(f"  解释方差比: {pca.explained_variance_ratio_}")
