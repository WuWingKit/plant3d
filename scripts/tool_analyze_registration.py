"""分析配准结果"""
import open3d as o3d
import numpy as np
import json
import os

# 读取位姿
poses_path = "capture_xxx/output_v2/poses.json"
with open(poses_path) as f:
    poses = json.load(f)

print(f"=== 配准结果分析 ===")
print(f"总帧数: {len(poses)}")

# 提取位移
translations = []
rotations = []
for p in poses:
    T = np.array(p["pose"])
    translations.append(T[:3, 3])
    rotations.append(T[:3, :3])

translations = np.array(translations)

print(f"\n位移范围:")
print(f"  X: {translations[:, 0].min():.1f} ~ {translations[:, 0].max():.1f} mm")
print(f"  Y: {translations[:, 1].min():.1f} ~ {translations[:, 1].max():.1f} mm")
print(f"  Z: {translations[:, 2].min():.1f} ~ {translations[:, 2].max():.1f} mm")

# 计算累积旋转角度
print(f"\n累积旋转角度:")
cumulative_angle = 0
for i in range(len(rotations) - 1):
    R_diff = rotations[i+1] @ rotations[i].T
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)
    cumulative_angle += angle_deg

print(f"  总累积旋转: {cumulative_angle:.1f}°")

# 计算相邻帧的旋转角度
print(f"\n相邻帧旋转角度（前 20 帧）:")
for i in range(min(20, len(rotations) - 1)):
    R_diff = rotations[i+1] @ rotations[i].T
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)
    print(f"  帧 {i:3d} -> {i+1:3d}: {angle_deg:.2f}°")

# 检查最后一帧和第一帧的关系
print(f"\n首尾帧关系:")
R_first = rotations[0]
R_last = rotations[-1]
R_diff = R_last @ R_first.T
angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
angle_deg = np.degrees(angle)
print(f"  旋转差异: {angle_deg:.2f}°")

# 计算位移的质心
centroid = translations.mean(axis=0)
print(f"\n位移质心: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

# 计算每帧相对于质心的距离
print(f"\n每帧相对于质心的距离（前 20 帧）:")
for i in range(min(20, len(translations))):
    dist = np.linalg.norm(translations[i] - centroid)
    print(f"  帧 {i:3d}: {dist:.1f}mm")
