"""分析旋转角度的准确性"""
import open3d as o3d
import numpy as np
import csv
import glob
import os

# 读取时间戳
timestamps = []
with open("capture_xxx/timestamps.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamps.append({
            "id": int(row["id"]),
            "time": float(row["timestamp_sec"])
        })

print(f"=== 分析旋转角度的准确性 ===")
print(f"总帧数: {len(timestamps)}")

# 计算帧间隔
intervals = []
for i in range(len(timestamps) - 1):
    interval = timestamps[i+1]["time"] - timestamps[i]["time"]
    intervals.append(interval)

print(f"\n帧间隔统计:")
print(f"  平均: {np.mean(intervals):.3f} 秒")
print(f"  最小: {np.min(intervals):.3f} 秒")
print(f"  最大: {np.max(intervals):.3f} 秒")

# 计算理论旋转角度
rotation_speed = 18.157  # °/秒
print(f"\n旋转速度: {rotation_speed}°/秒")

# 计算累积旋转角度
cumulative_angle = 0
for i in range(len(intervals)):
    angle = intervals[i] * rotation_speed
    cumulative_angle += angle

print(f"\n理论累积旋转角度（一圈）: {cumulative_angle:.1f}°")

# 分析帧 0 和帧 117 的相似度（应该是一圈）
pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))

if len(pcd_files) > 117:
    print(f"\n帧 0 和帧 117 的相似度分析:")

    # 加载帧 0 和帧 117
    pcd0 = o3d.io.read_point_cloud(pcd_files[0])
    pcd117 = o3d.io.read_point_cloud(pcd_files[117])

    pts0 = np.asarray(pcd0.points)
    pts117 = np.asarray(pcd117.points)

    # 计算质心
    centroid0 = pts0.mean(axis=0)
    centroid117 = pts117.mean(axis=0)

    print(f"  帧 0 质心: ({centroid0[0]:.1f}, {centroid0[1]:.1f}, {centroid0[2]:.1f})")
    print(f"  帧 117 质心: ({centroid117[0]:.1f}, {centroid117[1]:.1f}, {centroid117[2]:.1f})")
    print(f"  质心距离: {np.linalg.norm(centroid117 - centroid0):.1f}mm")

    # 计算大小差异
    size_diff = abs(len(pts0) - len(pts117)) / max(len(pts0), len(pts117))
    print(f"  大小差异: {size_diff*100:.1f}%")

    # 计算 XZ 平面上的角度
    angle0 = np.degrees(np.arctan2(centroid0[2], centroid0[0]))
    angle117 = np.degrees(np.arctan2(centroid117[2], centroid117[0]))
    angle_diff = angle117 - angle0
    print(f"  XZ 平面角度差: {angle_diff:.1f}°")
