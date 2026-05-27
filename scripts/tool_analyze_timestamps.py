"""分析时间戳和旋转角度"""
import numpy as np
import csv
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

# 计算帧间隔
intervals = []
for i in range(len(timestamps) - 1):
    interval = timestamps[i+1]["time"] - timestamps[i]["time"]
    intervals.append({
        "id": timestamps[i]["id"],
        "interval": interval
    })

print(f"=== 时间戳分析 ===")
print(f"总帧数: {len(timestamps)}")
print(f"总时间: {timestamps[-1]['time']:.2f} 秒")

print(f"\n帧间隔统计:")
interval_values = [x["interval"] for x in intervals]
print(f"  平均: {np.mean(interval_values):.3f} 秒")
print(f"  最小: {np.min(interval_values):.3f} 秒")
print(f"  最大: {np.max(interval_values):.3f} 秒")
print(f"  标准差: {np.std(interval_values):.3f} 秒")

# 根据 metadata.json 的旋转速度计算每帧旋转角度
rotation_speed = 18.157  # °/秒
print(f"\n旋转速度: {rotation_speed}°/秒")

# 计算每帧旋转角度
print(f"\n每帧旋转角度（前 20 帧）:")
cumulative_angle = 0
for i in range(min(20, len(intervals))):
    angle = intervals[i]["interval"] * rotation_speed
    cumulative_angle += angle
    print(f"  帧 {i:3d} -> {i+1:3d}: 间隔={intervals[i]['interval']:.3f}s, "
          f"旋转={angle:.2f}°, 累积={cumulative_angle:.2f}°")

# 计算累积旋转角度
print(f"\n累积旋转角度（每 10 帧）:")
cumulative_angle = 0
for i in range(len(intervals)):
    angle = intervals[i]["interval"] * rotation_speed
    cumulative_angle += angle
    if (i + 1) % 10 == 0 or i == len(intervals) - 1:
        print(f"  帧 {i+1:3d}: 累积旋转={cumulative_angle:.1f}°")

# 找出转完一圈（360°）的位置
print(f"\n找转完一圈（360°）的位置:")
cumulative_angle = 0
for i in range(len(intervals)):
    angle = intervals[i]["interval"] * rotation_speed
    cumulative_angle += angle
    if cumulative_angle >= 360:
        print(f"  帧 {i+1:3d}: 累积旋转={cumulative_angle:.1f}° (超过 360°)")
        break

# 找出转完两圈（720°）的位置
print(f"\n找转完两圈（720°）的位置:")
cumulative_angle = 0
for i in range(len(intervals)):
    angle = intervals[i]["interval"] * rotation_speed
    cumulative_angle += angle
    if cumulative_angle >= 720:
        print(f"  帧 {i+1:3d}: 累积旋转={cumulative_angle:.1f}° (超过 720°)")
        break
