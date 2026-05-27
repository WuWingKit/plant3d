"""检查旋转角度计算"""
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

print(f"=== 检查旋转角度计算 ===")
print(f"总帧数: {len(timestamps)}")

# 计算帧间隔
print(f"\n前 10 帧的时间间隔:")
for i in range(min(10, len(timestamps) - 1)):
    interval = timestamps[i+1]["time"] - timestamps[i]["time"]
    print(f"  帧 {i:3d} -> {i+1:3d}: {interval:.3f} 秒")

# 根据 metadata.json 的旋转速度计算每帧旋转角度
rotation_speed = 18.157  # °/秒
print(f"\n旋转速度: {rotation_speed}°/秒")

# 计算每帧旋转角度
print(f"\n前 10 帧的理论旋转角度:")
cumulative_angle = 0
for i in range(min(10, len(timestamps) - 1)):
    interval = timestamps[i+1]["time"] - timestamps[i]["time"]
    angle = interval * rotation_speed
    cumulative_angle += angle
    print(f"  帧 {i:3d} -> {i+1:3d}: 间隔={interval:.3f}s, 旋转={angle:.2f}°, 累积={cumulative_angle:.2f}°")

# 检查帧 0 和帧 1 的时间间隔
print(f"\n帧 0 和帧 1 的时间间隔:")
interval_01 = timestamps[1]["time"] - timestamps[0]["time"]
angle_01 = interval_01 * rotation_speed
print(f"  间隔: {interval_01:.3f} 秒")
print(f"  理论旋转: {angle_01:.2f}°")
