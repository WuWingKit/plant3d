"""检查原始颜色图的旋转"""
import cv2
import numpy as np
import glob
import os

color_dir = "capture_xxx/color"
color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))[:20]

print(f"=== 检查原始颜色图旋转 ===")
print(f"分析前 20 帧")

# 分析第一帧和最后一帧
color_first = cv2.imread(color_files[0])
color_last = cv2.imread(color_files[-1])

print(f"\n第一帧:")
print(f"  形状: {color_first.shape}")

print(f"\n最后一帧:")
print(f"  形状: {color_last.shape}")

# 计算颜色图的质心（基于亮度）
def compute_color_centroid(color):
    """计算颜色图的质心（基于亮度）"""
    # 转换为灰度图
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # 计算加权质心
    centroid_u = np.average(u, weights=gray)
    centroid_v = np.average(v, weights=gray)
    return centroid_u, centroid_v

centroid_first = compute_color_centroid(color_first)
centroid_last = compute_color_centroid(color_last)

print(f"\n颜色图质心（基于亮度）:")
print(f"  第一帧: ({centroid_first[0]:.1f}, {centroid_first[1]:.1f})")
print(f"  最后帧: ({centroid_last[0]:.1f}, {centroid_last[1]:.1f})")

# 计算质心距离
dist = np.sqrt((centroid_last[0] - centroid_first[0])**2 + (centroid_last[1] - centroid_first[1])**2)
print(f"  质心距离: {dist:.1f} pixels")

# 分析相邻帧的颜色图质心变化
print(f"\n相邻帧颜色图质心变化（每 5 帧）:")
for i in range(0, min(20, len(color_files)) - 1, 5):
    color1 = cv2.imread(color_files[i])
    color2 = cv2.imread(color_files[i + 1])
    centroid1 = compute_color_centroid(color1)
    centroid2 = compute_color_centroid(color2)
    dist = np.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)
    print(f"  帧 {i:3d} -> {i+1:3d}: 质心距离={dist:.1f} pixels")

# 检查颜色图中是否有明显的旋转特征
print(f"\n检查颜色图中是否有明显的旋转特征:")
# 计算颜色图的主方向
gray_first = cv2.cvtColor(color_first, cv2.COLOR_BGR2GRAY)
gray_last = cv2.cvtColor(color_last, cv2.COLOR_BGR2GRAY)

# 使用 Harris 角点检测
harris_first = cv2.cornerHarris(gray_first, 2, 3, 0.04)
harris_last = cv2.cornerHarris(gray_last, 2, 3, 0.04)

# 计算角点数量
threshold = 0.01 * harris_first.max()
corners_first = np.sum(harris_first > threshold)
corners_last = np.sum(harris_last > threshold)

print(f"  第一帧角点数量: {corners_first}")
print(f"  最后帧角点数量: {corners_last}")
