"""检查相邻帧之间的颜色图差异"""
import cv2
import numpy as np
import glob
import os

color_dir = "capture_xxx/color"
color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))[:20]

print(f"=== 检查相邻帧颜色图差异 ===")
print(f"分析前 20 帧")

# 分析相邻帧的差异
for i in range(min(20, len(color_files)) - 1):
    color1 = cv2.imread(color_files[i])
    color2 = cv2.imread(color_files[i + 1])

    # 计算差异
    diff = cv2.absdiff(color1, color2)
    diff_mean = diff.mean()
    diff_max = diff.max()

    # 计算差异区域
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    threshold = 30
    diff_mask = diff_gray > threshold
    diff_ratio = diff_mask.sum() / diff_mask.size

    print(f"帧 {i:3d} -> {i+1:3d}: 差异均值={diff_mean:.1f}, 差异最大值={diff_max}, "
          f"差异区域比例={diff_ratio*100:.1f}%")

# 检查第一帧和最后一帧的差异
print(f"\n第一帧和最后一帧的差异:")
color_first = cv2.imread(color_files[0])
color_last = cv2.imread(color_files[-1])
diff = cv2.absdiff(color_first, color_last)
diff_mean = diff.mean()
diff_max = diff.max()
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
threshold = 30
diff_mask = diff_gray > threshold
diff_ratio = diff_mask.sum() / diff_mask.size
print(f"  差异均值={diff_mean:.1f}, 差异最大值={diff_max}, "
      f"差异区域比例={diff_ratio*100:.1f}%")

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
