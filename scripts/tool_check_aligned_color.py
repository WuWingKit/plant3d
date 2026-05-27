"""检查新对齐图的颜色分布"""
import cv2
import numpy as np
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_kinect import load_calib, get_intrinsics, get_stereo, make_aligned_color

# 加载标定
calib = load_calib("calib_imgs/calibration.json")
K_d, _, _ = get_intrinsics(calib, "depth")
K_c, dist_c, _ = get_intrinsics(calib, "color")
R, T = get_stereo(calib)

print(f"=== 检查新对齐图的颜色分布 ===")

# 加载第一帧
depth_dir = "capture_xxx/depth"
color_dir = "capture_xxx/color"

depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))

depth = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
color = cv2.imread(color_files[0])

# 使用新的对齐函数
aligned = make_aligned_color(depth, color, K_d, K_c, dist_c, R, T)

print(f"\n第一帧:")
print(f"  深度图形状: {depth.shape}")
print(f"  对齐图形状: {aligned.shape}")

# 检查对齐图的颜色分布
def analyze_color_distribution(img, name):
    """分析颜色分布"""
    print(f"\n{name} 颜色分布:")
    # 计算每个通道的统计
    for c in range(3):
        channel = img[:, :, c]
        print(f"  通道 {c}: 最小值={channel.min()}, 最大值={channel.max()}, "
              f"均值={channel.mean():.1f}, 标准差={channel.std():.1f}")

    # 计算黑色像素
    black = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    print(f"  黑色像素: {black.sum()} ({black.sum()/img.shape[0]/img.shape[1]*100:.1f}%)")

    # 计算非黑色像素的颜色分布
    non_black = ~black
    if non_black.any():
        print(f"  非黑色像素颜色分布:")
        for c in range(3):
            channel = img[:, :, c][non_black]
            print(f"    通道 {c}: 最小值={channel.min()}, 最大值={channel.max()}, "
                  f"均值={channel.mean():.1f}, 标准差={channel.std():.1f}")

analyze_color_distribution(aligned, "对齐图")

# 检查深度图的有效区域
valid_depth = depth > 0
print(f"\n深度图有效区域:")
print(f"  有效像素: {valid_depth.sum()} ({valid_depth.sum()/depth.shape[0]/depth.shape[1]*100:.1f}%)")

# 检查对齐图中非黑色像素与深度图有效区域的对应关系
aligned_non_black = (aligned[:, :, 0] != 0) | (aligned[:, :, 1] != 0) | (aligned[:, :, 2] != 0)
print(f"\n对齐图中非黑色像素: {aligned_non_black.sum()} ({aligned_non_black.sum()/aligned.shape[0]/aligned.shape[1]*100:.1f}%)")

# 检查对齐图中非黑色像素是否与深度图有效区域对应
both_valid = valid_depth & aligned_non_black
print(f"  深度图有效且对齐图非黑色: {both_valid.sum()} ({both_valid.sum()/depth.shape[0]/depth.shape[1]*100:.1f}%)")
