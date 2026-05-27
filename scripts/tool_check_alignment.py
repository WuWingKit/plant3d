"""检查颜色图对齐"""
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

print(f"=== 检查颜色图对齐 ===")

# 加载第一帧
depth_dir = "capture_xxx/depth"
color_dir = "capture_xxx/color"
aligned_dir = "capture_xxx/aligned"

depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
aligned_files = sorted(glob.glob(os.path.join(aligned_dir, "*.png")))

depth = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
color = cv2.imread(color_files[0])
aligned_old = cv2.imread(aligned_files[0])

# 使用新的对齐函数
aligned_new = make_aligned_color(depth, color, K_d, K_c, dist_c, R, T)

print(f"\n第一帧:")
print(f"  深度图形状: {depth.shape}")
print(f"  原始颜色图形状: {color.shape}")
print(f"  旧对齐图形状: {aligned_old.shape}")
print(f"  新对齐图形状: {aligned_new.shape}")

# 检查对齐图的黑色像素
def count_black(img):
    """计算黑色像素数量"""
    if len(img.shape) == 3:
        black = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    else:
        black = img == 0
    return black.sum()

black_old = count_black(aligned_old)
black_new = count_black(aligned_new)
total = aligned_old.shape[0] * aligned_old.shape[1]

print(f"\n黑色像素数量:")
print(f"  旧对齐图: {black_old} ({black_old/total*100:.1f}%)")
print(f"  新对齐图: {black_new} ({black_new/total*100:.1f}%)")

# 检查对齐图的差异
diff = cv2.absdiff(aligned_old, aligned_new)
diff_mean = diff.mean()
diff_max = diff.max()

print(f"\n对齐图差异:")
print(f"  差异均值: {diff_mean:.1f}")
print(f"  差异最大值: {diff_max}")

# 检查深度图的有效区域
valid_depth = depth > 0
print(f"\n深度图有效区域:")
print(f"  有效像素: {valid_depth.sum()} ({valid_depth.sum()/total*100:.1f}%)")
