"""检查原始颜色图"""
import cv2
import numpy as np
import glob
import os

color_dir = "capture_xxx/color"
aligned_dir = "capture_xxx/aligned"
color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))[:3]

for f in color_files:
    fid = os.path.splitext(os.path.basename(f))[0]
    color = cv2.imread(f)
    aligned = cv2.imread(os.path.join(aligned_dir, f"{fid}.png"))

    print(f"\n=== {fid}.png ===")
    print(f"原始颜色图形状: {color.shape}")
    print(f"对齐图形状: {aligned.shape}")

    # 检查原始颜色图的黑色像素
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color_black = (color_rgb[:, :, 0] == 0) & (color_rgb[:, :, 1] == 0) & (color_rgb[:, :, 2] == 0)
    print(f"原始颜色图黑色像素: {color_black.sum()} / {color.size//3} ({color_black.sum()/(color.size//3)*100:.1f}%)")

    # 检查对齐图的黑色像素
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    aligned_black = (aligned_rgb[:, :, 0] == 0) & (aligned_rgb[:, :, 1] == 0) & (aligned_rgb[:, :, 2] == 0)
    print(f"对齐图黑色像素: {aligned_black.sum()} / {aligned.size//3} ({aligned_black.sum()/(aligned.size//3)*100:.1f}%)")

    # 检查颜色范围
    print(f"原始颜色图范围: R={color_rgb[:, :, 0].min()}~{color_rgb[:, :, 0].max()}, "
          f"G={color_rgb[:, :, 1].min()}~{color_rgb[:, :, 1].max()}, "
          f"B={color_rgb[:, :, 2].min()}~{color_rgb[:, :, 2].max()}")
    print(f"对齐图范围: R={aligned_rgb[:, :, 0].min()}~{aligned_rgb[:, :, 0].max()}, "
          f"G={aligned_rgb[:, :, 1].min()}~{aligned_rgb[:, :, 1].max()}, "
          f"B={aligned_rgb[:, :, 2].min()}~{aligned_rgb[:, :, 2].max()}")
