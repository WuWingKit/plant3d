"""分析黑色点的分布"""
import cv2
import numpy as np
import glob
import os

depth_dir = "capture_xxx/depth"
aligned_dir = "capture_xxx/aligned"
depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:3]

for f in depth_files:
    fid = os.path.splitext(os.path.basename(f))[0]
    depth = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    aligned = cv2.imread(os.path.join(aligned_dir, f"{fid}.png"))

    print(f"\n=== {fid}.png ===")
    print(f"深度图形状: {depth.shape}")
    print(f"对齐图形状: {aligned.shape}")

    # 检查深度图
    valid_depth = depth[depth > 0]
    print(f"有效深度像素: {len(valid_depth)} / {depth.size} ({len(valid_depth)/depth.size*100:.1f}%)")

    # 检查对齐图的颜色分布
    # 将对齐图转换为 RGB
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    # 检查黑色像素
    black_mask = (aligned_rgb[:, :, 0] == 0) & (aligned_rgb[:, :, 1] == 0) & (aligned_rgb[:, :, 2] == 0)
    print(f"对齐图黑色像素: {black_mask.sum()} / {aligned.size//3} ({black_mask.sum()/(aligned.size//3)*100:.1f}%)")

    # 检查深度图和对齐图的对应关系
    # 有效深度像素对应的颜色
    valid_depth_mask = depth > 0
    valid_colors = aligned_rgb[valid_depth_mask]
    valid_black = (valid_colors[:, 0] == 0) & (valid_colors[:, 1] == 0) & (valid_colors[:, 2] == 0)
    print(f"有效深度对应黑色像素: {valid_black.sum()} / {len(valid_colors)} ({valid_black.sum()/len(valid_colors)*100:.1f}%)")

    # 检查黑色像素的空间分布
    if black_mask.any():
        black_coords = np.where(black_mask)
        print(f"黑色像素 Y 范围: {black_coords[0].min()} ~ {black_coords[0].max()}")
        print(f"黑色像素 X 范围: {black_coords[1].min()} ~ {black_coords[1].max()}")

        # 检查边缘区域
        h, w = black_mask.shape
        margin = 50
        edge_black = black_mask[:margin, :].sum() + black_mask[-margin:, :].sum() + \
                     black_mask[:, :margin].sum() + black_mask[:, -margin:].sum()
        print(f"边缘区域黑色像素: {edge_black} ({edge_black/black_mask.sum()*100:.1f}%)")
