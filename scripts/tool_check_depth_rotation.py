"""检查原始深度图的旋转"""
import cv2
import numpy as np
import glob
import os

depth_dir = "capture_xxx/depth"
depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:20]

print(f"=== 检查原始深度图旋转 ===")
print(f"分析前 20 帧")

# 分析第一帧和最后一帧
depth_first = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
depth_last = cv2.imread(depth_files[-1], cv2.IMREAD_UNCHANGED)

print(f"\n第一帧:")
print(f"  形状: {depth_first.shape}")
print(f"  深度范围: {depth_first.min()} ~ {depth_first.max()} mm")

print(f"\n最后一帧:")
print(f"  形状: {depth_last.shape}")
print(f"  深度范围: {depth_last.min()} ~ {depth_last.max()} mm")

# 计算深度图的质心
def compute_depth_centroid(depth):
    """计算深度图的质心"""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid_mask = depth > 0
    if not valid_mask.any():
        return None
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]
    # 计算加权质心
    centroid_u = np.average(u_valid, weights=depth_valid)
    centroid_v = np.average(v_valid, weights=depth_valid)
    return centroid_u, centroid_v

centroid_first = compute_depth_centroid(depth_first)
centroid_last = compute_depth_centroid(depth_last)

print(f"\n深度图质心:")
print(f"  第一帧: ({centroid_first[0]:.1f}, {centroid_first[1]:.1f})")
print(f"  最后帧: ({centroid_last[0]:.1f}, {centroid_last[1]:.1f})")

# 计算质心距离
dist = np.sqrt((centroid_last[0] - centroid_first[0])**2 + (centroid_last[1] - centroid_first[1])**2)
print(f"  质心距离: {dist:.1f} pixels")

# 分析相邻帧的深度图质心变化
print(f"\n相邻帧深度图质心变化（每 5 帧）:")
for i in range(0, min(20, len(depth_files)) - 1, 5):
    depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth_files[i + 1], cv2.IMREAD_UNCHANGED)
    centroid1 = compute_depth_centroid(depth1)
    centroid2 = compute_depth_centroid(depth2)
    if centroid1 and centroid2:
        dist = np.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)
        print(f"  帧 {i:3d} -> {i+1:3d}: 质心距离={dist:.1f} pixels")
