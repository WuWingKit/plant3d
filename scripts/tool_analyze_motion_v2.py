"""分析点云的运动，找出不动的底座"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))

print(f"=== 分析点云运动 ===")
print(f"总帧数: {len(pcd_files)}")

# 选择几帧跨度大的进行对比
frame_indices = [0, 30, 60, 90, 116]
frames = []
for idx in frame_indices:
    if idx < len(pcd_files):
        pcd = o3d.io.read_point_cloud(pcd_files[idx])
        frames.append({"idx": idx, "pcd": pcd, "pts": np.asarray(pcd.points)})

print(f"\n对比帧: {[f['idx'] for f in frames]}")

# 分析 Y 坐标分布
ref_pts = frames[0]["pts"]
y_min = ref_pts[:, 1].min()
y_max = ref_pts[:, 1].max()
print(f"\nY 范围: {y_min:.1f} ~ {y_max:.1f} mm")

# 将 Y 轴分成 10 个区间，分析每个区间的运动
n_bins = 10
y_bins = np.linspace(y_min, y_max, n_bins + 1)

print(f"\n=== Y 坐标区间运动分析 ===")
print(f"每个区间在不同帧中的质心变化:")

for i in range(n_bins):
    y_low = y_bins[i]
    y_high = y_bins[i+1]

    # 计算每帧在这个 Y 区间的质心
    centroids = []
    for f in frames:
        mask = (f["pts"][:, 1] >= y_low) & (f["pts"][:, 1] < y_high)
        if mask.any():
            centroid = f["pts"][mask].mean(axis=0)
            centroids.append(centroid)
        else:
            centroids.append(None)

    # 计算质心变化
    if centroids[0] is not None:
        changes = []
        for c in centroids[1:]:
            if c is not None:
                change = np.linalg.norm(c - centroids[0])
                changes.append(change)
        avg_change = np.mean(changes) if changes else 0

        # 判断是否是底座（运动小）
        is_base = avg_change < 2.0  # 阈值 2mm

        print(f"  Y={y_low:6.0f} ~ {y_high:6.0f}mm: "
              f"平均运动={avg_change:.1f}mm {'← 底座' if is_base else ''}")
    else:
        print(f"  Y={y_low:6.0f} ~ {y_high:6.0f}mm: 无数据")
