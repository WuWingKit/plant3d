"""分析点云的运动，找出不动的底座"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))

print(f"=== 分析点云运动 ===")
print(f"总帧数: {len(pcd_files)}")

# 选择几帧跨度大的进行对比（间隔约 30 帧）
frame_indices = [0, 30, 60, 90]
frames = []
for idx in frame_indices:
    if idx < len(pcd_files):
        pcd = o3d.io.read_point_cloud(pcd_files[idx])
        frames.append({"idx": idx, "pcd": pcd, "pts": np.asarray(pcd.points)})

print(f"\n对比帧: {[f['idx'] for f in frames]}")

# 分析每个点在不同帧中的变化
# 使用第一帧作为参考
ref_pts = frames[0]["pts"]
ref_centroid = ref_pts.mean(axis=0)

print(f"\n参考帧 (帧 {frames[0]['idx']}) 质心: ({ref_centroid[0]:.1f}, {ref_centroid[1]:.1f}, {ref_centroid[2]:.1f})")

# 计算每帧相对于参考帧的质心变化
print(f"\n各帧质心变化:")
for f in frames:
    centroid = f["pts"].mean(axis=0)
    dist = np.linalg.norm(centroid - ref_centroid)
    print(f"  帧 {f['idx']:3d}: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}), "
          f"距离={dist:.1f}mm")

# 分析 Y 坐标分布，找出可能的底座区域
print(f"\n=== Y 坐标分布分析 ===")
y_min = ref_pts[:, 1].min()
y_max = ref_pts[:, 1].max()
print(f"Y 范围: {y_min:.1f} ~ {y_max:.1f} mm")

# 将 Y 轴分成 10 个区间
n_bins = 10
y_bins = np.linspace(y_min, y_max, n_bins + 1)
print(f"\nY 坐标区间分析:")
for i in range(n_bins):
    mask = (ref_pts[:, 1] >= y_bins[i]) & (ref_pts[:, 1] < y_bins[i+1])
    if mask.any():
        n_points = mask.sum()
        # 计算这个区间的点在其他帧中的变化
        avg_changes = []
        for f in frames[1:]:
            # 找到最近的点（简化：使用相同的索引）
            if len(f["pts"]) >= len(ref_pts):
                diff = np.linalg.norm(f["pts"][mask] - ref_pts[mask], axis=1)
                avg_changes.append(diff.mean())
        avg_change = np.mean(avg_changes) if avg_changes else 0
        print(f"  Y={y_bins[i]:6.0f} ~ {y_bins[i+1]:6.0f}mm: {n_points:5d} 点, "
              f"平均运动={avg_change:.1f}mm")
