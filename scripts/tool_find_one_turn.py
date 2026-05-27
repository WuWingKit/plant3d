"""找到转完一圈的位置"""
import open3d as o3d
import numpy as np
import glob
import os

pcds_dir = "capture_xxx/pcds_seg"
pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))

print(f"=== 找到转完一圈的位置 ===")
print(f"总帧数: {len(pcd_files)}")

# 加载第一帧作为参考
ref_pcd = o3d.io.read_point_cloud(pcd_files[0])
ref_pts = np.asarray(ref_pcd.points)
ref_centroid = ref_pts.mean(axis=0)

print(f"\n参考帧 (帧 0):")
print(f"  质心: ({ref_centroid[0]:.1f}, {ref_centroid[1]:.1f}, {ref_centroid[2]:.1f})")

# 在大概一圈后的区间（帧 100-140）找与开始帧最相似的帧
# 根据 metadata.json，总帧数 241，转了 2 圈，所以一圈约 120 帧
search_start = 100
search_end = 150

print(f"\n在帧 {search_start}-{search_end} 中找最相似的帧:")

similarities = []
for i in range(search_start, min(search_end, len(pcd_files))):
    pcd = o3d.io.read_point_cloud(pcd_files[i])
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)

    # 计算质心距离
    dist = np.linalg.norm(centroid - ref_centroid)

    # 计算 XZ 平面角度差
    angle_ref = np.degrees(np.arctan2(ref_centroid[2], ref_centroid[0]))
    angle_cur = np.degrees(np.arctan2(centroid[2], centroid[0]))
    angle_diff = abs(angle_cur - angle_ref)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # 计算点云相似度（使用点数和范围）
    size_diff = abs(len(pts) - len(ref_pts)) / max(len(pts), len(ref_pts))

    # 综合相似度（距离越小、角度差越小、大小越接近 = 越相似）
    similarity = dist + angle_diff * 10 + size_diff * 100

    similarities.append({
        "frame": i,
        "centroid": centroid,
        "dist": dist,
        "angle_diff": angle_diff,
        "size_diff": size_diff,
        "similarity": similarity
    })

    print(f"  帧 {i:3d}: 质心距离={dist:.1f}mm, 角度差={angle_diff:.1f}°, "
          f"大小差异={size_diff*100:.1f}%, 综合相似度={similarity:.1f}")

# 找到最相似的帧
best_frame = min(similarities, key=lambda x: x["similarity"])
print(f"\n最相似的帧: {best_frame['frame']}")
print(f"  质心: ({best_frame['centroid'][0]:.1f}, {best_frame['centroid'][1]:.1f}, {best_frame['centroid'][2]:.1f})")
print(f"  质心距离: {best_frame['dist']:.1f}mm")
print(f"  角度差: {best_frame['angle_diff']:.1f}°")

# 计算一圈的帧数
one_turn_frames = best_frame['frame']
print(f"\n一圈的帧数: {one_turn_frames}")

# 计算每帧旋转角度
angle_per_frame = 360.0 / one_turn_frames
print(f"每帧旋转角度: {angle_per_frame:.2f}°")
