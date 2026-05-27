"""
06_stage3_register_correct.py — 使用正确的旋转轴

旋转轴在 (-44.2, 907.9)，而不是原点 (0, 0)
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import csv
import glob
import copy
import argparse
import time
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)


def load_timestamps(input_dir):
    """加载时间戳"""
    timestamps = []
    with open(os.path.join(input_dir, "timestamps.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append({
                "id": int(row["id"]),
                "time": float(row["timestamp_sec"])
            })
    return timestamps


def find_rotation_axis(pcd_files, n_frames=5):
    """找到旋转轴的位置"""
    print(f"\n  找到旋转轴的位置...")

    # 计算前几帧的质心
    centroids = []
    for i in range(min(n_frames, len(pcd_files))):
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        pts = np.asarray(pcd.points)
        centroid = pts.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # 计算质心在 XZ 平面上的分布
    xz_centroids = centroids[:, [0, 2]]

    # 使用最小二乘法拟合圆
    def fit_circle(x, y):
        """拟合圆"""
        A = np.array([
            [2*x[0], 2*y[0], 1],
            [2*x[1], 2*y[1], 1],
            [2*x[2], 2*y[2], 1],
        ])
        b = np.array([x[0]**2 + y[0]**2, x[1]**2 + y[1]**2, x[2]**2 + y[2]**2])
        try:
            params = np.linalg.solve(A, b)
            cx, cy = params[0], params[1]
            r = np.sqrt(params[2] + cx**2 + cy**2)
            return cx, cy, r
        except:
            return None, None, None

    cx, cy, r = fit_circle(xz_centroids[:, 0], xz_centroids[:, 1])
    if cx is not None:
        print(f"    旋转轴: ({cx:.1f}, {cy:.1f})")
        print(f"    半径: {r:.1f}mm")
        return cx, cy
    else:
        # 如果拟合失败，使用质心平均位置
        avg_centroid = centroids.mean(axis=0)
        print(f"    旋转轴（质心平均）: ({avg_centroid[0]:.1f}, {avg_centroid[2]:.1f})")
        return avg_centroid[0], avg_centroid[2]


def find_one_turn(pcd_files, search_range=(100, 150)):
    """找到转完一圈的位置"""
    print(f"\n  找到转完一圈的位置...")

    # 加载第一帧作为参考
    ref_pcd = o3d.io.read_point_cloud(pcd_files[0])
    ref_pts = np.asarray(ref_pcd.points)
    ref_centroid = ref_pts.mean(axis=0)

    # 在大概一圈后的区间找与开始帧最相似的帧
    best_similarity = float('inf')
    best_frame = search_range[0]

    for i in range(search_range[0], min(search_range[1], len(pcd_files))):
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        pts = np.asarray(pcd.points)
        centroid = pts.mean(axis=0)

        # 计算质心距离
        dist = np.linalg.norm(centroid - ref_centroid)

        # 计算大小差异
        size_diff = abs(len(pts) - len(ref_pts)) / max(len(pts), len(ref_pts))

        # 综合相似度
        similarity = dist + size_diff * 100

        if similarity < best_similarity:
            best_similarity = similarity
            best_frame = i

    print(f"    最相似的帧: {best_frame} (相似度: {best_similarity:.1f})")
    return best_frame


def main():
    parser = argparse.ArgumentParser(description="Stage 3: 使用正确的旋转轴")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-ratio", type=float, default=0.35,
                        help="抽帧比例 (默认 0.35)")
    parser.add_argument("--search-start", type=int, default=100,
                        help="搜索一圈位置的起始帧 (默认 100)")
    parser.add_argument("--search-end", type=int, default=150,
                        help="搜索一圈位置的结束帧 (默认 150)")
    parser.add_argument("--rotation-speed", type=float, default=18.157,
                        help="旋转速度 °/秒 (默认 18.157)")
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 3: 使用正确的旋转轴")
    print("=" * 64)
    print(f"抽帧比例 = {args.sample_ratio*100:.0f}%")
    print(f"旋转速度 = {args.rotation_speed}°/秒")
    print()

    pcds_dir = os.path.join(args.input, "pcds_seg")
    if not os.path.isdir(pcds_dir):
        print(f"✗ 找不到 {pcds_dir}，先跑 05_stage2_segment.py")
        sys.exit(1)

    out_dir = args.output or os.path.join(args.input, "output_v2")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 加载时间戳 ----
    print("[1/5] 加载时间戳...")
    timestamps = load_timestamps(args.input)
    print(f"  加载 {len(timestamps)} 个时间戳")

    # ---- 找到一圈的帧数 ----
    pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if len(pcd_files) < 3:
        print(f"✗ 帧数太少 ({len(pcd_files)})")
        sys.exit(1)

    one_turn_frame = find_one_turn(pcd_files, (args.search_start, args.search_end))
    print(f"  一圈的帧数: {one_turn_frame}")

    # 只使用一圈的数据
    pcd_files_one_turn = pcd_files[:one_turn_frame]
    print(f"  使用帧 0-{one_turn_frame-1} (共 {one_turn_frame} 帧)")

    # 平均抽帧，只使用 30-40%
    n_samples = int(one_turn_frame * args.sample_ratio)
    sample_indices = np.linspace(0, one_turn_frame - 1, n_samples, dtype=int)
    print(f"  抽帧数量: {n_samples} (比例 {args.sample_ratio*100:.0f}%)")

    # ---- 找到旋转轴 ----
    print("\n[2/5] 找到旋转轴...")
    axis_x, axis_z = find_rotation_axis(pcd_files_one_turn)
    print(f"  旋转轴: ({axis_x:.1f}, {axis_z:.1f})")

    # ---- 计算每帧的理论旋转角度 ----
    print("\n[3/5] 计算理论旋转角度...")
    cumulative_angle = 0
    rotation_angles = [0]  # 第一帧旋转角度为 0
    for i in range(one_turn_frame - 1):
        interval = timestamps[i+1]["time"] - timestamps[i]["time"]
        angle = interval * args.rotation_speed
        cumulative_angle += angle
        rotation_angles.append(cumulative_angle)
    print(f"  累积旋转角度: {cumulative_angle:.1f}°")

    # ---- 加载点云并应用理论旋转 ----
    print("\n[4/5] 加载点云并应用理论旋转...")
    merged = o3d.geometry.PointCloud()
    merged_color_coded = o3d.geometry.PointCloud()
    cmap = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1],
        [0.5, 0.5, 0.5], [0.7, 0.3, 0.3], [0.3, 0.7, 0.3], [0.3, 0.3, 0.7],
    ])

    poses = []
    t0 = time.time()
    for k, idx in enumerate(sample_indices):
        path = pcd_files[idx]
        fid = int(os.path.splitext(os.path.basename(path))[0])
        pcd = o3d.io.read_point_cloud(path)

        if len(pcd.points) < 200:
            print(f"  ⚠ #{fid} 点数太少 ({len(pcd.points)}), 跳过")
            continue

        # 创建旋转矩阵（绕 Y 轴旋转，旋转轴在 (axis_x, axis_z)）
        angle_rad = np.radians(rotation_angles[idx])
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # 先平移到旋转轴，再旋转，再平移回来
        # T = T_back @ R @ T_to_axis
        T_to_axis = np.array([
            [1, 0, 0, -axis_x],
            [0, 1, 0, 0],
            [0, 0, 1, -axis_z],
            [0, 0, 0, 1]
        ])

        R = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

        T_back = np.array([
            [1, 0, 0, axis_x],
            [0, 1, 0, 0],
            [0, 0, 1, axis_z],
            [0, 0, 0, 1]
        ])

        # 组合变换
        T = T_back @ R @ T_to_axis

        # 应用变换
        pcd_aligned = copy.deepcopy(pcd)
        pcd_aligned.transform(T)
        merged += pcd_aligned

        # 染色版
        coded = copy.deepcopy(pcd_aligned)
        color = cmap[k % len(cmap)]
        coded.paint_uniform_color(color)
        merged_color_coded += coded

        poses.append({"id": fid, "pose": T.tolist()})

        if (k + 1) % 5 == 0 or k == len(sample_indices) - 1:
            print(f"  {k+1}/{len(sample_indices)}  #{fid}  "
                  f"旋转={rotation_angles[idx]:.1f}°  "
                  f"{len(pcd.points)} pts")

    print(f"  加载 {len(sample_indices)} 帧，耗时 {time.time()-t0:.1f}s")
    print(f"  合并 {len(merged.points):,} 点")

    # ---- 后处理 ----
    print("\n[5/5] 后处理...")
    merged_clean, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    merged_clean = merged_clean.voxel_down_sample(3.0)
    print(f"  去离群+降采样: {len(merged_clean.points):,} 点")

    # ---- 输出 ----
    raw_path = os.path.join(out_dir, "merged_raw.ply")
    clean_path = os.path.join(out_dir, "merged_clean.ply")
    coded_path = os.path.join(out_dir, "merged_color_coded.ply")

    o3d.io.write_point_cloud(raw_path, merged)
    o3d.io.write_point_cloud(clean_path, merged_clean)
    o3d.io.write_point_cloud(coded_path, merged_color_coded)
    print(f"  → {raw_path}")
    print(f"  → {clean_path}")
    print(f"  → {coded_path}")

    # 位姿
    poses_path = os.path.join(out_dir, "poses.json")
    with open(poses_path, 'w') as f:
        json.dump(poses, f, indent=2)

    print()
    print("下一步:")
    print(f"  1. 用 MeshLab 打开 {coded_path} → 检查是否覆盖 360°")
    print(f"  2. 打开 {clean_path} → 看完整形状")
    print(f"  3. OK 后：python 07_stage4_align.py --input {args.input}")


if __name__ == "__main__":
    main()
