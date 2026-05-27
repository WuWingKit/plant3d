"""
06_stage3_register.py — Pose Graph 多帧配准（纯 6 DOF，不假设转轴）

算法：
  1. 加载 stage 2 的分离后点云
  2. 相邻帧 FPFH+RANSAC 粗配 → ICP 精配（含 Color ICP）
  3. 闭环帧（首尾、对面位置）也配准
  4. 构建 Open3D PoseGraph，相邻为确定边，闭环为不确定边
  5. global_optimization 求解所有帧位姿
  6. 用解出的位姿合并所有点云

为什么这样设计：
  - 不假设旋转轴 → 兼容任意拍摄方式（转盘、手持、绕拍）
  - Pose graph 全局优化 → 单条边失败不致命，其他边能纠正
  - Color ICP 弥补几何对称物体（光滑花瓶口）的歧义

使用：
  python 06_stage3_register.py --input capture_xxx
  python 06_stage3_register.py --input capture_xxx --voxel 8 --loop-closures 4
  python 06_stage3_register.py --input capture_xxx --no-color-icp
"""

import os
import sys
import csv
import json
import glob
import copy
import argparse
import time
import numpy as np

try:
    import cv2
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)


# ============================================================
#  预处理（计算 FPFH 等）
# ============================================================

def preprocess(pcd, voxel):
    """降采样 + 估计法向量 + 计算 FPFH"""
    ds = pcd.voxel_down_sample(voxel)
    if len(ds.points) < 30:
        return None, None
    ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        ds, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
    return ds, fpfh


# ============================================================
#  两两配准（粗 → 精 → 颜色）
# ============================================================

def ransac_register(src_ds, tgt_ds, src_fpfh, tgt_fpfh, voxel):
    """FPFH + RANSAC 粗配准"""
    threshold = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds, src_fpfh, tgt_fpfh, True,
        threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # ransac_n
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def icp_geometric(src, tgt, init_T, voxel):
    """Multi-scale Point-to-Plane ICP"""
    current = init_T.copy()
    schedule = [
        (voxel * 4, 50, voxel * 8),
        (voxel * 2, 30, voxel * 4),
        (voxel * 1, 20, voxel * 2),
    ]
    last_result = None
    for v, max_iter, thr in schedule:
        src_ds = src.voxel_down_sample(v)
        tgt_ds = tgt.voxel_down_sample(v)
        if len(src_ds.points) < 30 or len(tgt_ds.points) < 30:
            continue
        src_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v*2.5, max_nn=30))
        tgt_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v*2.5, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            src_ds, tgt_ds, thr, current,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        current = result.transformation
        last_result = result
    return last_result, current


def icp_colored(src, tgt, init_T, voxel):
    """Color ICP（用颜色 + 几何）"""
    current = init_T.copy()
    schedule = [(voxel * 4, 30), (voxel * 2, 20), (voxel * 1, 15)]
    last_result = None
    for v, max_iter in schedule:
        src_ds = src.voxel_down_sample(v)
        tgt_ds = tgt.voxel_down_sample(v)
        if len(src_ds.points) < 30 or len(tgt_ds.points) < 30:
            continue
        # Color ICP 需要法向
        src_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v*2.5, max_nn=30))
        tgt_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v*2.5, max_nn=30))
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                src_ds, tgt_ds, v,
                current,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=max_iter,
                ),
            )
            current = result.transformation
            last_result = result
        except Exception as e:
            # 有些版本的 Open3D 调用方式不同，回退到几何 ICP
            return icp_geometric(src, tgt, init_T, voxel)
    return last_result, current


def rotation_matrix_y(angle_deg, center, sign=-1):
    """绕过 center 的 Y 轴旋转（用于生成角度初值）"""
    theta = np.deg2rad(angle_deg * sign)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy, cz = center[0], center[1], center[2]
    return np.array([
        [c,  0, s, cx*(1-c) - cz*s],
        [0,  1, 0, 0],
        [-s, 0, c, cz*(1-c) + cx*s],
        [0,  0, 0, 1],
    ])


def pairwise_register(src, tgt, src_ds, tgt_ds, src_fpfh, tgt_fpfh,
                       voxel, use_color_icp=True, T_angle_init=None):
    """
    一对帧的完整配准。

    T_angle_init: 由转盘角度推算的初始变换矩阵（4x4）。
      - 提供时：先用角度初值做 ICP，同时跑 RANSAC 验证；
        若 RANSAC 结果和角度初值差 < 10°，则用 RANSAC（更精确）；
        否则坚持用角度初值（RANSAC 找错了）。
      - 不提供时：退回原来的 FPFH → RANSAC → ICP 流程。

    关键动机：花瓶光滑 → FPFH 描述子区分度低 → RANSAC 可能给出
    接近 0° 的错误 T → ICP 从错误初值收敛到局部最优（0.5°而非正确3°）。
    用已知角度作初值可避免此问题。
    """
    log = {}

    # 确定 ICP 初值
    if T_angle_init is not None:
        T_init_for_icp = T_angle_init.copy()
        log["init_source"] = "angle"
    else:
        ransac = ransac_register(src_ds, tgt_ds, src_fpfh, tgt_fpfh, voxel)
        log["ransac_fitness"] = float(ransac.fitness)
        log["ransac_rmse"] = float(ransac.inlier_rmse)
        if ransac.fitness < 0.1:
            return np.eye(4), np.eye(6), False, log
        T_init_for_icp = ransac.transformation
        log["init_source"] = "ransac"

    # 有角度初值时，也跑 RANSAC 做交叉验证
    if T_angle_init is not None:
        ransac = ransac_register(src_ds, tgt_ds, src_fpfh, tgt_fpfh, voxel)
        log["ransac_fitness"] = float(ransac.fitness)
        log["ransac_rmse"] = float(ransac.inlier_rmse)
        if ransac.fitness > 0.3:
            R_ransac = ransac.transformation[:3, :3]
            R_angle  = T_angle_init[:3, :3]
            cos_val  = np.clip((np.trace(R_ransac @ R_angle.T) - 1) / 2, -1, 1)
            angle_diff = np.degrees(np.arccos(cos_val))
            log["ransac_vs_angle_deg"] = float(angle_diff)
            if angle_diff < 10:
                T_init_for_icp = ransac.transformation
                log["init_source"] = "ransac_verified"
            else:
                log["init_source"] = "angle_kept (ransac diverged)"

    # 几何 ICP（从正确初值开始）
    geo_result, T_after_geo = icp_geometric(src, tgt, T_init_for_icp, voxel)
    if geo_result is None:
        return T_init_for_icp, np.eye(6), False, log
    log["geo_fitness"] = float(geo_result.fitness)
    log["geo_rmse"]    = float(geo_result.inlier_rmse)

    # Color ICP（可选）
    if use_color_icp and len(np.asarray(src.colors)) > 0 and len(np.asarray(tgt.colors)) > 0:
        col_result, T_final = icp_colored(src, tgt, T_after_geo, voxel)
        if col_result is not None:
            log["color_fitness"] = float(col_result.fitness)
            log["color_rmse"]    = float(col_result.inlier_rmse)
        else:
            T_final = T_after_geo
            log["color_fitness"] = log["geo_fitness"]
    else:
        T_final = T_after_geo
        log["color_fitness"] = log["geo_fitness"]

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, voxel * 1.4, T_final)

    success = log["color_fitness"] > 0.3
    return T_final, info, success, log


# ============================================================
#  Pose Graph 构建和优化
# ============================================================

def build_pose_graph(pcds_data, voxel, loop_closures=2, use_color_icp=True,
                     deg_per_frame=None, rotation_center=None):
    """构建 pose graph，用角度初值解决 ICP 卡在局部最优的问题"""
    n = len(pcds_data)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(odometry))

    edges_log = []
    center = rotation_center if rotation_center is not None else np.zeros(3)

    def angle_init(i, j):
        """相邻帧的角度初值矩阵。i→j 表示 j 帧相对 i 帧的变换"""
        if deg_per_frame is None:
            return None
        angle = deg_per_frame * (j - i)
        # 绕旋转中心的 Y 轴旋转（花瓶在转盘上，旋转轴是竖直轴 Y）
        return rotation_matrix_y(angle, center, sign=-1)

    # ---- 相邻边 ----
    print(f"\n  构建相邻边...")
    for i in range(n - 1):
        src_data = pcds_data[i + 1]
        tgt_data = pcds_data[i]
        T_init = angle_init(i, i + 1)
        T, info, success, log = pairwise_register(
            src_data["pcd"], tgt_data["pcd"],
            src_data["ds"], tgt_data["ds"],
            src_data["fpfh"], tgt_data["fpfh"],
            voxel, use_color_icp,
            T_angle_init=T_init)

        if success:
            odometry = T @ odometry
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i + 1, i, T, info, uncertain=False))
            status = "OK"
        else:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i + 1, i, np.eye(4), np.eye(6) * 0.01, uncertain=True))
            status = "FAIL"

        log.update({"i": i, "j": i + 1, "type": "adjacent", "status": status})
        edges_log.append(log)

        if (i + 1) % 5 == 0 or i == n - 2:
            angle_out = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
            print(f"    {i+1}/{n-1}  fit={log.get('color_fitness',0):.2f}  "
                  f"angle={angle_out:.2f}°  {log.get('init_source','')}  {status}")

    # ---- 跳跃边 ----
    print(f"\n  构建跳跃边 (i <-> i+2)...")
    for i in range(n - 2):
        src_data = pcds_data[i + 2]
        tgt_data = pcds_data[i]
        T_init = angle_init(i, i + 2)
        T, info, success, log = pairwise_register(
            src_data["pcd"], tgt_data["pcd"],
            src_data["ds"], tgt_data["ds"],
            src_data["fpfh"], tgt_data["fpfh"],
            voxel, use_color_icp, T_angle_init=T_init)
        log.update({"i": i, "j": i + 2, "type": "skip2",
                    "status": "OK" if success else "FAIL"})
        edges_log.append(log)
        if success:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i + 2, i, T, info, uncertain=True))

    # ---- 闭环边 ----
    print(f"\n  构建闭环边...")
    pairs = [(0, n - 1), (0, n // 2)]
    if loop_closures > 2:
        pairs.append((n // 4, 3 * n // 4))
    if loop_closures > 3:
        pairs.append((n // 4, n - 1))

    for i, j in pairs[:loop_closures]:
        if i == j or i < 0 or j >= n:
            continue
        src_data = pcds_data[j]
        tgt_data = pcds_data[i]
        T_init = angle_init(i, j)
        T, info, success, log = pairwise_register(
            src_data["pcd"], tgt_data["pcd"],
            src_data["ds"], tgt_data["ds"],
            src_data["fpfh"], tgt_data["fpfh"],
            voxel, use_color_icp, T_angle_init=T_init)
        log.update({"i": i, "j": j, "type": "loop",
                    "status": "OK" if success else "FAIL"})
        edges_log.append(log)
        if success:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    j, i, T, info, uncertain=True))
            angle_out = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
            print(f"    闭环 ({i},{j})  fit={log.get('color_fitness',0):.2f}  "
                  f"angle={angle_out:.2f}°  OK")
        else:
            print(f"    闭环 ({i},{j})  fit={log.get('color_fitness',0):.2f}  FAIL")

    return pose_graph, edges_log


def optimize_pose_graph(pose_graph, voxel):
    """全局优化"""
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel * 1.4,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    return pose_graph


# ============================================================
#  找到一圈的位置
# ============================================================

def compute_similarity(pcd1, pcd2):
    """计算两个点云的相似度（基于质心距离和大小）"""
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    centroid1 = pts1.mean(axis=0)
    centroid2 = pts2.mean(axis=0)
    centroid_dist = np.linalg.norm(centroid2 - centroid1)
    size_diff = abs(len(pts1) - len(pts2)) / max(len(pts1), len(pts2))
    return centroid_dist + size_diff * 100


def find_one_turn(pcd_files, search_start=100, search_end=150):
    """找到转完一圈的帧索引（对比第一帧）"""
    ref_pcd = o3d.io.read_point_cloud(pcd_files[0])
    best_similarity = float('inf')
    best_frame = search_start

    for i in range(search_start, min(search_end, len(pcd_files))):
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        sim = compute_similarity(ref_pcd, pcd)
        if sim < best_similarity:
            best_similarity = sim
            best_frame = i

    return best_frame


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Pose Graph 配准")
    parser.add_argument("--input", required=True)
    parser.add_argument("--voxel", type=float, default=8.0)
    parser.add_argument("--loop-closures", type=int, default=3)
    parser.add_argument("--no-color-icp", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--deg-per-frame", type=float, default=None,
                        help="每帧旋转角度（度）。不指定时从 metadata.json 自动算")
    parser.add_argument("--rotation-center", type=float, nargs=3, default=None,
                        help="旋转中心 (x y z) mm。不指定时用原点")
    parser.add_argument("--search-start", type=int, default=100,
                        help="搜索一圈位置的起始帧 (默认 100)")
    parser.add_argument("--search-end", type=int, default=150,
                        help="搜索一圈位置的结束帧 (默认 150)")
    parser.add_argument("--full-turns", type=int, default=1,
                        help="使用几圈 (默认 1)")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                        help="在一圈内抽帧比例 (默认 1.0，不抽帧)")
    parser.add_argument("--frame-range", type=int, nargs=2, default=None,
                        help="手动指定帧范围 start end (如 --frame-range 5 123)")
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 3: Pose Graph 配准（6 DOF + Color ICP + 角度初值）")
    print("=" * 64)

    # ---- 读取 metadata，算每帧角度 ----
    import json as _json
    deg_per_frame = args.deg_per_frame
    rotation_center = np.array(args.rotation_center) if args.rotation_center else None

    meta_path = os.path.join(args.input, "metadata.json")
    if deg_per_frame is None and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)
        fps = meta.get("fps_actual", 1.0)
        speed = meta.get("rotation_speed_deg_per_sec", 0)
        if speed > 0 and fps > 0:
            deg_per_frame = speed / fps
            print(f"  从 metadata 读取: 转速={speed:.2f}°/s, fps={fps:.2f}")
            print(f"  → 每帧旋转 {deg_per_frame:.3f}°")

    if rotation_center is None:
        picked = os.path.join(args.input, "picked_center.json")
        if os.path.isfile(picked):
            with open(picked) as f:
                rotation_center = np.array(_json.load(f)["center"])
            print(f"  旋转中心从 picked_center.json: "
                  f"({rotation_center[0]:.1f}, {rotation_center[1]:.1f}, {rotation_center[2]:.1f})")
        else:
            rotation_center = np.zeros(3)
            print(f"  旋转中心: 原点 (0,0,0)")

    if deg_per_frame:
        print(f"  每帧角度初值: {deg_per_frame:.3f}°（给 ICP 用，避免从单位矩阵开始）")
    else:
        print(f"  ⚠ 未知每帧角度，ICP 将从 RANSAC 结果开始")

    print(f"  voxel={args.voxel}mm, loop_closures={args.loop_closures}, "
          f"Color ICP={'OFF' if args.no_color_icp else 'ON'}")
    print()

    pcds_dir = os.path.join(args.input, "pcds_seg")
    if not os.path.isdir(pcds_dir):
        print(f"✗ 找不到 {pcds_dir}，先跑 05_stage2_segment.py")
        sys.exit(1)

    out_dir = args.output or os.path.join(args.input, "output_v2")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 找到一圈并限制帧数 ----
    pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if len(pcd_files) < 3:
        print(f"✗ 帧数太少 ({len(pcd_files)})")
        sys.exit(1)

    one_turn_frame = find_one_turn(pcd_files, args.search_start, args.search_end)

    if args.frame_range:
        start_f, end_f = args.frame_range
        pcd_files = [f for f in pcd_files
                     if start_f <= int(os.path.splitext(os.path.basename(f))[0]) <= end_f]
        print(f"  手动帧范围: {start_f}-{end_f} = {len(pcd_files)} 帧")
    else:
        n_frames_per_turn = one_turn_frame
        frames_to_use = min(args.full_turns, 2) * n_frames_per_turn
        pcd_files = pcd_files[:frames_to_use]
        print(f"  一圈 = {n_frames_per_turn} 帧, 使用 {args.full_turns} 圈 = {len(pcd_files)} 帧")

    if args.sample_ratio < 1.0:
        n_samples = max(3, int(len(pcd_files) * args.sample_ratio))
        sample_indices = np.linspace(0, len(pcd_files) - 1, n_samples, dtype=int)
        pcd_files = [pcd_files[i] for i in sample_indices]
        print(f"  抽帧: {n_samples} 帧 (比例 {args.sample_ratio*100:.0f}%)")

    # ---- 加载点云 ----
    print("\n[1/4] 加载点云 + 预处理...")

    pcds_data = []
    t0 = time.time()
    for k, path in enumerate(pcd_files):
        fid = int(os.path.splitext(os.path.basename(path))[0])
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) < 200:
            print(f"  ⚠ #{fid} 点数太少 ({len(pcd.points)}), 跳过")
            continue
        ds, fpfh = preprocess(pcd, args.voxel)
        if ds is None:
            print(f"  ⚠ #{fid} 预处理失败")
            continue
        pcds_data.append({"id": fid, "pcd": pcd, "ds": ds, "fpfh": fpfh})
        if (k + 1) % 5 == 0 or k == len(pcd_files) - 1:
            print(f"  {k+1}/{len(pcd_files)}  #{fid}  {len(pcd.points)} pts")
    print(f"  加载 {len(pcds_data)} 帧，耗时 {time.time()-t0:.1f}s")

    if len(pcds_data) < 3:
        print("✗ 有效帧不足 3")
        sys.exit(1)

    # ---- 构建 Pose Graph ----
    print("\n[2/4] 构建 Pose Graph...")
    t0 = time.time()
    pose_graph, edges_log = build_pose_graph(
        pcds_data, args.voxel,
        loop_closures=args.loop_closures,
        use_color_icp=not args.no_color_icp,
        deg_per_frame=deg_per_frame,
        rotation_center=rotation_center,
    )
    print(f"  节点 {len(pose_graph.nodes)}, 边 {len(pose_graph.edges)}  "
          f"耗时 {time.time()-t0:.1f}s")

    # ---- 全局优化 ----
    print("\n[3/4] 全局优化...")
    t0 = time.time()
    pose_graph = optimize_pose_graph(pose_graph, args.voxel)
    print(f"  完成，耗时 {time.time()-t0:.1f}s")

    # ---- 合并 ----
    print("\n[4/4] 用优化后的位姿合并...")
    merged = o3d.geometry.PointCloud()
    merged_color_coded = o3d.geometry.PointCloud()
    cmap = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1],
        [0.5, 0.5, 0.5], [0.7, 0.3, 0.3], [0.3, 0.7, 0.3], [0.3, 0.3, 0.7],
    ])

    poses = []
    for i, data in enumerate(pcds_data):
        pose = pose_graph.nodes[i].pose
        # pose 是 i 帧到参考帧的变换
        pcd_aligned = copy.deepcopy(data["pcd"])
        pcd_aligned.transform(pose)
        merged += pcd_aligned

        # 染色版（每帧不同颜色）
        coded = copy.deepcopy(pcd_aligned)
        color = cmap[i % len(cmap)]
        coded.paint_uniform_color(color)
        merged_color_coded += coded

        poses.append({"id": data["id"], "pose": pose.tolist()})

    print(f"  合并 {len(pcds_data)} 帧 → {len(merged.points):,} 点")

    # 简单后处理
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
    print(f"  → {coded_path}  (每帧不同颜色，用 MeshLab 查对齐质量)")

    # 位姿
    poses_path = os.path.join(out_dir, "poses.json")
    with open(poses_path, 'w') as f:
        json.dump(poses, f, indent=2)

    # 边日志（详细配准记录）
    log_path = os.path.join(out_dir, "registration_log.csv")
    with open(log_path, 'w', newline='') as f:
        if edges_log:
            keys = sorted({k for e in edges_log for k in e.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(edges_log)

    # 统计
    n_total = len(edges_log)
    n_ok = sum(1 for e in edges_log if e.get("status") == "OK")
    print()
    print(f"配准统计: {n_ok}/{n_total} 边成功 ({100*n_ok/n_total:.0f}%)")
    types = {}
    for e in edges_log:
        t = e.get("type", "?")
        types.setdefault(t, [0, 0])
        types[t][0] += 1
        if e.get("status") == "OK":
            types[t][1] += 1
    for t, (tot, ok) in types.items():
        print(f"  {t}: {ok}/{tot}")

    print()
    print("下一步:")
    print(f"  1. 用 MeshLab 打开 {coded_path} → 不同颜色的帧应该叠在一起")
    print(f"     如果出现彩色花瓣状分布 → 配准失败")
    print(f"  2. 打开 {clean_path} → 看花瓶完整 360° 形状")
    print(f"  3. OK 后：python 07_stage4_align.py --input {args.input}")


if __name__ == "__main__":
    main()
