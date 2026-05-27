"""
06_stage3_register_v2.py — 正确处理两圈数据的配准

关键改进：
1. 不使用闭环边（避免错误约束）
2. 只使用相邻边和跳跃边
3. 让 Pose Graph 自由优化位姿
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
    import cv2
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)


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


def ransac_register(src_ds, tgt_ds, src_fpfh, tgt_fpfh, voxel):
    """FPFH + RANSAC 粗配准"""
    threshold = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds, src_fpfh, tgt_fpfh, True,
        threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
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
    """Color ICP"""
    current = init_T.copy()
    schedule = [(voxel * 4, 30), (voxel * 2, 20), (voxel * 1, 15)]
    last_result = None
    for v, max_iter in schedule:
        src_ds = src.voxel_down_sample(v)
        tgt_ds = tgt.voxel_down_sample(v)
        if len(src_ds.points) < 30 or len(tgt_ds.points) < 30:
            continue
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
            return icp_geometric(src, tgt, init_T, voxel)
    return last_result, current


def pairwise_register(src, tgt, src_ds, tgt_ds, src_fpfh, tgt_fpfh,
                       voxel, use_color_icp=True):
    """一对帧的完整配准"""
    log = {}

    # 1. FPFH + RANSAC 粗配
    ransac = ransac_register(src_ds, tgt_ds, src_fpfh, tgt_fpfh, voxel)
    log["ransac_fitness"] = float(ransac.fitness)
    log["ransac_rmse"] = float(ransac.inlier_rmse)

    if ransac.fitness < 0.1:
        return np.eye(4), np.eye(6), False, log

    # 2. 几何 ICP 精配
    geo_result, T_after_geo = icp_geometric(src, tgt, ransac.transformation, voxel)
    if geo_result is None:
        return ransac.transformation, np.eye(6), False, log
    log["geo_fitness"] = float(geo_result.fitness)
    log["geo_rmse"] = float(geo_result.inlier_rmse)

    # 3. Color ICP（可选）
    if use_color_icp and len(np.asarray(src.colors)) > 0 and len(np.asarray(tgt.colors)) > 0:
        col_result, T_final = icp_colored(src, tgt, T_after_geo, voxel)
        if col_result is not None:
            log["color_fitness"] = float(col_result.fitness)
            log["color_rmse"] = float(col_result.inlier_rmse)
        else:
            T_final = T_after_geo
            log["color_fitness"] = log["geo_fitness"]
    else:
        T_final = T_after_geo
        log["color_fitness"] = log["geo_fitness"]

    # 4. 计算信息矩阵
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, voxel * 1.4, T_final)

    success = log["color_fitness"] > 0.3
    return T_final, info, success, log


def build_pose_graph_no_loop(pcds_data, voxel, use_color_icp=True):
    """
    构建 pose graph（不使用闭环边）

    只使用相邻边和跳跃边，让 Pose Graph 自由优化位姿
    """
    n = len(pcds_data)
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # 添加第一个节点（pose = identity）
    odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(odometry))

    edges_log = []

    # ---- 相邻边 ----
    print(f"\n  构建相邻边...")
    for i in range(n - 1):
        src_data = pcds_data[i + 1]
        tgt_data = pcds_data[i]
        T, info, success, log = pairwise_register(
            src_data["pcd"], tgt_data["pcd"],
            src_data["ds"], tgt_data["ds"],
            src_data["fpfh"], tgt_data["fpfh"],
            voxel, use_color_icp)

        if success:
            # 累积里程计
            odometry = T @ odometry
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i + 1, i, T, info, uncertain=False))
            status = "OK"
        else:
            # 失败时假设 identity
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i + 1, i, np.eye(4),
                    np.eye(6) * 0.01,
                    uncertain=True))
            status = "FAIL"

        log.update({"i": i, "j": i + 1, "type": "adjacent", "status": status})
        edges_log.append(log)

        if (i + 1) % 5 == 0 or i == n - 2:
            print(f"    {i+1}/{n-1}  fit={log.get('color_fitness',0):.2f}  {status}")

    # ---- 跳跃边（每隔 2 帧） ----
    print(f"\n  构建跳跃边 (i <-> i+2)...")
    for i in range(n - 2):
        src_data = pcds_data[i + 2]
        tgt_data = pcds_data[i]
        T, info, success, log = pairwise_register(
            src_data["pcd"], tgt_data["pcd"],
            src_data["ds"], tgt_data["ds"],
            src_data["fpfh"], tgt_data["fpfh"],
            voxel, use_color_icp)
        log.update({"i": i, "j": i + 2, "type": "skip2",
                    "status": "OK" if success else "FAIL"})
        edges_log.append(log)
        if success:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i + 2, i, T, info, uncertain=True))

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


def main():
    parser = argparse.ArgumentParser(description="Stage 3: 正确处理两圈数据")
    parser.add_argument("--input", required=True)
    parser.add_argument("--voxel", type=float, default=8.0,
                        help="配准用降采样体素 mm (默认 8)")
    parser.add_argument("--no-color-icp", action="store_true",
                        help="禁用 Color ICP")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 3: 正确处理两圈数据的配准")
    print("=" * 64)
    print(f"voxel = {args.voxel} mm")
    print(f"Color ICP: {'OFF' if args.no_color_icp else 'ON'}")
    print(f"注意：不使用闭环边，避免错误约束旋转")
    print()

    pcds_dir = os.path.join(args.input, "pcds_seg")
    if not os.path.isdir(pcds_dir):
        print(f"✗ 找不到 {pcds_dir}，先跑 05_stage2_segment.py")
        sys.exit(1)

    out_dir = args.output or os.path.join(args.input, "output_v2")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 加载所有点云 ----
    print("[1/4] 加载点云 + 预处理...")
    pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if len(pcd_files) < 3:
        print(f"✗ 帧数太少 ({len(pcd_files)})")
        sys.exit(1)

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
    print("\n[2/4] 构建 Pose Graph（不使用闭环边）...")
    t0 = time.time()
    pose_graph, edges_log = build_pose_graph_no_loop(
        pcds_data, args.voxel,
        use_color_icp=not args.no_color_icp,
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
        pcd_aligned = copy.deepcopy(data["pcd"])
        pcd_aligned.transform(pose)
        merged += pcd_aligned

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
    print(f"  → {coded_path}")

    # 位姿
    poses_path = os.path.join(out_dir, "poses.json")
    with open(poses_path, 'w') as f:
        json.dump(poses, f, indent=2)

    # 边日志
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
    print(f"  1. 用 MeshLab 打开 {coded_path} → 检查是否覆盖 360°")
    print(f"  2. 打开 {clean_path} → 看完整形状")
    print(f"  3. OK 后：python 07_stage4_align.py --input {args.input}")


if __name__ == "__main__":
    main()
