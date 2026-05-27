"""
09_postprocess.py — Stage 3 点云后处理 → 高质量 Mesh

流水线：
  1. SOR 统计滤波     去漂浮孤立噪点
  2. ROR 半径滤波     去边缘飞点
  3. 体素降采样       合并配准误差导致的双层点
  4. RANSAC 去残留平面（可选，如果 Stage 2 没去干净）
  5. 泊松重建 depth=10
  6. Taubin 平滑      mesh 表面光滑，不缩水不失细节
  7. 删除孤立小面片

使用：
  python 09_postprocess.py --input capture_xxx

  # 从 merged_clean.ply（Stage 3 output）开始
  # 输出到 capture_xxx/output_v2/plant_model_clean.obj

调参建议：
  --sor-k 20 --sor-std 2.0     SOR 参数（std 越小删越多）
  --ror-radius 10 --ror-min 5  ROR 参数（radius mm，min 近邻数）
  --voxel 2.0                  降采样体素 mm
  --poisson-depth 10           泊松深度（9/10/11）
  --taubin-iter 20             Taubin 迭代次数（越多越光滑）
  --remove-plane               启用 RANSAC 去残留平面（默认关）

注意：
  - 每步都会打印点数/面数变化，方便调参
  - 同时保留 plant_model_clean.ply 和 .stl
"""

import os
import sys
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("✗ 需要 open3d: pip install open3d")
    sys.exit(1)


# ============================================================
#  点云后处理（步骤 1-4）
# ============================================================

def step_sor(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    统计离群点滤波（SOR）。
    去除"邻域平均距离 > μ + std_ratio×σ"的点。
    std_ratio 越小 → 删得越激进。
    2.0 对应 ~5% 离群点（正态分布 95% 置信区间外）
    1.65 对应严格版本（5% 尾部）
    """
    n_before = len(pcd.points)
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio)
    n_after = len(pcd_clean.points)
    removed = n_before - n_after
    print(f"  SOR (k={nb_neighbors}, std={std_ratio}): "
          f"{n_before:,} → {n_after:,}  删除 {removed:,} 点 "
          f"({removed/n_before*100:.1f}%)")
    return pcd_clean


def step_ror(pcd, nb_points=5, radius=10.0):
    """
    半径离群点滤波（ROR）。
    半径 radius(mm) 内近邻 < nb_points 的点视为边缘飞点删除。
    适合去除物体轮廓外的稀疏飞点。
    """
    n_before = len(pcd.points)
    pcd_clean, ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius)
    n_after = len(pcd_clean.points)
    removed = n_before - n_after
    print(f"  ROR (min_pts={nb_points}, r={radius}mm): "
          f"{n_before:,} → {n_after:,}  删除 {removed:,} 点 "
          f"({removed/n_before*100:.1f}%)")
    return pcd_clean


def step_voxel(pcd, voxel_size=2.0):
    """
    体素降采样。
    合并配准误差导致的"双层点"，同时控制点密度。
    每个体素内所有点用加权质心替代。
    """
    n_before = len(pcd.points)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    n_after = len(pcd_down.points)
    print(f"  体素降采样 (voxel={voxel_size}mm): "
          f"{n_before:,} → {n_after:,} 点")
    return pcd_down


def step_upsample(pcd, voxel_size=1.0, points_per_voxel=3):
    """
    上采样：在每个体素内插值新点，增加点云密度。

    原理：
    1. 对每个体素内的点，计算质心
    2. 在质心附近随机生成新点（沿法向量方向偏移）
    3. 新点继承原有点的颜色（插值）

    voxel_size: 体素尺寸 mm（越小插值点越多）
    points_per_voxel: 每个体素插值的点数（默认 3）
    """
    n_before = len(pcd.points)

    # 计算法向量（用于插值方向）
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 获取数据
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(np.asarray(pcd.colors)) > 0 else None
    normals = np.asarray(pcd.normals) if len(np.asarray(pcd.normals)) > 0 else None

    # 计算每个点所属的体素索引
    origin = pts.min(axis=0)
    voxel_indices = {}
    for i, pt in enumerate(pts):
        voxel_idx = tuple(((pt - origin) / voxel_size).astype(int))
        if voxel_idx not in voxel_indices:
            voxel_indices[voxel_idx] = []
        voxel_indices[voxel_idx].append(i)

    # 在每个体素内插值新点
    new_points = []
    new_colors = []
    n_voxels_with_points = 0

    for voxel_idx, point_indices in voxel_indices.items():
        if len(point_indices) < 2:
            continue

        n_voxels_with_points += 1

        # 获取体素内的点
        voxel_pts = pts[point_indices]
        centroid = voxel_pts.mean(axis=0)

        # 计算体素内点的标准差（用于控制插值范围）
        std = voxel_pts.std(axis=0).mean()

        # 在质心附近生成新点（沿法向量方向）
        if normals is not None:
            voxel_normals = normals[point_indices]
            avg_normal = voxel_normals.mean(axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)

            # 在法向量方向上插值
            for _ in range(points_per_voxel):
                offset = np.random.uniform(-std, std)
                new_pt = centroid + offset * avg_normal
                new_points.append(new_pt)

                # 颜色插值（取最近的原有点的颜色）
                if colors is not None:
                    dists = np.linalg.norm(voxel_pts - new_pt, axis=1)
                    nearest_idx = point_indices[np.argmin(dists)]
                    new_colors.append(colors[nearest_idx])

    # 合并新点和原有点
    if new_points:
        all_points = np.vstack([pts, np.array(new_points)])
        if colors is not None and new_colors:
            all_colors = np.vstack([colors, np.array(new_colors)])
        else:
            all_colors = colors

        pcd_upsampled = o3d.geometry.PointCloud()
        pcd_upsampled.points = o3d.utility.Vector3dVector(all_points)
        if all_colors is not None:
            pcd_upsampled.colors = o3d.utility.Vector3dVector(all_colors)
        if normals is not None:
            pcd_upsampled.normals = o3d.utility.Vector3dVector(
                np.vstack([normals, np.zeros((len(new_points), 3))]))
    else:
        pcd_upsampled = pcd

    n_after = len(pcd_upsampled.points)
    print(f"  上采样 (voxel={voxel_size}mm, {points_per_voxel}点/体素): "
          f"{n_before:,} → {n_after:,} 点 (+{n_after-n_before:,})")
    print(f"  有效体素: {n_voxels_with_points:,}")
    return pcd_upsampled


def step_recolor_from_images(pcd, input_dir, poses, calib, max_dist=3000):
    """
    从原始照片重新采样颜色。

    原理：
    1. 读取每帧的对齐照片（aligned/XXXX.png）
    2. 用位姿将点云变换到每帧的相机坐标系
    3. 用相机内参投影到图像平面
    4. 从图像中采样颜色
    5. 对每个点，取所有可见帧的颜色加权平均（距离越近权重越大）

    pcd: 点云
    input_dir: capture 目录
    poses: 位姿列表 [{'id': 5, 'pose': 4x4}, ...]
    calib: 标定数据
    max_dist: 最大深度 mm（超过的点不采样）
    """
    import cv2

    pts = np.asarray(pcd.points)
    n_points = len(pts)

    # 相机内参（aligned 照片是深度图对齐到颜色图的，用深度相机内参）
    depth_K = np.array(calib['depth']['K'])
    fx, fy = depth_K[0, 0], depth_K[1, 1]
    cx, cy = depth_K[0, 2], depth_K[1, 2]

    # 累积颜色和权重
    color_sum = np.zeros((n_points, 3))
    weight_sum = np.zeros(n_points)

    aligned_dir = os.path.join(input_dir, "aligned")

    for pose_data in poses:
        frame_id = pose_data['id']
        T = np.array(pose_data['pose'])  # 4x4 位姿矩阵（从参考帧到每帧）

        # 读取对齐照片
        img_path = os.path.join(aligned_dir, f"{frame_id:04d}.png")
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # poses 是从参考帧到每帧的变换
        # 点云在参考帧坐标系，需要变换到每帧的相机坐标系
        # 注意：aligned 照片已经是深度图对齐到颜色图的，所以用深度相机内参
        pts_cam = (T[:3, :3] @ pts.T + T[:3, 3:]).T

        # 投影到图像平面
        u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(int)
        v = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(int)

        # 过滤有效投影（在图像范围内且深度合理）
        h, w = img.shape[:2]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & \
                (pts_cam[:, 2] > 0) & (pts_cam[:, 2] < max_dist)

        if not np.any(valid):
            continue

        # 采样颜色（BGR → RGB，归一化到 0-1）
        sampled_bgr = img[v[valid], u[valid]]
        sampled_rgb = sampled_bgr[:, ::-1].astype(float) / 255.0

        # 权重：基于深度（越近权重越大）
        depth = pts_cam[valid, 2]
        weight = 1.0 / (depth / 1000.0)  # 归一化到米

        # 累积
        color_sum[valid] += sampled_rgb * weight[:, np.newaxis]
        weight_sum[valid] += weight

    # 计算加权平均颜色
    valid_points = weight_sum > 0
    if not np.any(valid_points):
        print("  警告：没有点被成功采样颜色")
        return pcd

    colors = np.zeros((n_points, 3))
    colors[valid_points] = color_sum[valid_points] / weight_sum[valid_points, np.newaxis]

    # 对没有被采样的点，保留原颜色
    if len(np.asarray(pcd.colors)) > 0:
        colors[~valid_points] = np.asarray(pcd.colors)[~valid_points]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"  颜色重采样: {valid_points.sum():,}/{n_points:,} 点被更新 "
          f"({valid_points.sum()/n_points*100:.1f}%)")

    return pcd


def step_remove_plane(pcd, dist_threshold=8.0, min_plane_ratio=0.05):
    """
    RANSAC 去除残留水平平面（Stage 2 没去干净的转盘/桌面）。
    只在平面内点数 > 总点数 × min_plane_ratio 时才执行去除。
    """
    n_before = len(pcd.points)
    if n_before < 1000:
        print("  RANSAC 跳过（点数太少）")
        return pcd

    plane, inliers = pcd.segment_plane(
        distance_threshold=dist_threshold,
        ransac_n=3,
        num_iterations=1000)
    a, b, c, d = plane

    n_inliers = len(inliers)
    ratio = n_inliers / n_before

    if ratio < min_plane_ratio:
        print(f"  RANSAC: 最大平面内点占比 {ratio*100:.1f}% < {min_plane_ratio*100:.0f}%，"
              f"跳过（可能已经干净了）")
        return pcd

    # 法向量接近水平才去（避免误删花瓶侧面）
    normal = np.array([abs(a), abs(b), abs(c)])
    vert_axis = int(np.argmax(normal))
    axis_align = normal[vert_axis]
    if axis_align < 0.80:
        print(f"  RANSAC: 法向量非水平（最大分量 {axis_align:.2f} < 0.80），跳过")
        return pcd

    pts = np.asarray(pcd.points)
    signed = a*pts[:, 0] + b*pts[:, 1] + c*pts[:, 2] + d
    # 取平面"之上"的点（物体所在侧）
    n_above = np.count_nonzero(signed > dist_threshold)
    n_below = np.count_nonzero(signed < -dist_threshold)
    if n_above > n_below:
        keep = signed > dist_threshold
    else:
        keep = signed < -dist_threshold

    pcd_clean = pcd.select_by_index(np.where(keep)[0])
    n_after = len(pcd_clean.points)
    print(f"  RANSAC 去残留平面 (内点占 {ratio*100:.1f}%): "
          f"{n_before:,} → {n_after:,} 点")
    return pcd_clean


# ============================================================
#  泊松重建（步骤 5）
# ============================================================

def step_poisson(pcd, depth=10, density_cut=0.10, normal_radius=8.0, normal_k=30):
    """
    泊松表面重建。

    depth=10 比 Stage 5 默认的 9 更精细。
    normal_radius 从 10 降到 8，棱角更锐利。
    法向量一致化 k=30（比默认 20 更稳定）。
    """
    print(f"  估计法向量 (radius={normal_radius}mm, k={normal_k})...")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_k))
    pcd.orient_normals_consistent_tangent_plane(k=normal_k)

    print(f"  泊松重建 (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    densities = np.asarray(densities)

    # 裁掉低密度顶点（边角碎片）
    threshold = np.quantile(densities, density_cut)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  泊松产出: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面")

    # 顶点颜色（最近邻）
    if len(np.asarray(pcd.colors)) > 0:
        print("  映射顶点颜色...")
        tree = o3d.geometry.KDTreeFlann(pcd)
        mv = np.asarray(mesh.vertices)
        pc = np.asarray(pcd.colors)
        colors = np.zeros_like(mv)
        for i in range(len(mv)):
            _, idx, _ = tree.search_knn_vector_3d(mesh.vertices[i], 1)
            colors[i] = pc[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


# ============================================================
#  Mesh 后处理（步骤 6-7）
# ============================================================

def step_taubin(mesh, n_iter=20, lamb=0.5, mu=-0.53):
    """
    Taubin 平滑。

    比 Laplacian 更好：交替正向（收缩）和负向（膨胀）平滑，
    消除 Laplacian 的"缩水"问题，保留整体形状同时去除高频噪声。

    lamb: 正向步长（0.5 推荐）
    mu:   负向步长（-0.53 推荐，满足 Taubin 条件 |mu| > |lamb|）
    n_iter: 迭代次数。每次迭代 = 一次正向 + 一次负向。
      10 次：轻微平滑，保留细节
      20 次：中度平滑，花瓶表面明显更光滑
      50 次：重度平滑，植物细节丢失但花瓶非常光滑
    """
    n_vert_before = len(mesh.vertices)
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=n_iter,
        lambda_filter=lamb,
        mu=mu)
    mesh.compute_vertex_normals()
    print(f"  Taubin 平滑 ({n_iter} 次迭代): "
          f"{n_vert_before:,} 顶点 → {len(mesh.vertices):,} 顶点（顶点数不变，位置变化）")
    return mesh


def step_remove_small_clusters(mesh, min_triangle_ratio=0.01):
    """
    删除孤立小面片簇（碎片）。
    删除三角形数量 < 总数 × min_triangle_ratio 的连通分量。
    """
    n_tri_before = len(mesh.triangles)
    if n_tri_before == 0:
        return mesh

    # 获取连通分量
    triangle_clusters, cluster_n_triangles, cluster_area = \
        mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # 找最大簇
    max_cluster_id = int(np.argmax(cluster_n_triangles))
    min_triangles = int(n_tri_before * min_triangle_ratio)

    # 保留大簇，删除小簇
    triangles_to_remove = np.array([
        cluster_n_triangles[triangle_clusters[i]] < min_triangles
        for i in range(n_tri_before)
    ])
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    n_tri_after = len(mesh.triangles)
    print(f"  删除孤立小面片 (最小{min_triangles}面): "
          f"{n_tri_before:,} → {n_tri_after:,} 面片")
    return mesh


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3 点云后处理 → 高质量 Mesh")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--source", default="merged_clean.ply",
                        help="输入点云文件名（在 output_v2/ 下，默认 merged_clean.ply）")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录（默认同 output_v2/）")

    # 点云滤波参数
    parser.add_argument("--sor-k", type=int, default=20,
                        help="SOR 近邻数 K（默认 20）")
    parser.add_argument("--sor-std", type=float, default=2.0,
                        help="SOR 标准差倍数（默认 2.0，越小删越多）")
    parser.add_argument("--ror-radius", type=float, default=10.0,
                        help="ROR 搜索半径 mm（默认 10）")
    parser.add_argument("--ror-min", type=int, default=5,
                        help="ROR 最少近邻数（默认 5）")
    parser.add_argument("--voxel", type=float, default=2.0,
                        help="降采样体素 mm（默认 2.0）")
    parser.add_argument("--remove-plane", action="store_true",
                        help="启用 RANSAC 去残留平面（默认关）")

    # 上采样和颜色重采样参数
    parser.add_argument("--upsample", action="store_true",
                        help="启用上采样（增加点云密度）")
    parser.add_argument("--upsample-voxel", type=float, default=1.0,
                        help="上采样体素 mm（默认 1.0，越小插值点越多）")
    parser.add_argument("--recolor", action="store_true",
                        help="从原始照片重新采样颜色（需要 aligned/ 目录）")

    # 泊松参数
    parser.add_argument("--poisson-depth", type=int, default=10,
                        help="泊松深度（默认 10，比 Stage 5 的 9 更精细）")
    parser.add_argument("--density-cut", type=float, default=0.10,
                        help="低密度顶点裁剪比例（默认 0.10）")
    parser.add_argument("--normal-radius", type=float, default=8.0,
                        help="法向量估计半径 mm（默认 8）")

    # Mesh 平滑参数
    parser.add_argument("--taubin-iter", type=int, default=20,
                        help="Taubin 平滑迭代次数（默认 20）"
                             " 10=轻微 20=中度 50=重度")
    parser.add_argument("--no-taubin", action="store_true",
                        help="跳过 Taubin 平滑")
    parser.add_argument("--min-cluster-ratio", type=float, default=0.01,
                        help="孤立面片删除阈值（默认 0.01 = 1%%）")

    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, args.source)

    if not os.path.isfile(input_path):
        print(f"✗ 找不到输入文件: {input_path}")
        print(f"  确认已经跑过 Stage 3 (06_stage3_register.py)")
        sys.exit(1)

    print("=" * 64)
    print("后处理流水线：点云清洗 → 泊松重建 → Mesh 平滑")
    print("=" * 64)
    print(f"输入: {input_path}")
    print()

    # ---- 加载点云 ----
    print("[加载] 读取点云...")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"  {len(pcd.points):,} 点")
    if len(pcd.points) < 1000:
        print("✗ 点数太少，检查输入文件")
        sys.exit(1)

    has_colors = len(np.asarray(pcd.colors)) > 0
    print(f"  颜色: {'有' if has_colors else '无'}")
    aabb = pcd.get_axis_aligned_bounding_box()
    sz = aabb.max_bound - aabb.min_bound
    print(f"  包围盒: {sz[0]:.1f} × {sz[1]:.1f} × {sz[2]:.1f} mm")
    print()

    # ---- Step 1: SOR ----
    print("[Step 1/7] 统计离群点滤波（SOR）...")
    pcd = step_sor(pcd, nb_neighbors=args.sor_k, std_ratio=args.sor_std)
    print()

    # ---- Step 2: ROR ----
    print("[Step 2/7] 半径离群点滤波（ROR）...")
    pcd = step_ror(pcd, nb_points=args.ror_min, radius=args.ror_radius)
    print()

    # ---- Step 3: 体素降采样 ----
    print("[Step 3/7] 体素降采样...")
    pcd = step_voxel(pcd, voxel_size=args.voxel)
    print()

    # ---- Step 4: RANSAC 去残留平面（可选）----
    if args.remove_plane:
        print("[Step 4/7] RANSAC 去残留平面...")
        pcd = step_remove_plane(pcd)
    else:
        print("[Step 4/7] 跳过 RANSAC（加 --remove-plane 开启）")
    print()

    # ---- Step 4b: 上采样（可选）----
    if args.upsample:
        print("[Step 4b] 上采样（增加点云密度）...")
        pcd = step_upsample(pcd, voxel_size=args.upsample_voxel)
        print()

    # ---- Step 4c: 颜色重采样（可选）----
    if args.recolor:
        print("[Step 4c] 从原始照片重新采样颜色...")
        import json as _json
        poses_path = os.path.join(out_dir, "poses.json")
        calib_path = os.path.join(args.input, "..", "calib_imgs", "calibration.json")
        if not os.path.isfile(calib_path):
            calib_path = os.path.join("calib_imgs", "calibration.json")
        if os.path.isfile(poses_path) and os.path.isfile(calib_path):
            with open(poses_path) as f:
                poses = _json.load(f)
            with open(calib_path) as f:
                calib = _json.load(f)
            pcd = step_recolor_from_images(pcd, args.input, poses, calib)
        else:
            print("  警告：找不到 poses.json 或 calibration.json，跳过颜色重采样")
        print()

    # 保存清洗后的点云（方便调试）
    clean_pcd_path = os.path.join(out_dir, "pcd_postprocessed.ply")
    o3d.io.write_point_cloud(clean_pcd_path, pcd)
    print(f"  中间结果（清洗后点云）: {clean_pcd_path}")
    print(f"  最终点数: {len(pcd.points):,}")
    print()

    # ---- Step 5: 泊松重建 ----
    print("[Step 5/7] 泊松表面重建...")
    mesh = step_poisson(pcd,
                        depth=args.poisson_depth,
                        density_cut=args.density_cut,
                        normal_radius=args.normal_radius)
    print()

    # ---- Step 6: Taubin 平滑 ----
    if not args.no_taubin:
        print(f"[Step 6/7] Taubin 平滑（{args.taubin_iter} 次迭代）...")
        mesh = step_taubin(mesh, n_iter=args.taubin_iter)
    else:
        print("[Step 6/7] 跳过 Taubin 平滑（--no-taubin）")
        mesh.compute_vertex_normals()
    print()

    # ---- Step 7: 删除孤立小面片 ----
    print("[Step 7/7] 删除孤立小面片...")
    mesh = step_remove_small_clusters(mesh, min_triangle_ratio=args.min_cluster_ratio)
    print()

    # ---- 输出 ----
    print("输出...")
    base = os.path.join(out_dir, "plant_model_clean")
    obj_path  = base + ".obj"
    stl_path  = base + ".stl"
    ply_path  = base + ".ply"

    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(stl_path, mesh)
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)

    # ---- 统计 ----
    print()
    print("=" * 64)
    print("完成！")
    aabb2 = mesh.get_axis_aligned_bounding_box()
    sz2 = aabb2.max_bound - aabb2.min_bound
    print(f"  顶点: {len(mesh.vertices):,}")
    print(f"  面片: {len(mesh.triangles):,}")
    print(f"  水密: {'是' if mesh.is_watertight() else '否'}")
    print(f"  尺寸: {sz2[0]:.1f} × {sz2[1]:.1f} × {sz2[2]:.1f} mm")
    print()
    print(f"  → {obj_path}")
    print(f"  → {stl_path}")
    print(f"  → {ply_path}")
    print()
    print("调参建议:")
    print("  模型有很多漂浮点  → --sor-std 1.5（更激进）")
    print("  边缘仍有飞点      → --ror-radius 8 --ror-min 8")
    print("  表面还不够光滑    → --taubin-iter 30 或 50")
    print("  细节丢失太多      → --taubin-iter 10 --no-taubin 或减小 --voxel")
    print("  有转盘残留        → --remove-plane")


if __name__ == "__main__":
    main()
