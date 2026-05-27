"""
09_postprocess.py — Stage 3 点云后处理 → 高质量 Mesh

流水线：
  1. SOR 统计滤波     去漂浮孤立噪点
  2. ROR 半径滤波     去边缘飞点
  3. 体素降采样       合并配准误差导致的双层点
  4. RANSAC 去残留平面（可选，如果 Stage 2 没去干净）
  5. 上采样（可选）   插值新点，颜色从附近 K 点均值获取
  6. 泊松重建 depth=9
  7. 填补孔洞         fill_holes 修补 mesh 缺失区域
  8. Taubin 平滑      mesh 表面光滑，不缩水不失细节
  9. 删除孤立小面片

使用：
  python 09_postprocess.py --input capture_xxx --source pcd_upsampled.ply

  # 输出到 capture_xxx/output_v2/plant_model_clean.obj

调参建议：
  --poisson-depth 9            泊松深度（8/9/10，越小越不容易出孔洞）
  --density-cut 0.02           低密度顶点裁剪比例（越小保留越多）
  --normal-radius 12           法向量估计半径 mm（越大越稳定）
  --fill-hole-size 500         填洞的最大边界边长
  --taubin-iter 30             Taubin 迭代次数（越多越光滑）
  --no-fill                    跳过填洞
  --no-taubin                  跳过 Taubin 平滑

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
    print("需要 open3d: pip install open3d")
    sys.exit(1)


# ============================================================
#  点云后处理（步骤 1-5）
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


def step_upsample(pcd, neighbor_k=6, color_k=8, jitter=0.1):
    """
    上采样：在每对近邻之间插值中点，增加点云密度。

    原理：
    1. 对每个点找 K 个最近邻
    2. 在每对相邻点的中点处生成新点
    3. 沿法向量方向加微小随机偏移（避免共面）
    4. 新点颜色 = 附近 color_k 个原有点的颜色均值
    5. 用去重避免 A-B 和 B-A 生成重复中点

    neighbor_k: 每个点的近邻数（默认 6，生成点数约 N×K/2）
    color_k: 颜色插值时取最近 K 个点的均值（默认 8）
    jitter: 法向量随机偏移比例（0.1 = 距离的 10%）
    """
    n_before = len(pcd.points)

    # 计算法向量
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(np.asarray(pcd.colors)) > 0 else None
    normals = np.asarray(pcd.normals) if len(np.asarray(pcd.normals)) > 0 else None

    # 建 KD 树
    tree = o3d.geometry.KDTreeFlann(pcd)

    # 记录已生成的边，避免重复（用 (min_idx, max_idx) 作为 key）
    seen_edges = set()
    new_points = []
    new_colors = []

    for i in range(n_before):
        # 找 neighbor_k+1 个近邻（包含自身）
        _, idx, dist = tree.search_knn_vector_3d(pcd.points[i], neighbor_k + 1)
        idx = np.asarray(idx)
        dist = np.asarray(dist)

        for j in range(1, len(idx)):  # 跳过自身 (idx[0] == i)
            ni = idx[j]
            edge = (min(i, ni), max(i, ni))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            # 中点
            mid = (pts[i] + pts[ni]) / 2.0

            # 沿法向量方向加微小偏移
            if normals is not None:
                avg_normal = (normals[i] + normals[ni]) / 2.0
                norm_len = np.linalg.norm(avg_normal)
                if norm_len > 1e-8:
                    avg_normal /= norm_len
                    d = np.sqrt(dist[j])  # 近邻距离
                    offset = np.random.uniform(-jitter * d, jitter * d)
                    mid = mid + offset * avg_normal

            new_points.append(mid)

            # 颜色：取最近 color_k 个原有点的均值
            if colors is not None:
                _, cidx, _ = tree.search_knn_vector_3d(mid, color_k)
                avg_color = colors[np.asarray(cidx)].mean(axis=0)
                new_colors.append(avg_color)

    # 合并
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
    print(f"  上采样 (neighbor_k={neighbor_k}, color_k={color_k}, jitter={jitter}): "
          f"{n_before:,} → {n_after:,} 点 (+{n_after-n_before:,})")
    return pcd_upsampled


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
#  泊松重建 + Mesh 修补（步骤 6-7）
# ============================================================

def step_poisson(pcd, depth=9, density_cut=0.02, normal_radius=12.0, normal_k=30):
    """
    泊松表面重建。

    depth=9: 网格粒度适中，稀疏区域不容易出孔洞。
    density_cut=0.02: 只裁最底部 2% 的低密度顶点（花盆区域点少，不能裁太多）。
    normal_radius=12: 法向量估计半径更大，结果更稳定。
    """
    print(f"  估计法向量 (radius={normal_radius}mm, k={normal_k})...")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_k))
    pcd.orient_normals_consistent_tangent_plane(k=normal_k)

    print(f"  泊松重建 (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    densities = np.asarray(densities)

    # 裁掉最低密度顶点（边角碎片），保留大部分
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


def step_fill_holes(mesh, max_hole_size=500):
    """
    填补 mesh 上的孔洞（使用 pymeshlab）。

    max_hole_size: 孔洞边界的最大边数。
                   500 能覆盖大部分中等孔洞。
                   太大的孔洞（如花盆底部大面积缺失）填出来会是平的补丁。
    """
    import pymeshlab

    n_tri_before = len(mesh.triangles)

    # Open3D mesh -> pymeshlab mesh
    ms = pymeshlab.MeshSet()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    m = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
    ms.add_mesh(m)

    # 先修复非流形（close_holes 要求边是流形的）
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    # 填洞
    ms.meshing_close_holes(maxholesize=max_hole_size)

    # pymeshlab mesh -> Open3D mesh
    result_mesh = ms.current_mesh()
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(result_mesh.vertex_matrix())
    out.triangles = o3d.utility.Vector3iVector(result_mesh.face_matrix())

    # 保留顶点颜色（如果原 mesh 有的话）
    if len(np.asarray(mesh.vertex_colors)) > 0:
        # 用最近邻映射颜色：新顶点取原 mesh 最近顶点的颜色
        old_colors = np.asarray(mesh.vertex_colors)
        old_verts = np.asarray(mesh.vertices)
        tree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(old_verts)))
        new_verts = np.asarray(out.vertices)
        new_colors = np.zeros((len(new_verts), 3))
        for i in range(len(new_verts)):
            _, idx, _ = tree.search_knn_vector_3d(out.vertices[i], 1)
            new_colors[i] = old_colors[idx[0]]
        out.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    out.compute_vertex_normals()

    n_tri_after = len(out.triangles)
    filled = n_tri_after - n_tri_before
    print(f"  填补孔洞 (max_hole_size={max_hole_size}): "
          f"{n_tri_before:,} → {n_tri_after:,} 面片 (+{filled:,} 补丁)")
    return out


# ============================================================
#  Mesh 后处理（步骤 8-9）
# ============================================================

def step_smooth(mesh, n_iter=50, method="pymeshlab_taubin"):
    """
    Mesh 平滑。支持两种后端：

    method:
      'pymeshlab_taubin'  - pymeshlab Taubin（默认，效果更好）
      'pymeshlab_laplacian' - pymeshlab Laplacian（更强但会缩水）
      'open3d_taubin'     - Open3D Taubin（兜底）

    n_iter: 迭代次数。
    """
    n_vert_before = len(mesh.vertices)

    if method.startswith("pymeshlab"):
        import pymeshlab

        # Open3D -> pymeshlab
        ms = pymeshlab.MeshSet()
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        m = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
        ms.add_mesh(m)

        if method == "pymeshlab_taubin":
            ms.apply_coord_taubin_smoothing(stepsmoothnum=n_iter)
            print(f"  pymeshlab Taubin ({n_iter} 次): ", end="")
        elif method == "pymeshlab_laplacian":
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=n_iter)
            print(f"  pymeshlab Laplacian ({n_iter} 次): ", end="")

        # pymeshlab -> Open3D
        result = ms.current_mesh()
        out = o3d.geometry.TriangleMesh()
        out.vertices = o3d.utility.Vector3dVector(result.vertex_matrix())
        out.triangles = o3d.utility.Vector3iVector(result.face_matrix())

        # 保留顶点颜色
        if len(np.asarray(mesh.vertex_colors)) > 0:
            old_colors = np.asarray(mesh.vertex_colors)
            old_verts = np.asarray(mesh.vertices)
            tree = o3d.geometry.KDTreeFlann(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(old_verts)))
            new_verts = np.asarray(out.vertices)
            new_colors = np.zeros((len(new_verts), 3))
            for i in range(len(new_verts)):
                _, idx, _ = tree.search_knn_vector_3d(out.vertices[i], 1)
                new_colors[i] = old_colors[idx[0]]
            out.vertex_colors = o3d.utility.Vector3dVector(new_colors)

        out.compute_vertex_normals()
        print(f"{n_vert_before:,} 顶点 -> {len(out.vertices):,} 顶点")
        return out

    else:
        # Open3D Taubin 兜底
        out = mesh.filter_smooth_taubin(number_of_iterations=n_iter,
                                        lambda_filter=0.5, mu=-0.53)
        out.compute_vertex_normals()
        print(f"  Open3D Taubin ({n_iter} 次): "
              f"{n_vert_before:,} 顶点 -> {len(out.vertices):,} 顶点")
        return out


def step_fix_normals(mesh):
    """
    修正法向量方向，确保朝外。

    用符号体积检测：如果体积为负，说明大部分面片朝内，需要翻转。
    翻转后重新计算法向量。
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if len(tris) == 0 or len(verts) == 0:
        return mesh

    # 计算符号体积（正 = 法向朝外，负 = 法向朝内）
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    signed_vol = np.sum(v0 * np.cross(v1, v2)) / 6.0

    if signed_vol < 0:
        # 法向朝内，翻转所有三角形
        mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
        print(f"  法向量修正: 检测到朝内 (vol={signed_vol:.1f}), 已翻转")
    else:
        print(f"  法向量检查: 朝向正确 (vol={signed_vol:.1f})")

    mesh.compute_vertex_normals()
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

    # 上采样参数
    parser.add_argument("--upsample", action="store_true",
                        help="启用上采样（在近邻间插值中点）")
    parser.add_argument("--neighbor-k", type=int, default=6,
                        help="上采样近邻数（默认 6，生成点数约 N*K/2）")
    parser.add_argument("--color-k", type=int, default=8,
                        help="上采样颜色插值近邻数（默认 8）")
    parser.add_argument("--jitter", type=float, default=0.1,
                        help="上采样法向量随机偏移比例（默认 0.1）")

    # 泊松参数
    parser.add_argument("--poisson-depth", type=int, default=9,
                        help="泊松深度（默认 9，越小越不容易出孔洞）")
    parser.add_argument("--density-cut", type=float, default=0.02,
                        help="低密度顶点裁剪比例（默认 0.02，只裁最底部 2%%）")
    parser.add_argument("--normal-radius", type=float, default=12.0,
                        help="法向量估计半径 mm（默认 12，越大越稳定）")

    # 填洞参数
    parser.add_argument("--fill-hole-size", type=int, default=500,
                        help="填补孔洞的最大边界边长（默认 500）")
    parser.add_argument("--no-fill", action="store_true",
                        help="跳过孔洞填补")

    # Mesh 平滑参数
    parser.add_argument("--smooth-method", default="pymeshlab_taubin",
                        choices=["pymeshlab_taubin", "pymeshlab_laplacian", "open3d_taubin"],
                        help="平滑方法（默认 pymeshlab_taubin）")
    parser.add_argument("--taubin-iter", type=int, default=50,
                        help="平滑迭代次数（默认 50）")
    parser.add_argument("--no-taubin", action="store_true",
                        help="跳过平滑")
    parser.add_argument("--min-cluster-ratio", type=float, default=0.01,
                        help="孤立面片删除阈值（默认 0.01 = 1%%）")

    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, args.source)

    if not os.path.isfile(input_path):
        print(f"找不到输入文件: {input_path}")
        print(f"  确认已经跑过 Stage 3 (06_stage3_register.py)")
        sys.exit(1)

    print("=" * 64)
    print("后处理流水线：点云清洗 → 泊松重建 → 填洞 → Mesh 平滑")
    print("=" * 64)
    print(f"输入: {input_path}")
    print()

    # ---- 加载点云 ----
    print("[加载] 读取点云...")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"  {len(pcd.points):,} 点")
    if len(pcd.points) < 1000:
        print("点数太少，检查输入文件")
        sys.exit(1)

    has_colors = len(np.asarray(pcd.colors)) > 0
    print(f"  颜色: {'有' if has_colors else '无'}")
    aabb = pcd.get_axis_aligned_bounding_box()
    sz = aabb.max_bound - aabb.min_bound
    print(f"  包围盒: {sz[0]:.1f} x {sz[1]:.1f} x {sz[2]:.1f} mm")
    print()

    # ---- Step 1: SOR ----
    print("[Step 1/9] 统计离群点滤波（SOR）...")
    pcd = step_sor(pcd, nb_neighbors=args.sor_k, std_ratio=args.sor_std)
    print()

    # ---- Step 2: ROR ----
    print("[Step 2/9] 半径离群点滤波（ROR）...")
    pcd = step_ror(pcd, nb_points=args.ror_min, radius=args.ror_radius)
    print()

    # ---- Step 3: 体素降采样 ----
    print("[Step 3/9] 体素降采样...")
    pcd = step_voxel(pcd, voxel_size=args.voxel)
    print()

    # ---- Step 4: RANSAC 去残留平面（可选）----
    if args.remove_plane:
        print("[Step 4/9] RANSAC 去残留平面...")
        pcd = step_remove_plane(pcd)
    else:
        print("[Step 4/9] 跳过 RANSAC（加 --remove-plane 开启）")
    print()

    # ---- Step 5: 上采样（可选）----
    if args.upsample:
        print("[Step 5/9] 上采样（增加点云密度）...")
        pcd = step_upsample(pcd, neighbor_k=args.neighbor_k,
                            color_k=args.color_k, jitter=args.jitter)
        print()

    # 保存清洗后的点云（方便调试）
    clean_pcd_path = os.path.join(out_dir, "pcd_postprocessed.ply")
    o3d.io.write_point_cloud(clean_pcd_path, pcd)
    print(f"  中间结果（清洗后点云）: {clean_pcd_path}")
    print(f"  最终点数: {len(pcd.points):,}")
    print()

    # ---- Step 6: 泊松重建 ----
    print("[Step 6/9] 泊松表面重建...")
    mesh = step_poisson(pcd,
                        depth=args.poisson_depth,
                        density_cut=args.density_cut,
                        normal_radius=args.normal_radius)
    print()

    # ---- Step 7: 填补孔洞 ----
    if not args.no_fill:
        print(f"[Step 7/9] 填补孔洞 (hole_size={args.fill_hole_size})...")
        mesh = step_fill_holes(mesh, max_hole_size=args.fill_hole_size)
    else:
        print("[Step 7/9] 跳过填洞（--no-fill）")
    print()

    # ---- Step 8: 平滑 ----
    if not args.no_taubin:
        print(f"[Step 8/9] 平滑（{args.smooth_method}, {args.taubin_iter} 次）...")
        mesh = step_smooth(mesh, n_iter=args.taubin_iter,
                           method=args.smooth_method)
    else:
        print("[Step 8/9] 跳过平滑（--no-taubin）")
        mesh.compute_vertex_normals()
    print()

    # ---- Step 9: 删除孤立小面片 ----
    print("[Step 9/9] 删除孤立小面片...")
    mesh = step_remove_small_clusters(mesh, min_triangle_ratio=args.min_cluster_ratio)
    print()

    # ---- 法向量修正 ----
    print("[修正] 法向量方向检查...")
    mesh = step_fix_normals(mesh)
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
    print(f"  尺寸: {sz2[0]:.1f} x {sz2[1]:.1f} x {sz2[2]:.1f} mm")
    print()
    print(f"  -> {obj_path}")
    print(f"  -> {stl_path}")
    print(f"  -> {ply_path}")
    print()
    print("调参建议:")
    print("  孔洞很多          -> --density-cut 0.01 --poisson-depth 8")
    print("  花盆区域有洞      -> --density-cut 0.01（保留更多低密度顶点）")
    print("  填洞后不自然      -> --taubin-iter 50（更多平滑）")
    print("  模型有很多漂浮点  -> --sor-std 1.5（更激进）")
    print("  边缘仍有飞点      -> --ror-radius 8 --ror-min 8")
    print("  表面还不够光滑    -> --taubin-iter 50")
    print("  细节丢失太多      -> --taubin-iter 10 或减小 --voxel")
    print("  有转盘残留        -> --remove-plane")


if __name__ == "__main__":
    main()
