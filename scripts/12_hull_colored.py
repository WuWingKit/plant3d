"""
12_hull_colored.py — 凸包/凹包包裹点云 + 附近点云颜色上色

原理：
  1. 对点云做凸包（或 alpha shape），生成包裹面片
  2. 每个面片中心找附近 K 个点，取颜色均值上色
  3. 输出带颜色的立体 mesh

使用：
  python 12_hull_colored.py --input capture_xxx

  # 凹包（更贴合细节）
  python 12_hull_colored.py --input capture_xxx --alpha 5

  # 调颜色采样距离
  python 12_hull_colored.py --input capture_xxx --color-radius 15
"""

import os
import sys
import io
import argparse
import numpy as np
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d")
    sys.exit(1)

from scipy.spatial import ConvexHull, Delaunay


def build_convex_hull(pts):
    """凸包包裹。返回 (vertices, faces)。"""
    hull = ConvexHull(pts)
    return pts[hull.vertices], hull.simplices


def build_alpha_hull(pts, alpha=5.0):
    """
    Alpha shape 包裹。
    去掉外接球半径 > alpha 的四面体，保留边界面。
    """
    tri = Delaunay(pts)
    simplices = tri.simplices

    # 找每个四面体的外接球半径
    keep = []
    for s in simplices:
        # 四面体 4 个顶点
        tet_pts = pts[s]
        # 外接球半径公式
        # R = |a-d| / (6*V) * |b-d| * |c-d|  (简化)
        # 更简单：用最长边的一半作为近似
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                edges.append(np.linalg.norm(tet_pts[i] - tet_pts[j]))
        max_edge = max(edges)
        # 如果最长边 > 2*alpha，认为是外部四面体
        if max_edge <= 2 * alpha:
            keep.append(s)

    if not keep:
        return None, None

    keep = np.array(keep)

    # 提取边界面（只属于一个四面体的面）
    face_count = {}
    for s in keep:
        faces_in_tet = [
            tuple(sorted([s[0], s[1], s[2]])),
            tuple(sorted([s[0], s[1], s[3]])),
            tuple(sorted([s[0], s[2], s[3]])),
            tuple(sorted([s[1], s[2], s[3]])),
        ]
        for f in faces_in_tet:
            face_count[f] = face_count.get(f, 0) + 1

    boundary_faces = [list(f) for f, c in face_count.items() if c == 1]

    if not boundary_faces:
        return None, None

    return pts, np.array(boundary_faces)


def color_faces(pts, colors, face_verts, face_indices, color_radius=10.0):
    """
    给每个面上色。
    每个面的颜色 = 面中心附近 color_radius 范围内点的颜色均值。
    """
    if colors is None or len(colors) == 0:
        return None

    # 建 KD 树
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    tree = o3d.geometry.KDTreeFlann(pcd)

    face_colors = []
    for fi in face_indices:
        # 面中心
        center = face_verts[fi].mean(axis=0)
        # 搜半径内的点
        [_, idx, dist2] = tree.search_radius_vector_3d(center, color_radius)
        if len(idx) > 0:
            nearby_colors = colors[idx]
            face_colors.append(nearby_colors.mean(axis=0))
        else:
            face_colors.append(np.array([0.5, 0.5, 0.5]))

    return np.array(face_colors)


def color_faces_vectorized(pts, colors, face_verts, face_indices, color_radius=10.0):
    """
    向量化版本：用 numpy 广播加速。
    对每个面中心，找最近邻点的颜色。
    """
    if colors is None or len(colors) == 0:
        return None

    # 面中心
    centers = np.array([face_verts[fi].mean(axis=0) for fi in face_indices])

    # 用 KD 树搜半径
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    tree = o3d.geometry.KDTreeFlann(pcd)

    face_colors = np.zeros((len(face_indices), 3))
    for i, center in enumerate(centers):
        [_, idx, _] = tree.search_radius_vector_3d(center, color_radius)
        if len(idx) > 0:
            face_colors[i] = colors[idx].mean(axis=0)
        else:
            face_colors[i] = np.array([0.5, 0.5, 0.5])

        if (i + 1) % 500 == 0:
            print(f"    上色 {i+1}/{len(face_indices)} 面...")

    return face_colors


def remove_outlier_percentile(pts, colors, remove_pct=0.05):
    """
    去掉离质心最远的 remove_pct 比例的点。
    返回清理后的 pts 和 colors。
    """
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    threshold = np.percentile(dists, 100 * (1 - remove_pct))
    mask = dists <= threshold
    print(f"  去掉离群点: {len(pts)} -> {mask.sum()} ({remove_pct*100:.0f}% 去除)")
    pts_clean = pts[mask]
    colors_clean = colors[mask] if colors is not None else None
    return pts_clean, colors_clean, mask


def main():
    parser = argparse.ArgumentParser(description="凸包/凹包包裹点云 + 颜色上色")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--source", default="pcd_postprocessed.ply",
                        help="输入点云（默认 pcd_postprocessed.ply）")
    parser.add_argument("--alpha", type=float, default=0,
                        help="alpha shape 参数（0=凸包，>0=凹包，值越小越贴合）")
    parser.add_argument("--color-radius", type=float, default=10.0,
                        help="上色搜索半径 mm（默认 10）")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Taubin 平滑次数（默认 0）")
    parser.add_argument("--outlier-pct", type=float, default=0.05,
                        help="去掉离质心最远的百分比（默认 0.05 = 5%%）")
    parser.add_argument("--subdivide", type=int, default=0,
                        help="Loop 细分次数（增加面数，默认 0）")
    args = parser.parse_args()

    out_dir = os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, args.source)

    if not os.path.isfile(input_path):
        print(f"找不到输入: {input_path}")
        sys.exit(1)

    print("=" * 64)
    print("凸包/凹包包裹 + 颜色上色")
    print("=" * 64)
    print(f"输入: {input_path}")
    print(f"模式: {'凹包 alpha=' + str(args.alpha) if args.alpha > 0 else '凸包'}")
    print(f"上色半径: {args.color_radius}mm")
    print(f"离群点去除: {args.outlier_pct*100:.0f}%")
    print()

    # 加载点云
    pcd_orig = o3d.io.read_point_cloud(input_path)
    pts_orig = np.asarray(pcd_orig.points)
    colors_orig = np.asarray(pcd_orig.colors) if len(np.asarray(pcd_orig.colors)) > 0 else None
    print(f"原始点云: {len(pts_orig):,} 点")
    print(f"尺寸: {pts_orig.max(axis=0) - pts_orig.min(axis=0)}")
    print()

    # 去除离群点
    if args.outlier_pct > 0:
        pts, colors, _ = remove_outlier_percentile(pts_orig, colors_orig, args.outlier_pct)
    else:
        pts, colors = pts_orig, colors_orig

    t0 = time.time()

    # Step 1: 生成包裹 mesh
    print("[Step 1/4] 生成包裹 mesh...")
    if args.alpha > 0:
        print(f"  alpha shape (alpha={args.alpha})...")
        hull_verts, faces = build_alpha_hull(pts, alpha=args.alpha)
        if hull_verts is None:
            print("  alpha shape 失败，退回凸包")
            hull = ConvexHull(pts)
            hull_verts = pts[hull.vertices]
            idx_map = {v: i for i, v in enumerate(hull.vertices)}
            faces = np.array([[idx_map[v] for v in f] for f in hull.simplices])
        else:
            used = np.unique(faces)
            idx_map = {v: i for i, v in enumerate(used)}
            hull_verts = pts[used]
            faces = np.array([[idx_map[v] for v in f] for f in faces])
    else:
        print("  凸包...")
        hull = ConvexHull(pts)
        hull_verts = pts[hull.vertices]
        idx_map = {v: i for i, v in enumerate(hull.vertices)}
        faces = np.array([[idx_map[v] for v in f] for f in hull.simplices])

    print(f"  顶点: {len(hull_verts)}, 面片: {len(faces)}")

    # Step 2: 构建 mesh
    print("[Step 2/4] 构建 mesh...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(hull_verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # 法向量方向修正
    verts_arr = np.asarray(mesh.vertices)
    tris_arr = np.asarray(mesh.triangles)
    v0 = verts_arr[tris_arr[:, 0]]
    v1 = verts_arr[tris_arr[:, 1]]
    v2 = verts_arr[tris_arr[:, 2]]
    vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
    if vol < 0:
        mesh.triangles = o3d.utility.Vector3iVector(tris_arr[:, [0, 2, 1]])
        mesh.compute_vertex_normals()
        print("  法向量已翻转")

    # 细分
    if args.subdivide > 0:
        for _ in range(args.subdivide):
            mesh = mesh.subdivide_loop(number_of_iterations=1)
        mesh.compute_vertex_normals()
        print(f"  细分后: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # 平滑
    if args.smooth > 0:
        print(f"  Taubin 平滑 {args.smooth} 次...")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth)
        mesh.compute_vertex_normals()

    # 修复浮点精度导致的"不水密"：所有顶点向质心微偏移
    if args.subdivide > 0:
        verts_arr = np.asarray(mesh.vertices)
        center = verts_arr.mean(axis=0)
        # 闭合曲面：每个顶点向质心方向微偏移，消除浮点缝隙
        for i in range(len(verts_arr)):
            d = center - verts_arr[i]
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                verts_arr[i] += (d / norm) * 0.005
        mesh.vertices = o3d.utility.Vector3dVector(verts_arr)
        mesh.compute_vertex_normals()
        print(f"  浮点精度修复完成")

    # Step 3: 上色（用原始点云颜色，包含所有点）
    print("[Step 3/4] 顶点上色...")
    mesh_verts = np.asarray(mesh.vertices)
    tree = o3d.geometry.KDTreeFlann(pcd_orig)
    vert_colors = np.zeros((len(mesh_verts), 3))
    for i, v in enumerate(mesh_verts):
        [_, idx, _] = tree.search_radius_vector_3d(v, args.color_radius)
        if len(idx) > 0:
            vert_colors[i] = colors_orig[idx].mean(axis=0)
        else:
            [_, idx2, _] = tree.search_knn_vector_3d(v, 1)
            vert_colors[i] = colors_orig[idx2[0]]
        if (i + 1) % 500 == 0:
            print(f"    上色 {i+1}/{len(mesh_verts)} 顶点...")
    vert_colors = np.clip(vert_colors, 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    print(f"  上色完成")

    elapsed = time.time() - t0
    print(f"  总耗时: {elapsed:.1f}s")
    print()

    # Step 4: 输出
    base = os.path.join(out_dir, "plant_model_hull")
    obj_path = base + ".obj"
    stl_path = base + ".stl"
    ply_path = base + ".ply"

    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(stl_path, mesh)
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)

    # 统计有多少点在 mesh 外面
    aabb = mesh.get_axis_aligned_bounding_box()
    inside_mask = (
        (pts_orig[:, 0] >= aabb.min_bound[0]) & (pts_orig[:, 0] <= aabb.max_bound[0]) &
        (pts_orig[:, 1] >= aabb.min_bound[1]) & (pts_orig[:, 1] <= aabb.max_bound[1]) &
        (pts_orig[:, 2] >= aabb.min_bound[2]) & (pts_orig[:, 2] <= aabb.max_bound[2])
    )
    outside_pct = 100 * (1 - inside_mask.sum() / len(pts_orig))

    sz = aabb.max_bound - aabb.min_bound
    print(f"完成!")
    print(f"  顶点: {len(mesh.vertices):,}")
    print(f"  面片: {len(mesh.triangles):,}")
    print(f"  水密: {'是' if mesh.is_watertight() else '否'}")
    print(f"  尺寸: {sz[0]:.1f} x {sz[1]:.1f} x {sz[2]:.1f} mm")
    print(f"  AABB 外点: {outside_pct:.1f}%")
    print()
    print(f"  -> {obj_path}")
    print(f"  -> {stl_path}")
    print(f"  -> {ply_path}")


if __name__ == "__main__":
    main()
