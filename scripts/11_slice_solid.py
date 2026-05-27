"""
11_slice_solid.py — 切片堆叠法生成立体模型

原理：
  1. 沿 Y 轴将点云切成薄层（每层 1-2mm）
  2. 每层投影到 XZ 平面，拟合轮廓多边形
  3. 相邻层之间用三角面片连接，形成封闭立体

使用：
  python 11_slice_solid.py --input capture_xxx

  # 调参
  python 11_slice_solid.py --input capture_xxx --slice-height 1.5 --alpha 5
  python 11_slice_solid.py --input capture_xxx --source pcd_postprocessed.ply
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

from scipy.spatial import Delaunay


def slice_pointcloud(pts, colors, slice_height=1.0):
    """
    沿 Y 轴切片。
    返回每层的 XZ 点集和颜色均值。
    """
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    slices = []

    y = y_min
    while y < y_max:
        mask = (pts[:, 1] >= y) & (pts[:, 1] < y + slice_height)
        if mask.sum() < 3:
            y += slice_height
            continue

        slice_pts = pts[mask]
        slice_xz = slice_pts[:, [0, 2]]  # 投影到 XZ
        slice_y = y + slice_height / 2.0  # 层中心 Y

        # 颜色取该层均值
        slice_color = colors[mask].mean(axis=0) if colors is not None else None

        slices.append({
            'y': slice_y,
            'xz': slice_xz,
            'color': slice_color,
            'count': mask.sum(),
        })
        y += slice_height

    return slices


def fit_outline_concave(xz_pts, alpha=3.0):
    """
    用 alpha shape 拟合 XZ 平面上的轮廓。
    alpha 越小越贴合（细节多），越大越接近凸包。

    返回轮廓顶点（有序，逆时针）。
    """
    if len(xz_pts) < 4:
        return None

    try:
        tri = Delaunay(xz_pts)
    except Exception:
        return None

    # 提取边界边（只属于一个三角形的边）
    edges = {}
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edges[edge] = edges.get(edge, 0) + 1

    boundary_edges = {e for e, count in edges.items() if count == 1}

    if not boundary_edges:
        return None

    # 过滤 alpha：去掉过长的边（alpha shape 核心）
    if alpha > 0:
        filtered = set()
        for e in boundary_edges:
            p1, p2 = xz_pts[e[0]], xz_pts[e[1]]
            dist = np.linalg.norm(p1 - p2)
            if dist < alpha:
                filtered.add(e)
        if filtered:
            boundary_edges = filtered

    # 构建有序轮廓
    adj = {}
    for e in boundary_edges:
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])

    # 从任意边界点开始，沿边界走
    start = next(iter(adj))
    outline = [start]
    visited = {start}
    current = start

    while True:
        neighbors = adj.get(current, [])
        next_pt = None
        for n in neighbors:
            if n not in visited:
                next_pt = n
                break
        if next_pt is None:
            break
        outline.append(next_pt)
        visited.add(next_pt)
        current = next_pt

    if len(outline) < 3:
        return None

    return xz_pts[np.array(outline)]


def fit_outline_convex(xz_pts):
    """
    凸包拟合轮廓。
    """
    if len(xz_pts) < 3:
        return None
    try:
        hull = Delaunay(xz_pts)
        # 找边界点
        from scipy.spatial import ConvexHull
        ch = ConvexHull(xz_pts)
        return xz_pts[ch.vertices]
    except Exception:
        return None


def connect_layers(outline_below, outline_above):
    """
    将两层轮廓用三角面片连接。
    用最近邻配对，生成三角带。

    返回顶点列表和三角形索引列表。
    """
    n_below = len(outline_below)
    n_above = len(outline_above)

    if n_below == 0 or n_above == 0:
        return [], []

    # 合并顶点
    verts = np.vstack([outline_below, outline_above])

    tris = []
    i, j = 0, 0

    while i < n_below - 1 or j < n_above - 1:
        if i >= n_below - 1:
            j += 1
        elif j >= n_above - 1:
            i += 1
        else:
            # 选对角线更短的三角形
            d1 = np.linalg.norm(outline_below[i + 1] - outline_above[j])
            d2 = np.linalg.norm(outline_below[i] - outline_above[j + 1])
            if d1 < d2:
                i += 1
            else:
                j += 1

        # 生成两个三角形（一个四边形拆成两个三角形）
        b0 = i
        b1 = min(i + 1, n_below - 1)
        a0 = j
        a1 = min(j + 1, n_above - 1)

        tri1 = [b0, n_below + a0, b1]
        tri2 = [b1, n_below + a0, n_below + a1]

        # 去重（退化三角形）
        if len(set(tri1)) == 3:
            tris.append(tri1)
        if len(set(tri2)) == 3:
            tris.append(tri2)

    return verts, tris


def build_solid_mesh(slices, outlines, alpha=3.0):
    """
    从切片轮廓构建封闭立体 mesh。

    策略：
    1. 每层轮廓 → 三角化（填顶面和底面）
    2. 相邻层之间 → 三角带连接
    """
    all_verts = []
    all_tris = []
    all_colors = []

    layer_offsets = []  # 每层在 all_verts 中的起始索引

    current_offset = 0

    for i, (s, outline) in enumerate(zip(slices, outlines)):
        if outline is None:
            layer_offsets.append(None)
            continue

        n = len(outline)
        # 该层的 3D 顶点（XZ → XYZ）
        y = s['y']
        verts_3d = np.column_stack([
            outline[:, 0],
            np.full(n, y),
            outline[:, 1]
        ])

        layer_offsets.append(current_offset)
        all_verts.append(verts_3d)

        # 颜色
        if s['color'] is not None:
            all_colors.append(np.tile(s['color'], (n, 1)))

        # 三角化该层轮廓（扇形 triangulation，从中心点出发）
        center = np.array([outline[:, 0].mean(), y, outline[:, 1].mean()])
        all_verts.append(center.reshape(1, 3))
        center_idx = current_offset + n
        if s['color'] is not None:
            all_colors.append(s['color'].reshape(1, 3))

        for k in range(n):
            tri = [current_offset + k, current_offset + (k + 1) % n, center_idx]
            all_tris.append(tri)

        current_offset += n + 1  # +1 for center

    # 连接相邻层
    for i in range(len(slices) - 1):
        if layer_offsets[i] is None or layer_offsets[i + 1] is None:
            continue

        s_curr = slices[i]
        s_next = slices[i + 1]
        outline_curr = outlines[i]
        outline_next = outlines[i + 1]

        if outline_curr is None or outline_next is None:
            continue

        # 当前层和下一层的轮廓顶点
        n_curr = len(outline_curr)
        n_next = len(outline_next)

        # 合并顶点
        verts_curr = np.column_stack([
            outline_curr[:, 0],
            np.full(n_curr, s_curr['y']),
            outline_curr[:, 1]
        ])
        verts_next = np.column_stack([
            outline_next[:, 0],
            np.full(n_next, s_next['y']),
            outline_next[:, 1]
        ])

        # 简单连接：用最近邻配对
        offset_curr = layer_offsets[i]
        offset_next = layer_offsets[i + 1]

        # 贪心三角带
        ci, ni = 0, 0
        while ci < n_curr - 1 or ni < n_next - 1:
            c0 = offset_curr + ci
            c1 = offset_curr + min(ci + 1, n_curr - 1)
            n0 = offset_next + ni
            n1 = offset_next + min(ni + 1, n_next - 1)

            if ci < n_curr - 1 and ni < n_next - 1:
                # 选对角线更短的方向
                d1 = np.linalg.norm(
                    np.asarray(all_verts)[c1] if not isinstance(all_verts, list)
                    else all_verts[ci + 1 if ci + 1 < n_curr else ci] -
                         (all_verts[ni] if isinstance(all_verts, list) else np.asarray(all_verts)[n0]))
                # 简化：交替前进
                if ci <= ni:
                    ci += 1
                else:
                    ni += 1
            elif ci < n_curr - 1:
                ci += 1
            else:
                ni += 1

            c0 = offset_curr + min(ci, n_curr - 1)
            c1 = offset_curr + min(ci + 1, n_curr - 1) if ci + 1 < n_curr else c0
            n0 = offset_next + min(ni, n_next - 1)
            n1 = offset_next + min(ni + 1, n_next - 1) if ni + 1 < n_next else n0

            tri1 = [c0, n0, c1]
            tri2 = [c1, n0, n1]
            if len(set(tri1)) == 3:
                all_tris.append(tri1)
            if len(set(tri2)) == 3:
                all_tris.append(tri2)

    if not all_verts:
        return None

    verts = np.vstack(all_verts)
    tris = np.array(all_tris)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)

    if all_colors:
        colors = np.vstack(all_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    return mesh


def build_solid_mesh_simple(slices, outlines):
    """
    逐层堆叠，每层是封闭轮廓。
    相邻层之间用均匀配对的三角带连接。
    顶底加封盖（扇形三角化）。
    """
    # 收集有效层
    valid_layers = []
    for s, outline in zip(slices, outlines):
        if outline is not None and len(outline) >= 3:
            valid_layers.append((s, outline))

    if len(valid_layers) < 2:
        return None

    all_verts = []
    all_tris = []
    all_colors = []
    offset = 0

    layer_info = []  # (offset, n) for each layer

    for s, outline in valid_layers:
        y = s['y']
        n = len(outline)
        verts_3d = np.column_stack([outline[:, 0], np.full(n, y), outline[:, 1]])
        all_verts.append(verts_3d)
        layer_info.append((offset, n))

        if s['color'] is not None:
            all_colors.append(np.tile(s['color'], (n, 1)))

        offset += n

    # 三角带连接相邻层（均匀重采样到相同点数后配对）
    for i in range(len(layer_info) - 1):
        off_a, n_a = layer_info[i]
        off_b, n_b = layer_info[i + 1]
        outline_a = valid_layers[i][1]
        outline_b = valid_layers[i + 1][1]

        n_conn = max(n_a, n_b)
        a_idx = (np.linspace(0, n_a, n_conn, endpoint=False) % n_a).astype(int)
        b_idx = (np.linspace(0, n_b, n_conn, endpoint=False) % n_b).astype(int)

        for k in range(n_conn):
            k_next = (k + 1) % n_conn
            a0 = off_a + a_idx[k]
            a1 = off_a + a_idx[k_next]
            b0 = off_b + b_idx[k]
            b1 = off_b + b_idx[k_next]

            tri1 = [a0, b0, a1]
            tri2 = [a1, b0, b1]
            if len(set(tri1)) == 3:
                all_tris.append(tri1)
            if len(set(tri2)) == 3:
                all_tris.append(tri2)

    # 封底（第一层，扇形三角化）
    off_bot, n_bot = layer_info[0]
    s_bot = valid_layers[0][0]
    outline_bot = valid_layers[0][1]
    center_bot = np.array([outline_bot[:, 0].mean(), s_bot['y'], outline_bot[:, 1].mean()])
    all_verts.append(center_bot.reshape(1, 3))
    c_bot_idx = offset
    offset += 1
    if s_bot['color'] is not None:
        all_colors.append(s_bot['color'].reshape(1, 3))
    for k in range(n_bot):
        all_tris.append([off_bot + k, c_bot_idx, off_bot + (k + 1) % n_bot])

    # 封顶（最后一层，扇形三角化，法向朝上）
    off_top, n_top = layer_info[-1]
    s_top = valid_layers[-1][0]
    outline_top = valid_layers[-1][1]
    center_top = np.array([outline_top[:, 0].mean(), s_top['y'], outline_top[:, 1].mean()])
    all_verts.append(center_top.reshape(1, 3))
    c_top_idx = offset
    if s_top['color'] is not None:
        all_colors.append(s_top['color'].reshape(1, 3))
    for k in range(n_top):
        all_tris.append([off_top + (k + 1) % n_top, c_top_idx, off_top + k])

    verts = np.vstack(all_verts)
    tris = np.array(all_tris) if all_tris else np.zeros((0, 3), dtype=int)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if len(tris) > 0:
        mesh.triangles = o3d.utility.Vector3iVector(tris)

    if all_colors:
        colors = np.vstack(all_colors)
        if len(colors) == len(verts):
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    return mesh


def resample_outline(outline, n_target):
    """
    将轮廓均匀重采样到 n_target 个点。
    """
    n = len(outline)
    if n == n_target:
        return outline
    if n == 0:
        return outline

    # 计算周长
    indices = np.linspace(0, n, n_target, endpoint=False).astype(int)
    return outline[indices % n]


def main():
    parser = argparse.ArgumentParser(description="切片堆叠法生成立体模型")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--source", default="pcd_postprocessed.ply",
                        help="输入点云（默认 pcd_postprocessed.ply）")
    parser.add_argument("--slice-height", type=float, default=1.0,
                        help="切片厚度 mm（默认 1.0）")
    parser.add_argument("--alpha", type=float, default=0,
                        help="alpha shape 参数（0=凸包，>0=凹包，值越小越贴合）")
    parser.add_argument("--outline-points", type=int, default=64,
                        help="每层轮廓重采样点数（默认 64）")
    args = parser.parse_args()

    out_dir = os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, args.source)

    if not os.path.isfile(input_path):
        print(f"找不到输入: {input_path}")
        sys.exit(1)

    print("=" * 64)
    print("切片堆叠法生成立体模型")
    print("=" * 64)
    print(f"输入: {input_path}")
    print(f"切片厚度: {args.slice_height}mm")
    print(f"alpha: {args.alpha} ({'凸包' if args.alpha == 0 else '凹包'})")
    print()

    # 加载点云
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(np.asarray(pcd.colors)) > 0 else None
    print(f"点云: {len(pts):,} 点")
    print(f"Y 范围: {pts[:,1].min():.1f} ~ {pts[:,1].max():.1f} mm")
    print()

    t0 = time.time()

    # Step 1: 切片
    print("[Step 1/3] 切片...")
    slices = slice_pointcloud(pts, colors, slice_height=args.slice_height)
    print(f"  有效切片: {len(slices)} 层")

    # Step 2: 每层拟合轮廓
    print("[Step 2/3] 拟合轮廓...")
    outlines = []
    n_convex = 0
    n_concave = 0
    n_fallback = 0
    for s in slices:
        outline = None
        if args.alpha > 0:
            outline = fit_outline_concave(s['xz'], alpha=args.alpha)
            if outline is not None:
                n_concave += 1
            else:
                # alpha shape 失败，退回凸包
                outline = fit_outline_convex(s['xz'])
                if outline is not None:
                    n_fallback += 1
        else:
            outline = fit_outline_convex(s['xz'])
            if outline is not None:
                n_convex += 1

        if outline is not None:
            outline = resample_outline(outline, args.outline_points)
        outlines.append(outline)

    print(f"  凹包层: {n_concave}, 凸包层: {n_convex}, 凹包失败退回凸包: {n_fallback}")
    valid = sum(1 for o in outlines if o is not None)
    print(f"  有效轮廓: {valid}/{len(slices)}")

    # Step 3: 堆叠成 mesh
    print("[Step 3/3] 堆叠成 mesh...")
    mesh = build_solid_mesh_simple(slices, outlines)

    if mesh is None:
        print("生成失败")
        sys.exit(1)

    # 法向量修正
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(tris) > 0:
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        if np.sum(v0 * np.cross(v1, v2)) / 6.0 < 0:
            mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
            mesh.compute_vertex_normals()
            print("  法向量已翻转")

    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s")
    print()

    # 输出
    base = os.path.join(out_dir, "plant_model_solid")
    obj_path = base + ".obj"
    stl_path = base + ".stl"
    ply_path = base + ".ply"

    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(stl_path, mesh)
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)

    aabb = mesh.get_axis_aligned_bounding_box()
    sz = aabb.max_bound - aabb.min_bound
    print(f"完成!")
    print(f"  顶点: {len(mesh.vertices):,}")
    print(f"  面片: {len(mesh.triangles):,}")
    print(f"  水密: {'是' if mesh.is_watertight() else '否'}")
    print(f"  尺寸: {sz[0]:.1f} x {sz[1]:.1f} x {sz[2]:.1f} mm")
    print()
    print(f"  -> {obj_path}")
    print(f"  -> {stl_path}")
    print(f"  -> {ply_path}")
    print()
    print("调参建议:")
    print("  轮廓太粗糙        -> --slice-height 0.5（更薄的切片）")
    print("  轮廓太方/不贴合   -> --alpha 5（凹包，值越小越贴合）")
    print("  轮廓太圆/丢失细节 -> --alpha 0（凸包）")
    print("  轮廓锯齿明显      -> --outline-points 128（更多采样点）")


if __name__ == "__main__":
    main()
