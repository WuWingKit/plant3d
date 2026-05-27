"""
08_stage5_mesh.py — 表面重建（Open3D 默认泊松）

输入：capture_xxx/output_v2/merged_aligned.ply
输出：capture_xxx/output_v2/plant_model.{obj,stl,ply}

使用：
  python 08_stage5_mesh.py --input capture_xxx
  python 08_stage5_mesh.py --input capture_xxx --depth 10
"""

import os
import sys
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Stage 5: 泊松表面重建")
    parser.add_argument("--input", required=True)
    parser.add_argument("--depth", type=int, default=9, help="泊松深度（默认 9）")
    parser.add_argument("--density-cut", type=float, default=0.10,
                        help="低密度顶点裁剪比例（默认 0.10）")
    parser.add_argument("--normal-radius", type=float, default=10.0,
                        help="法向量估计半径 mm（默认 10）")
    parser.add_argument("--source", default="merged_aligned.ply",
                        help="输入点云文件名")
    args = parser.parse_args()

    out_dir = os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, args.source)
    if not os.path.isfile(input_path):
        print(f"✗ 找不到 {input_path}")
        # 备选：merged_clean.ply
        alt = os.path.join(out_dir, "merged_clean.ply")
        if os.path.isfile(alt):
            print(f"  使用备选: {alt}")
            input_path = alt
        else:
            sys.exit(1)

    print("=" * 64)
    print("Stage 5: 泊松重建")
    print("=" * 64)
    print(f"输入: {input_path}")
    print(f"depth={args.depth}, density_cut={args.density_cut}")
    print()

    pcd = o3d.io.read_point_cloud(input_path)
    print(f"加载: {len(pcd.points):,} 点")
    if len(pcd.points) < 1000:
        print("✗ 点数太少")
        sys.exit(1)

    # 法向量
    print(f"\n[1/3] 估计法向量 (radius={args.normal_radius}mm)...")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=20)

    # 泊松
    print(f"\n[2/3] 泊松重建 (depth={args.depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth)
    densities = np.asarray(densities)
    print(f"  泊松产出: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面")

    # 裁剪低密度区域
    threshold = np.quantile(densities, args.density_cut)
    mesh.remove_vertices_by_mask(densities < threshold)
    print(f"  裁剪后:   {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面")

    # 顶点颜色（从点云最近邻）
    if len(np.asarray(pcd.colors)) > 0:
        print(f"\n[3/3] 顶点颜色映射...")
        mv = np.asarray(mesh.vertices)
        pc = np.asarray(pcd.colors)
        tree = o3d.geometry.KDTreeFlann(pcd)
        colors = np.zeros_like(mv)
        for i in range(len(mv)):
            _, idx, _ = tree.search_knn_vector_3d(mesh.vertices[i], 1)
            colors[i] = pc[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 计算法向量（STL 需要）
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # 保存
    base = os.path.join(out_dir, "plant_model")
    obj_path = base + ".obj"
    stl_path = base + ".stl"
    ply_path = base + ".ply"

    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(stl_path, mesh)
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)

    print()
    print(f"✓ {obj_path}")
    print(f"✓ {stl_path}")
    print(f"✓ {ply_path}")

    # 统计
    print()
    print("模型统计:")
    print(f"  顶点: {len(mesh.vertices):,}")
    print(f"  面片: {len(mesh.triangles):,}")
    print(f"  水密: {'是' if mesh.is_watertight() else '否'}")
    print(f"  自相交: {'有' if mesh.is_self_intersecting() else '无'}")

    aabb = mesh.get_axis_aligned_bounding_box()
    sz = aabb.max_bound - aabb.min_bound
    print(f"  尺寸 (X×Y×Z): {sz[0]:.1f} × {sz[1]:.1f} × {sz[2]:.1f} mm")

    print()
    print("完成！用 MeshLab/Blender 打开 .obj 或 .ply 查看带颜色的模型")


if __name__ == "__main__":
    main()
