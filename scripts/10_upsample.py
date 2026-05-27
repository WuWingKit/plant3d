"""
10_upsample.py — 点云上采样（线性插值版）

原理：
  对每个点，找其 K 个最近邻，在该点和每个近邻之间插入 N 个等间距新点。
  新点的颜色 = 两端点颜色的线性插值（渐变）。
  同时也支持"颜色取 K 近邻加权平均"作为备选颜色模式。

使用：
  python 10_upsample.py --input capture_xxx

  # 调参
  python 10_upsample.py --input capture_xxx --k 6 --n-insert 1 --voxel 1.5
  python 10_upsample.py --input capture_xxx --color-mode average
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
    print("✗ 需要 open3d")
    sys.exit(1)


def upsample_pcd(pcd, k=6, n_insert=1, color_mode="lerp", voxel_merge=1.5):
    """
    点云上采样。

    参数：
      k:           每个点找 K 个最近邻
      n_insert:    两点之间插入几个点（1=插1个，即中点；2=插2个，即1/3和2/3处）
      color_mode:  'lerp'    - 颜色沿两点之间线性插值（渐变）
                   'average' - 新点颜色 = 该点所有 K 近邻颜色的加权平均
      voxel_merge: 上采样后做体素合并去重，防止重叠（mm）

    返回：上采样后的点云
    """
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(np.asarray(pcd.colors)) > 0 else None
    n = len(pts)

    print(f"  原始点数: {n:,}")
    print(f"  参数: K={k}, 每对插 {n_insert} 个点, 颜色模式={color_mode}")

    # 建 KD 树
    tree = o3d.geometry.KDTreeFlann(pcd)

    new_pts = []
    new_cols = []

    for i in range(n):
        p_i = pts[i]
        c_i = cols[i] if cols is not None else np.array([0.5, 0.5, 0.5])

        # 找 K+1 近邻（第一个是自己）
        [_, idx, _] = tree.search_knn_vector_3d(p_i, k + 1)
        neighbors = [j for j in idx if j != i][:k]

        if color_mode == "average":
            # 整体颜色 = K 近邻颜色的均值（不依赖具体的邻居对）
            neighbor_colors = np.array([cols[j] for j in neighbors]) if cols is not None \
                              else np.full((len(neighbors), 3), 0.5)
            avg_color = neighbor_colors.mean(axis=0)

        for j in neighbors:
            p_j = pts[j]
            c_j = cols[j] if cols is not None else np.array([0.5, 0.5, 0.5])

            # 在 p_i 和 p_j 之间插入 n_insert 个等间距点
            for k_step in range(1, n_insert + 1):
                t = k_step / (n_insert + 1)  # t ∈ (0, 1)

                # 新点坐标：线性插值
                p_new = (1 - t) * p_i + t * p_j
                new_pts.append(p_new)

                # 新点颜色
                if color_mode == "lerp":
                    # 沿两点之间线性插值（颜色渐变）
                    c_new = (1 - t) * c_i + t * c_j
                elif color_mode == "average":
                    # K 近邻颜色加权平均（距离越近权重越大）
                    c_new = avg_color
                else:
                    c_new = c_i

                new_cols.append(np.clip(c_new, 0, 1))

        if (i + 1) % 5000 == 0:
            print(f"    处理 {i+1}/{n} 点...")

    if not new_pts:
        print("  ⚠ 没有生成新点")
        return pcd

    new_pts = np.array(new_pts)
    new_cols = np.array(new_cols)

    print(f"  生成新点: {len(new_pts):,}")

    # 合并：原始点 + 新点
    all_pts = np.vstack([pts, new_pts])
    if cols is not None:
        all_cols = np.vstack([cols, new_cols])
    else:
        all_cols = None

    pcd_up = o3d.geometry.PointCloud()
    pcd_up.points = o3d.utility.Vector3dVector(all_pts)
    if all_cols is not None:
        pcd_up.colors = o3d.utility.Vector3dVector(all_cols)

    print(f"  合并后: {len(pcd_up.points):,} 点")

    # 体素合并去重（避免重叠点影响法向量估计）
    if voxel_merge > 0:
        pcd_up = pcd_up.voxel_down_sample(voxel_size=voxel_merge)
        print(f"  体素去重后: {len(pcd_up.points):,} 点")

    return pcd_up


def main():
    parser = argparse.ArgumentParser(description="点云上采样（线性插值）")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--source", default="pcd_postprocessed.ply",
                        help="输入点云（默认 pcd_postprocessed.ply，09_postprocess 的输出）")
    parser.add_argument("--k", type=int, default=6,
                        help="近邻数 K（默认 6）")
    parser.add_argument("--n-insert", type=int, default=1,
                        help="每对点之间插入几个点（默认 1 = 中点）")
    parser.add_argument("--color-mode", default="lerp",
                        choices=["lerp", "average"],
                        help="颜色模式：lerp=线性插值渐变 / average=K近邻均值")
    parser.add_argument("--voxel-merge", type=float, default=1.5,
                        help="上采样后的体素合并去重 mm（默认 1.5，0=不合并）")
    parser.add_argument("--out-name", default="pcd_upsampled.ply",
                        help="输出点云文件名")
    args = parser.parse_args()

    out_dir = os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, args.source)

    if not os.path.isfile(input_path):
        # 备选：从 merged_clean.ply 开始
        alt = os.path.join(out_dir, "merged_clean.ply")
        if os.path.isfile(alt):
            print(f"⚠ 找不到 {args.source}，使用 merged_clean.ply")
            input_path = alt
        else:
            print(f"✗ 找不到输入: {input_path}")
            sys.exit(1)

    print("=" * 64)
    print("点云上采样（线性插值）")
    print("=" * 64)
    print(f"输入: {input_path}")
    print()

    # 加载
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"加载: {len(pcd.points):,} 点")
    if len(pcd.points) < 100:
        print("✗ 点太少")
        sys.exit(1)
    print()

    # 上采样
    t0 = time.time()
    print("[上采样]")
    pcd_up = upsample_pcd(
        pcd,
        k=args.k,
        n_insert=args.n_insert,
        color_mode=args.color_mode,
        voxel_merge=args.voxel_merge,
    )
    print(f"  耗时: {time.time()-t0:.1f}s")
    print()

    # 保存
    out_path = os.path.join(out_dir, args.out_name)
    o3d.io.write_point_cloud(out_path, pcd_up)
    print(f"✓ 已保存: {out_path}")
    print()
    print("下一步：")
    print(f"  用 MeshLab 打开 {out_path} 检查点云是否合理")
    print(f"  满意后跑泊松重建:")
    print(f"  python 09_postprocess.py --input {args.input} "
          f"--source {args.out_name} --no-taubin")
    print(f"  或者直接把上采样点云传给 stage5:")
    print(f"  python 08_stage5_mesh.py --input {args.input} "
          f"--source output_v2/{args.out_name}")


if __name__ == "__main__":
    main()
