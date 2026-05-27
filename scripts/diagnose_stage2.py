"""
diagnose_stage2.py — 诊断 Stage 2 失败原因

检查 stage1 生成的点云：
  - 平面拟合结果（是不是水平面？是哪个轴？法向多大？）
  - 平面"之上"有多少点
  - 这些点的分布

用法:
  python diagnose_stage2.py --input capture_xxx
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("✗ 需要 open3d")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=3, help="检查前 N 帧")
    args = parser.parse_args()

    pcds_dir = os.path.join(args.input, "pcds")
    files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if not files:
        print(f"✗ {pcds_dir} 没有 PLY")
        sys.exit(1)

    print(f"诊断前 {args.n} 帧...")
    print()

    for path in files[:args.n]:
        fid = os.path.basename(path)
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        n = len(pts)

        print(f"━━━ {fid} ━━━")
        print(f"  总点数: {n}")
        if n < 1000:
            print("  ✗ 点数太少")
            continue

        # 包围盒
        aabb = pcd.get_axis_aligned_bounding_box()
        print(f"  包围盒:")
        print(f"    X: [{aabb.min_bound[0]:7.1f}, {aabb.max_bound[0]:7.1f}]  range={aabb.max_bound[0]-aabb.min_bound[0]:.1f}")
        print(f"    Y: [{aabb.min_bound[1]:7.1f}, {aabb.max_bound[1]:7.1f}]  range={aabb.max_bound[1]-aabb.min_bound[1]:.1f}")
        print(f"    Z: [{aabb.min_bound[2]:7.1f}, {aabb.max_bound[2]:7.1f}]  range={aabb.max_bound[2]-aabb.min_bound[2]:.1f}")

        # 尝试找平面（用 stage2 同款参数）
        for dist_th in [5, 8, 15, 25]:
            try:
                plane, inliers = pcd.segment_plane(
                    distance_threshold=dist_th, ransac_n=3, num_iterations=1000)
            except Exception:
                continue
            a, b, c, _ = plane
            normal_axis = ['X', 'Y', 'Z'][int(np.argmax([abs(a), abs(b), abs(c)]))]
            max_n = max(abs(a), abs(b), abs(c))
            n_plane = len(inliers)

            # 检查平面"之上"的点
            d = plane[3]
            signed = a*pts[:, 0] + b*pts[:, 1] + c*pts[:, 2] + d
            n_above = np.count_nonzero(signed > dist_th)
            n_below = np.count_nonzero(signed < -dist_th)

            print(f"  dist_thresh={dist_th:2d}: 平面 法向≈{normal_axis}({max_n:.2f})  内点 {n_plane:6}  之上 {max(n_above,n_below):6}  "
                  f"{'✓水平' if max_n > 0.85 else '✗ 非水平'}")

        # DBSCAN 不同 eps 测试（用 dist_thresh=8 的平面）
        plane, inliers = pcd.segment_plane(distance_threshold=8, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane
        signed = a*pts[:, 0] + b*pts[:, 1] + c*pts[:, 2] + d
        # 取数量多的一侧
        n_above = np.count_nonzero(signed > 8)
        n_below = np.count_nonzero(signed < -8)
        side_mask = (signed > 8) if n_above > n_below else (signed < -8)
        n_side = np.count_nonzero(side_mask)
        print(f"  平面之上点数: {n_side}")

        if n_side < 100:
            print(f"  ✗ 之上点太少（< 100），DBSCAN 不可能成功")
            continue

        pcd_above = pcd.select_by_index(np.where(side_mask)[0])
        for eps, min_pts in [(15, 200), (25, 100), (40, 50), (60, 20)]:
            try:
                labels = np.array(pcd_above.cluster_dbscan(eps=eps, min_points=min_pts))
            except Exception as e:
                print(f"    DBSCAN eps={eps} min_pts={min_pts}: 异常 {e}")
                continue
            if labels.max() < 0:
                print(f"    DBSCAN eps={eps:3d} min_pts={min_pts:3d}: 无簇")
            else:
                _, counts = np.unique(labels[labels >= 0], return_counts=True)
                print(f"    DBSCAN eps={eps:3d} min_pts={min_pts:3d}: {labels.max()+1} 个簇, 最大簇 {counts.max()} 点")

        print()

    print("━" * 50)
    print()
    print("解读:")
    print("  - 平面行: 应有至少一组显示 '✓ 水平'（法向某轴绝对值 > 0.85）")
    print("  - 之上点数 < 100 → 平面找错了，或者物体不在平面上方")
    print("  - DBSCAN 全部'无簇' → 点太分散，需要更大的 eps")
    print()
    print("建议参数:")
    print("  如果 dist_thresh=15 才水平 → 用 --dist-thresh 15")
    print("  如果只有 eps=40 才聚成簇 → 用 --eps 40 --min-points 50")


if __name__ == "__main__":
    main()
