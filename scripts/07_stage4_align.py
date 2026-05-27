"""
07_stage4_align.py — 摆正点云（盆底贴 XY 平面，盆边对齐 X 轴）

算法（你描述的方法）：
  1. 取点云最低 15% 区域（盆底）
  2. RANSAC 拟合盆底平面 → 法向量 N
  3. Rodrigues 公式：让 N → Z 轴（水平校正）
  4. 旋转后取盆底点投影到 XY 平面 → PCA → 最大主轴
  5. 绕 Z 轴旋转让主轴对齐 X 轴（方向校正）
  6. 最终变换 = R_z @ R_xyz，应用到点云

输入：
  capture_xxx/output_v2/merged_clean.ply

输出：
  capture_xxx/output_v2/merged_aligned.ply
  capture_xxx/output_v2/align_transform.json  (变换矩阵)

使用：
  python 07_stage4_align.py --input capture_xxx
  python 07_stage4_align.py --input capture_xxx --low-percent 15
  python 07_stage4_align.py --input capture_xxx --no-orient  # 跳过方向校正
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import cv2
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)


def find_base_plane(pcd, low_percent=15.0, dist_thresh=5.0):
    """
    找盆底平面：
      1. 取点云"底部" low_percent% 的点（按某一轴排序）
      2. 这些点用 RANSAC 拟合平面

    "底部" 的判定：尝试 X/Y/Z 三个轴，取拟合内点率最高的轴方向。
    返回 (plane_coeffs, normal, axis_used, sign_used)
    """
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 1000:
        return None, None, None, None

    candidates = []
    for axis in (0, 1, 2):  # X, Y, Z
        for sign in (+1, -1):
            # sign=+1: 取该轴最大值的点；sign=-1: 取最小值的点
            coords = pts[:, axis] * sign
            thresh = np.percentile(coords, 100 - low_percent)
            mask = coords >= thresh
            if np.count_nonzero(mask) < 200:
                continue
            sub_pcd = pcd.select_by_index(np.where(mask)[0])
            try:
                plane, inliers = sub_pcd.segment_plane(
                    distance_threshold=dist_thresh,
                    ransac_n=3, num_iterations=1000)
            except Exception:
                continue
            a, b, c, d = plane
            normal = np.array([a, b, c])
            # 评分：内点数 + 平面是否与该轴接近平行
            # 接近垂直于该轴 = |normal[axis]| 大
            axis_align = abs(normal[axis])
            score = len(inliers) * axis_align
            candidates.append({
                "plane": plane, "normal": normal,
                "n_inliers": len(inliers), "axis": axis, "sign": sign,
                "axis_align": axis_align, "score": score,
            })

    if not candidates:
        return None, None, None, None

    # 选 score 最大的
    best = max(candidates, key=lambda c: c["score"])
    print(f"  候选平面: ", end="")
    for c in candidates:
        marker = "★" if c is best else " "
        print(f"{marker}axis_{c['axis']}{'+' if c['sign']>0 else '-'}"
              f"(inl={c['n_inliers']},align={c['axis_align']:.2f}) ", end="")
    print()

    return best["plane"], best["normal"], best["axis"], best["sign"]


def rotation_from_vectors(src, dst):
    """求让 src 单位向量旋转到 dst 单位向量的旋转矩阵（Rodrigues）"""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    cos_theta = np.clip(np.dot(src, dst), -1.0, 1.0)
    if cos_theta > 0.9999:
        return np.eye(3)
    if cos_theta < -0.9999:
        # 180°：找任意正交轴
        if abs(src[0]) < 0.9:
            ortho = np.array([1, 0, 0])
        else:
            ortho = np.array([0, 1, 0])
        axis = np.cross(src, ortho)
        axis /= np.linalg.norm(axis)
        return cv2.Rodrigues(axis * np.pi)[0]
    axis = np.cross(src, dst)
    axis /= np.linalg.norm(axis)
    theta = np.arccos(cos_theta)
    return cv2.Rodrigues(axis * theta)[0]


def pca_main_axis(points_2d):
    """对 2D 点做 PCA，返回最大主轴的方向向量"""
    if len(points_2d) < 3:
        return np.array([1, 0])
    mean = points_2d.mean(0)
    centered = points_2d - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[0]  # 第一主成分


def main():
    parser = argparse.ArgumentParser(description="Stage 4: 摆正点云")
    parser.add_argument("--input", required=True)
    parser.add_argument("--low-percent", type=float, default=15.0,
                        help="取点云最低 N%% 区域找盆底 (默认 15)")
    parser.add_argument("--dist-thresh", type=float, default=5.0,
                        help="RANSAC 平面距离阈值 mm")
    parser.add_argument("--no-orient", action="store_true",
                        help="跳过方向校正（只做水平校正）")
    parser.add_argument("--output-name", default="merged_aligned.ply")
    args = parser.parse_args()

    out_dir = os.path.join(args.input, "output_v2")
    input_path = os.path.join(out_dir, "merged_clean.ply")
    if not os.path.isfile(input_path):
        print(f"✗ 找不到 {input_path}，先跑 06_stage3_register.py")
        sys.exit(1)

    print("=" * 64)
    print("Stage 4: 摆正点云")
    print("=" * 64)
    print(f"输入: {input_path}")
    print()

    pcd = o3d.io.read_point_cloud(input_path)
    print(f"加载: {len(pcd.points):,} 点")

    # 当前包围盒
    aabb_in = pcd.get_axis_aligned_bounding_box()
    print(f"当前包围盒: min={aabb_in.min_bound}, max={aabb_in.max_bound}")
    print()

    # ---- 1. 找盆底 ----
    print(f"[1/3] 找盆底平面（low {args.low_percent}%）...")
    plane, normal, axis, sign = find_base_plane(
        pcd, args.low_percent, args.dist_thresh)
    if plane is None:
        print("✗ 找不到合适的盆底平面")
        sys.exit(1)
    a, b, c, d = plane
    print(f"  平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"  法向: ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")
    print(f"  对应轴: {['X','Y','Z'][axis]}{'+' if sign>0 else '-'}")

    # ---- 2. 水平校正：法向 → Z 轴 ----
    print(f"\n[2/3] 水平校正：盆底法向 → Z 轴...")
    # 法向方向选择：让"上方"是 +Z 方向
    # 物体应该在盆底"远离相机"的一侧
    pts = np.asarray(pcd.points)
    signed_dist = a*pts[:, 0] + b*pts[:, 1] + c*pts[:, 2] + d
    if np.median(signed_dist) > 0:
        normal_oriented = normal  # 物体在 +normal 侧
    else:
        normal_oriented = -normal  # 物体在 -normal 侧

    R1 = rotation_from_vectors(normal_oriented, np.array([0, 0, 1]))
    print(f"  R1 (水平校正矩阵):")
    for row in R1:
        print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")

    # 应用 R1
    pts_r1 = (R1 @ pts.T).T

    # 检查：盆底 Z 应该接近常数
    # 找盆底点（用原始平面阈值）
    base_mask = np.abs(signed_dist) < args.dist_thresh * 2
    base_pts_r1 = pts_r1[base_mask]
    if len(base_pts_r1) > 0:
        z_med = np.median(base_pts_r1[:, 2])
        z_std = np.std(base_pts_r1[:, 2])
        print(f"  校正后盆底 Z: median={z_med:.1f}mm, std={z_std:.2f}mm")
        # 移到 Z=0
        z_offset = -z_med
        pts_r1[:, 2] += z_offset
        base_pts_r1[:, 2] += z_offset
    else:
        z_offset = 0

    # ---- 3. 方向校正：盆底 PCA → X 轴 ----
    if not args.no_orient and len(base_pts_r1) >= 100:
        print(f"\n[3/3] 方向校正：盆底 PCA → X 轴...")
        # 盆底点投影到 XY 平面
        base_xy = base_pts_r1[:, :2]
        main_axis = pca_main_axis(base_xy)
        # 计算与 X 轴的夹角
        theta = np.arctan2(main_axis[1], main_axis[0])
        # 让主轴 → +X
        theta_rot = -theta
        cos_t, sin_t = np.cos(theta_rot), np.sin(theta_rot)
        R2 = np.array([[cos_t, -sin_t, 0],
                       [sin_t,  cos_t, 0],
                       [0, 0, 1]])
        print(f"  主轴方向: ({main_axis[0]:+.4f}, {main_axis[1]:+.4f}) → 旋转 {np.degrees(theta_rot):+.2f}°")
        pts_final = (R2 @ pts_r1.T).T
    else:
        R2 = np.eye(3)
        pts_final = pts_r1
        if args.no_orient:
            print(f"\n[3/3] 跳过方向校正（--no-orient）")
        else:
            print(f"\n[3/3] 跳过方向校正（盆底点不足）")

    # 组合变换
    R_final = R2 @ R1
    t_final = np.array([0, 0, z_offset])

    # 输出
    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(pts_final)
    if len(np.asarray(pcd.colors)) > 0:
        aligned.colors = pcd.colors

    out_path = os.path.join(out_dir, args.output_name)
    o3d.io.write_point_cloud(out_path, aligned)
    print()
    print(f"OK {out_path}")

    aabb_out = aligned.get_axis_aligned_bounding_box()
    print(f"  对齐后包围盒: min={aabb_out.min_bound}, max={aabb_out.max_bound}")
    print(f"  高度（Z方向）: {aabb_out.max_bound[2] - aabb_out.min_bound[2]:.1f}mm")
    print(f"  宽度（X方向）: {aabb_out.max_bound[0] - aabb_out.min_bound[0]:.1f}mm")
    print(f"  深度（Y方向）: {aabb_out.max_bound[1] - aabb_out.min_bound[1]:.1f}mm")

    # 保存变换矩阵
    T = np.eye(4)
    T[:3, :3] = R_final
    T[:3, 3] = t_final
    tf_path = os.path.join(out_dir, "align_transform.json")
    with open(tf_path, 'w') as f:
        json.dump({
            "R": R_final.tolist(),
            "t": t_final.tolist(),
            "T_4x4": T.tolist(),
            "base_plane": [float(a), float(b), float(c), float(d)],
            "base_normal": normal_oriented.tolist(),
        }, f, indent=2)
    print(f"  → {tf_path}")

    print()
    print("下一步:")
    print(f"  1. 用 MeshLab 打开 {out_path}，应该看到：")
    print(f"     - 花瓶正立（盆底贴在 Z=0 平面）")
    print(f"     - 盆边对齐 X 轴（如果是方形盆）")
    print(f"  2. OK 后：python 08_stage5_mesh.py --input {args.input}")


if __name__ == "__main__":
    main()
