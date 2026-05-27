"""
05_stage2_segment_v3.py — 改进的盆栽裁剪

根据颜色和空间分布自动识别盆栽区域
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import glob
import argparse
import time
import numpy as np
import open3d as o3d


def detect_plant_region(pts, colors):
    """
    自动检测盆栽区域

    盆栽特征：
    1. 在转盘上方（Y 较大）
    2. 包含绿色（植物）和白色（花盆）区域
    3. 在 XZ 平面上居中

    返回：(y_min, y_max, cx, cz)
    """
    # 分析颜色分布
    # 绿色点：G > R 且 G > B
    green_mask = (colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]) & (colors[:, 1] > 0.2)

    # 白色点：RGB 都较高
    white_mask = (colors[:, 0] > 0.5) & (colors[:, 1] > 0.5) & (colors[:, 2] > 0.5)

    # 黑色点：RGB 都较低
    black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)

    # 找出绿色和白色点的 Y 范围
    plant_mask = green_mask | white_mask
    if not plant_mask.any():
        # 如果没有绿色或白色点，使用默认值
        return pts[:, 1].min(), pts[:, 1].max(), np.median(pts[:, 0]), np.median(pts[:, 2])

    plant_pts = pts[plant_mask]
    y_min = plant_pts[:, 1].min()
    y_max = plant_pts[:, 1].max()

    # 计算 XZ 中心
    cx = np.median(plant_pts[:, 0])
    cz = np.median(plant_pts[:, 2])

    return float(y_min), float(y_max), float(cx), float(cz)


def segment_plant(pcd, y_min, y_max, cx, cz, crop_radius=400):
    """
    裁剪盆栽区域

    参数：
        pcd: 输入点云
        y_min: 盆栽 Y 最小值
        y_max: 盆栽 Y 最大值
        cx: 盆栽中心 X
        cz: 盆栽中心 Z
        crop_radius: 裁剪半径（mm）

    返回：
        裁剪后的点云
    """
    pts = np.asarray(pcd.points)
    n0 = len(pts)

    if n0 < 100:
        return o3d.geometry.PointCloud(), {"n_input": n0, "stage": "too_few_input"}

    # Y 轴裁剪
    y_mask = (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)

    # XZ 平面裁剪
    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 2] - cz) ** 2)
    xz_mask = dist <= crop_radius

    # 组合掩码
    mask = y_mask & xz_mask

    idx = np.where(mask)[0]
    if len(idx) < 50:
        return o3d.geometry.PointCloud(), {"n_input": n0, "stage": "too_few_cropped"}

    cropped = pcd.select_by_index(idx)

    stats = {
        "n_input": n0,
        "stage": "ok",
        "y_range": [float(y_min), float(y_max)],
        "crop_center": [float(cx), float(cz)],
        "crop_radius": float(crop_radius),
        "n_cropped": len(idx),
        "kept_ratio": len(idx) / n0,
    }
    return cropped, stats


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 改进的盆栽裁剪")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--crop-radius", type=float, default=400,
                        help="裁剪半径 mm（默认 400）")
    parser.add_argument("--y-min", type=float, default=None,
                        help="手动指定盆栽 Y 最小值 mm")
    parser.add_argument("--y-max", type=float, default=None,
                        help="手动指定盆栽 Y 最大值 mm")
    args = parser.parse_args()

    pcds_dir = os.path.join(args.input, "pcds")
    if not os.path.isdir(pcds_dir):
        print(f"✗ 找不到 {pcds_dir}, 先跑 04_stage1_pcd.py")
        sys.exit(1)

    pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if not pcd_files:
        print(f"✗ {pcds_dir} 没有 PLY 文件")
        sys.exit(1)

    # 自动检测盆栽区域（用第一帧）
    first_pcd = o3d.io.read_point_cloud(pcd_files[0])
    first_pts = np.asarray(first_pcd.points)
    first_colors = np.asarray(first_pcd.colors) if first_pcd.has_colors() else None

    if args.y_min is not None and args.y_max is not None:
        y_min, y_max = args.y_min, args.y_max
        # 计算中心
        mask = (first_pts[:, 1] >= y_min) & (first_pts[:, 1] <= y_max)
        if mask.any():
            cx = np.median(first_pts[mask, 0])
            cz = np.median(first_pts[mask, 2])
        else:
            cx = np.median(first_pts[:, 0])
            cz = np.median(first_pts[:, 2])
        print(f"使用手动指定的 Y 范围: {y_min:.0f} ~ {y_max:.0f}mm")
    else:
        y_min, y_max, cx, cz = detect_plant_region(first_pts, first_colors)
        print(f"自动检测盆栽区域:")
        print(f"  Y 范围: {y_min:.0f} ~ {y_max:.0f}mm")
        print(f"  中心: X={cx:.0f}, Z={cz:.0f}mm")

    print("=" * 64)
    print("Stage 2: 改进的盆栽裁剪")
    print("=" * 64)
    print(f"输入: {pcds_dir} ({len(pcd_files)} 帧)")
    print(f"Y 范围: {y_min:.0f} ~ {y_max:.0f}mm")
    print(f"中心: X={cx:.0f}, Z={cz:.0f}mm")
    print(f"裁剪半径: {args.crop_radius}mm")
    print()

    out_dir = os.path.join(args.input, "pcds_seg")
    os.makedirs(out_dir, exist_ok=True)

    log = []
    failed = []
    t0 = time.time()

    for k, pcd_path in enumerate(pcd_files):
        fid = int(os.path.splitext(os.path.basename(pcd_path))[0])
        pcd = o3d.io.read_point_cloud(pcd_path)

        cropped, stats = segment_plant(
            pcd,
            y_min=y_min,
            y_max=y_max,
            cx=cx,
            cz=cz,
            crop_radius=args.crop_radius,
        )
        stats["id"] = fid

        if stats["stage"] != "ok":
            failed.append(stats)
            if (k + 1) % 10 == 0 or k == len(pcd_files) - 1:
                print(f"  [{k+1:3d}/{len(pcd_files)}] ✗ #{fid:04d}  {stats['stage']}")
            continue

        out_path = os.path.join(out_dir, f"{fid:04d}.ply")
        o3d.io.write_point_cloud(out_path, cropped)
        log.append(stats)

        if (k + 1) % 10 == 0 or k == len(pcd_files) - 1:
            elapsed = time.time() - t0
            print(f"  [{k+1:3d}/{len(pcd_files)}] #{fid:04d}  "
                  f"{stats['n_input']:6,} -> {stats['n_cropped']:6,}  "
                  f"({stats['kept_ratio']*100:.1f}%)")

    print(f"\n✓ 完成: {len(log)} 成功, {len(failed)} 失败  "
          f"耗时 {time.time() - t0:.1f}s")

    # 保存日志
    log_path = os.path.join(out_dir, "segment_log.json")
    params = {
        "y_range": [float(y_min), float(y_max)],
        "crop_center": [float(cx), float(cz)],
        "crop_radius": args.crop_radius,
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({"params": params, "frames": log, "failed": failed},
                  f, indent=2, ensure_ascii=False)
    print(f"  -> {log_path}")

    if log:
        kept_ratios = [s["kept_ratio"] for s in log]
        n_cropped = [s["n_cropped"] for s in log]
        print(f"\n统计:")
        print(f"  保留比例: 中位 {np.median(kept_ratios)*100:.1f}%  "
              f"min {min(kept_ratios)*100:.1f}%  max {max(kept_ratios)*100:.1f}%")
        print(f"  保留点数: 中位 {int(np.median(n_cropped)):,}  "
              f"min {min(n_cropped):,}  max {max(n_cropped):,}")

    print(f"\n下一步: python 06_stage3_register.py --input {args.input}")


if __name__ == "__main__":
    main()
