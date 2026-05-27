"""
05_stage2_segment_v2.py — 自动检测转盘并裁剪盆栽

自动检测黑色转盘位置，裁剪转盘以上的盆栽内容
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


def detect_turntable(pts, colors):
    """
    自动检测黑色转盘位置

    转盘特征：
    1. 颜色较黑（RGB < 0.2）
    2. 在 Y 轴中间区域
    3. 呈现圆盘形状

    返回：转盘顶部 Y 坐标
    """
    # 黑色点掩码
    black_mask = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)

    if not black_mask.any():
        # 如果没有黑色点，使用默认值
        return 400.0

    black_pts = pts[black_mask]

    # 分析黑色区域的 Y 分布
    y_black = black_pts[:, 1]

    # 找出黑色点最密集的 Y 区域
    y_min, y_max = y_black.min(), y_black.max()
    y_bins = np.linspace(y_min, y_max, 20)
    hist, _ = np.histogram(y_black, bins=y_bins)

    # 找出黑色点最多的 bin
    max_bin_idx = np.argmax(hist)
    y_center = (y_bins[max_bin_idx] + y_bins[max_bin_idx + 1]) / 2

    # 转盘顶部：黑色区域的 Y 最大值 + 一定偏移
    # 因为转盘是水平的，顶部是 Y 最大值
    y_top = y_black.max()

    return float(y_top)


def segment_plant(pcd, y_turntable_top, crop_radius=400):
    """
    裁剪转盘以上的盆栽

    参数：
        pcd: 输入点云
        y_turntable_top: 转盘顶部 Y 坐标
        crop_radius: 裁剪半径（mm）

    返回：
        裁剪后的点云
    """
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    n0 = len(pts)

    if n0 < 100:
        return o3d.geometry.PointCloud(), {"n_input": n0, "stage": "too_few_input"}

    # 步骤 1：裁剪 Y > 转盘顶部（保留转盘以上的盆栽）
    y_mask = pts[:, 1] > y_turntable_top

    # 步骤 2：在 XZ 平面上裁剪，围绕盆栽中心
    if y_mask.any():
        # 用 Y > 转盘顶部的点计算中心
        plant_pts = pts[y_mask]
        cx = np.median(plant_pts[:, 0])
        cz = np.median(plant_pts[:, 2])
    else:
        # fallback
        cx = np.median(pts[:, 0])
        cz = np.median(pts[:, 2])

    # XZ 平面上的距离裁剪
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
        "y_turntable_top": float(y_turntable_top),
        "crop_center": [float(cx), float(cz)],
        "crop_radius": float(crop_radius),
        "n_cropped": len(idx),
        "kept_ratio": len(idx) / n0,
    }
    return cropped, stats


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 自动检测转盘并裁剪盆栽")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--crop-radius", type=float, default=400,
                        help="裁剪半径 mm（默认 400）")
    parser.add_argument("--y-offset", type=float, default=0,
                        help="转盘顶部 Y 偏移 mm（正值向上，负值向下）")
    parser.add_argument("--y-manual", type=float, default=None,
                        help="手动指定转盘顶部 Y 坐标 mm（覆盖自动检测）")
    args = parser.parse_args()

    pcds_dir = os.path.join(args.input, "pcds")
    if not os.path.isdir(pcds_dir):
        print(f"✗ 找不到 {pcds_dir}, 先跑 04_stage1_pcd.py")
        sys.exit(1)

    pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if not pcd_files:
        print(f"✗ {pcds_dir} 没有 PLY 文件")
        sys.exit(1)

    # 自动检测转盘位置（用第一帧）
    first_pcd = o3d.io.read_point_cloud(pcd_files[0])
    first_pts = np.asarray(first_pcd.points)
    first_colors = np.asarray(first_pcd.colors) if first_pcd.has_colors() else None

    if args.y_manual is not None:
        y_turntable_top = args.y_manual
        print(f"使用手动指定的转盘顶部 Y: {y_turntable_top:.0f}mm")
    else:
        y_turntable_top = detect_turntable(first_pts, first_colors)
        y_turntable_top += args.y_offset
        print(f"自动检测转盘顶部 Y: {y_turntable_top:.0f}mm")

    print("=" * 64)
    print("Stage 2: 自动检测转盘并裁剪盆栽")
    print("=" * 64)
    print(f"输入: {pcds_dir} ({len(pcd_files)} 帧)")
    print(f"转盘顶部 Y: {y_turntable_top:.0f}mm")
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
            y_turntable_top=y_turntable_top,
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
        "y_turntable_top": float(y_turntable_top),
        "crop_radius": args.crop_radius,
        "y_offset": args.y_offset,
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
