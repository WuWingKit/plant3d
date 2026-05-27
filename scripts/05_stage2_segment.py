"""
05_stage2_segment.py — 单帧盆栽裁剪（简单圆柱/bbox 裁剪）

前提：04_stage1_pcd.py 已去掉远背景墙壁和地板
本步只做：围绕盆栽中心的空间裁剪 + Y 轴切掉桌面

使用：
  # 自动检测中心
  python 05_stage2_segment.py --input capture_xxx

  # 手动指定
  python 05_stage2_segment.py --input capture_xxx --crop-cx -10 --crop-cz 2150 --crop-radius 350

  # 调整 Y 裁剪（去掉桌面/凳子）
  python 05_stage2_segment.py --input capture_xxx --y-min 200
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


def detect_center(pts, y_min=400, z_min=1400):
    """从 Y>y_min 且 Z>z_min 的点取 XZ 中位数 = 盆栽中心"""
    mask = (pts[:, 1] >= y_min) & (pts[:, 2] >= z_min)
    pts_target = pts[mask]
    if len(pts_target) < 100:
        # fallback: just Y filter
        pts_target = pts[pts[:, 1] >= y_min]
    if len(pts_target) < 100:
        pts_target = pts
    return float(np.median(pts_target[:, 0])), float(np.median(pts_target[:, 2]))


def segment_crop(pcd, crop_cx=None, crop_cz=None, crop_radius=350,
                 y_min=None, y_max=None, mode="cylinder"):
    """简单空间裁剪"""
    pts = np.asarray(pcd.points)
    n0 = len(pts)
    if n0 < 500:
        return o3d.geometry.PointCloud(), {"n_input": n0, "stage": "too_few_input"}

    if crop_cx is None or crop_cz is None:
        crop_cx, crop_cz = detect_center(pts)

    mask = np.ones(n0, dtype=bool)

    if mode == "cylinder":
        dist = np.sqrt((pts[:, 0] - crop_cx) ** 2 + (pts[:, 2] - crop_cz) ** 2)
        mask &= dist <= crop_radius
    else:
        mask &= (pts[:, 0] >= crop_cx - crop_radius) & (pts[:, 0] <= crop_cx + crop_radius)
        mask &= (pts[:, 2] >= crop_cz - crop_radius) & (pts[:, 2] <= crop_cz + crop_radius)

    if y_min is not None:
        mask &= pts[:, 1] >= y_min
    if y_max is not None:
        mask &= pts[:, 1] <= y_max

    idx = np.where(mask)[0]
    if len(idx) < 100:
        return o3d.geometry.PointCloud(), {"n_input": n0, "stage": "too_few_cropped"}

    cropped = pcd.select_by_index(idx)
    stats = {
        "n_input": n0,
        "stage": "ok",
        "crop_center": [float(crop_cx), float(crop_cz)],
        "crop_radius": float(crop_radius),
        "y_min": float(y_min) if y_min is not None else None,
        "n_cropped": len(idx),
        "kept_ratio": len(idx) / n0,
    }
    return cropped, stats


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 简单盆栽裁剪")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--crop-cx", type=float, default=None, help="盆栽中心 X mm")
    parser.add_argument("--crop-cz", type=float, default=None, help="盆栽中心 Z mm")
    parser.add_argument("--crop-radius", type=float, default=350,
                        help="裁剪半径 mm（默认 350）")
    parser.add_argument("--y-min", type=float, default=None,
                        help="Y 轴最低点 mm（建议 150-250，去掉桌面/凳子）")
    parser.add_argument("--y-max", type=float, default=None,
                        help="Y 轴最高点 mm")
    parser.add_argument("--mode", choices=["cylinder", "bbox"], default="cylinder",
                        help="裁剪形状（默认 cylinder）")
    args = parser.parse_args()

    pcds_dir = os.path.join(args.input, "pcds")
    if not os.path.isdir(pcds_dir):
        print(f"✗ 找不到 {pcds_dir}, 先跑 04_stage1_pcd.py")
        sys.exit(1)

    pcd_files = sorted(glob.glob(os.path.join(pcds_dir, "*.ply")))
    if not pcd_files:
        print(f"✗ {pcds_dir} 没有 PLY 文件")
        sys.exit(1)

    out_dir = os.path.join(args.input, "pcds_seg")
    os.makedirs(out_dir, exist_ok=True)

    # 自动检测中心（用第一帧）
    first_pcd = o3d.io.read_point_cloud(pcd_files[0])
    first_pts = np.asarray(first_pcd.points)
    crop_cx, crop_cz = args.crop_cx, args.crop_cz
    if crop_cx is None or crop_cz is None:
        cx, cz = detect_center(first_pts)
        if crop_cx is None:
            crop_cx = cx
        if crop_cz is None:
            crop_cz = cz

    print("=" * 64)
    print("Stage 2: 简单盆栽裁剪")
    print("=" * 64)
    print(f"输入: {pcds_dir} ({len(pcd_files)} 帧)")
    print(f"中心: X={crop_cx:.0f}  Z={crop_cz:.0f}mm  "
          f"半径: {args.crop_radius}mm  模式: {args.mode}")
    if args.y_min is not None:
        print(f"Y 裁剪: {args.y_min:.0f} ~ {args.y_max or '不限'}mm")
    else:
        print(f"Y 裁剪: 不限（包含桌面/凳子）")
        print(f"  提示：加 --y-min 200 可去掉桌面")
    print()

    log = []
    failed = []
    t0 = time.time()

    for k, pcd_path in enumerate(pcd_files):
        fid = int(os.path.splitext(os.path.basename(pcd_path))[0])
        pcd = o3d.io.read_point_cloud(pcd_path)

        cropped, stats = segment_crop(
            pcd,
            crop_cx=crop_cx, crop_cz=crop_cz, crop_radius=args.crop_radius,
            y_min=args.y_min, y_max=args.y_max, mode=args.mode,
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
        "crop_cx": float(crop_cx), "crop_cz": float(crop_cz),
        "crop_radius": args.crop_radius, "mode": args.mode,
        "y_min": args.y_min, "y_max": args.y_max,
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
