"""
04_stage1_pcd.py — 单帧点云生成

输入：
  capture_xxx/
    depth/0000.png ...    (16-bit uint16, mm)
    aligned/0000.png ...  (BGR 对齐到深度坐标系)

输出：
  capture_xxx/pcds/
    0000.ply ...          每帧的彩色点云
    pcd_index.json        所有帧的统计信息
    quality_report.html   质量报告（浏览器打开）

使用：
  python 04_stage1_pcd.py --input capture_xxx --calib calibration.json
  python 04_stage1_pcd.py --input capture_xxx --calib calibration.json --n 36
  python 04_stage1_pcd.py --input capture_xxx --calib calibration.json --every 18
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

try:
    import cv2
    import open3d as o3d
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import load_calib, get_intrinsics


def make_colored_pcd(depth_path, aligned_path, K_depth, depth_trunc=2500.0,
                     z_min=500.0, y_min=-400.0):
    """从 depth + aligned RGB 生成彩色点云，去除地板和远背景"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    aligned = cv2.imread(aligned_path)
    if depth is None or aligned is None:
        return None, "图像读取失败"
    if depth.dtype != np.uint16:
        return None, f"深度图不是 uint16 (实际 {depth.dtype})"

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # 过滤：深度范围 + 地板 + 近处杂点
    mask = (Z > z_min) & (Z < depth_trunc) & (Y > y_min)
    n_valid = int(np.count_nonzero(mask))
    if n_valid < 100:
        return None, f"有效点太少 ({n_valid})"

    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    bgr = aligned[mask]
    rgb = bgr[:, ::-1] / 255.0  # BGR → RGB, 0-1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    stats = {
        "n_points": n_valid,
        "depth_min_mm": float(Z[mask].min()),
        "depth_max_mm": float(Z[mask].max()),
        "depth_median_mm": float(np.median(Z[mask])),
        "valid_ratio": n_valid / (h * w),
        "y_min_mm": float(Y[mask].min()),
        "y_max_mm": float(Y[mask].max()),
    }
    return (pcd, stats), None


def write_html_report(index, html_path):
    """生成质量报告 HTML"""
    rows = []
    for entry in index:
        s = entry["stats"]
        warn = ""
        if s["n_points"] < 5000:
            warn = "⚠ 点数偏少"
        elif s["depth_median_mm"] < 800 or s["depth_median_mm"] > 2500:
            warn = "⚠ 距离异常"
        rows.append(f"""
          <tr>
            <td>{entry['id']}</td>
            <td>{s['n_points']:,}</td>
            <td>{s['valid_ratio']*100:.1f}%</td>
            <td>{s['depth_min_mm']:.0f}</td>
            <td>{s['depth_max_mm']:.0f}</td>
            <td>{s['depth_median_mm']:.0f}</td>
            <td style="color:#e67e22">{warn}</td>
          </tr>""")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Stage 1 质量报告</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
  th {{ background: #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
</style></head><body>
<h2>Stage 1: {len(index)} 帧点云</h2>
<p>检查每帧的点数和深度范围。点数过少或深度异常的帧后续 ICP 会失败。</p>
<table>
  <tr><th>帧 ID</th><th>点数</th><th>有效率</th><th>最小深度(mm)</th>
      <th>最大深度(mm)</th><th>中位深度(mm)</th><th>告警</th></tr>
  {"".join(rows)}
</table></body></html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 生成单帧点云")
    parser.add_argument("--input", required=True, help="capture 目录")
    parser.add_argument("--calib", required=True, help="calibration.json")
    parser.add_argument("--n", type=int, default=36,
                        help="均匀抽取 N 帧（默认 36）")
    parser.add_argument("--every", type=int, default=0,
                        help="或每 N 帧取一帧（覆盖 --n）")
    parser.add_argument("--all", action="store_true",
                        help="生成所有帧（覆盖 --n 和 --every）")
    parser.add_argument("--depth-trunc", type=float, default=2500.0,
                        help="深度截断 mm（默认 2500，去掉远背景墙壁）")
    parser.add_argument("--z-min", type=float, default=500.0,
                        help="最小深度 mm（默认 500，去掉过近杂点）")
    parser.add_argument("--y-min", type=float, default=-400.0,
                        help="最小 Y 高度 mm（默认 -400，去掉地板）")
    args = parser.parse_args()

    print("=" * 64)
    print("Stage 1: 单帧点云生成")
    print("=" * 64)

    # 标定
    calib = load_calib(args.calib)
    K_d, _, _ = get_intrinsics(calib, "depth")
    print(f"IR 内参: fx={K_d[0,0]:.2f} fy={K_d[1,1]:.2f} "
          f"cx={K_d[0,2]:.2f} cy={K_d[1,2]:.2f}")

    # 找帧
    depth_dir = os.path.join(args.input, "depth")
    aligned_dir = os.path.join(args.input, "aligned")
    if not os.path.isdir(aligned_dir):
        print(f"✗ 缺少 aligned/ 目录。要求用 03_capture_scan.py 采集的数据")
        sys.exit(1)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    all_ids = sorted([int(os.path.splitext(os.path.basename(f))[0])
                      for f in depth_files])
    if not all_ids:
        print(f"✗ {depth_dir} 没有 PNG")
        sys.exit(1)

    # 抽帧
    if args.all:
        sel_ids = all_ids
    elif args.every > 0:
        sel_ids = all_ids[::args.every]
    else:
        if len(all_ids) <= args.n:
            sel_ids = all_ids
        else:
            idx = np.linspace(0, len(all_ids) - 1, args.n, dtype=int)
            sel_ids = [all_ids[i] for i in idx]

    print(f"总帧数: {len(all_ids)}, 选取: {len(sel_ids)} 帧")
    print(f"深度范围: {args.z_min:.0f} ~ {args.depth_trunc:.0f} mm")
    print(f"Y 最小值: {args.y_min:.0f} mm（去除地板）")
    print()

    # 输出目录
    pcds_dir = os.path.join(args.input, "pcds")
    os.makedirs(pcds_dir, exist_ok=True)

    # 生成
    index = []
    failed = []
    t0 = time.time()
    for k, fid in enumerate(sel_ids):
        d_path = os.path.join(depth_dir, f"{fid:04d}.png")
        a_path = os.path.join(aligned_dir, f"{fid:04d}.png")

        result, err = make_colored_pcd(d_path, a_path, K_d,
                           depth_trunc=args.depth_trunc,
                           z_min=args.z_min, y_min=args.y_min)
        if result is None:
            failed.append({"id": fid, "error": err})
            print(f"  ✗ #{fid:04d}  {err}")
            continue
        pcd, stats = result

        out_path = os.path.join(pcds_dir, f"{fid:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)

        index.append({"id": fid, "stats": stats})

        if (k + 1) % 5 == 0 or k == len(sel_ids) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(sel_ids) - k - 1)
            print(f"  [{k+1:3d}/{len(sel_ids)}] #{fid:04d}  "
                  f"点数 {stats['n_points']:6,}  "
                  f"中位深度 {stats['depth_median_mm']:.0f}mm  "
                  f"ETA {eta:.0f}s")

    print()
    print(f"✓ 完成: {len(index)} 帧成功, {len(failed)} 帧失败  "
          f"耗时 {time.time()-t0:.1f}s")

    # 保存索引
    index_path = os.path.join(pcds_dir, "pcd_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            "depth_trunc_mm": args.depth_trunc,
            "frames": index,
            "failed": failed,
            "K_depth": K_d.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {index_path}")

    # 质量报告
    html_path = os.path.join(pcds_dir, "quality_report.html")
    write_html_report(index, html_path)
    print(f"  → {html_path}")

    # 总结
    if index:
        n_pts = [e["stats"]["n_points"] for e in index]
        d_med = [e["stats"]["depth_median_mm"] for e in index]
        print()
        print("统计:")
        print(f"  点数: 中位 {int(np.median(n_pts)):,}  "
              f"min {min(n_pts):,}  max {max(n_pts):,}")
        print(f"  中位深度: 中位 {np.median(d_med):.0f}mm  "
              f"min {min(d_med):.0f}  max {max(d_med):.0f}")

    print()
    print("下一步：")
    print(f"  1. 用 MeshLab/CloudCompare 打开 {pcds_dir}/0000.ply 检查质量")
    print(f"  2. 浏览器打开 {html_path} 检查所有帧")
    print(f"  3. 如果 OK：python 05_stage2_segment.py --input {args.input}")


if __name__ == "__main__":
    main()
