"""
02_calibrate.py — 完整 Kinect v2 标定（RGB + IR + 立体外参）

输入：01_capture_calib.py 采集的同步棋盘对
输出：calibration.json，包含：
  - color.K, color.dist, color.image_size, color.rms_error
  - depth.K, depth.dist, depth.image_size, depth.rms_error
  - stereo_depth_to_color.R, T (IR/深度坐标系 → 彩色坐标系)

流程（张正友法 × 2 + 立体标定）：
  1. 加载所有同步对
  2. 在 RGB 图和 IR 图分别检测角点
  3. 只保留双方都检测到的对（typical 65-80% 成功率）
  4. 单相机标定 RGB → K_c, dist_c
  5. 单相机标定 IR/深度 → K_d, dist_d
  6. 立体标定 (固定内参) → R, T

使用：
  python 02_calibrate.py --input calib_imgs --pattern 7x6 --square 25
  python 02_calibrate.py --input calib_imgs --pattern 7x6 --square 25 \
                            --output calibration.json
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import (
    ir_to_8bit, find_chessboard, save_calib,
)


def detect_pair(color_path, ir_path, pattern_size):
    """检测一对 RGB+IR 图像的棋盘角点"""
    color = cv2.imread(color_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if color is None or ir is None:
        return None, None, None, None

    # RGB 检测
    gray_c = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    ret_c, corners_c = find_chessboard(gray_c, pattern_size, use_sb=True)

    # IR 检测（先 denoised 预处理 = CLAHE + 中值滤波，专门去散斑）
    ir_8bit = ir_to_8bit(ir, mode="denoised") if ir.dtype == np.uint16 else \
              cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY) if len(ir.shape) == 3 else ir
    ret_i, corners_i = find_chessboard(ir_8bit, pattern_size, use_sb=True)
    if not ret_i:
        # SB 失败 → 经典算法兜底
        ret_i, corners_i = find_chessboard(ir_8bit, pattern_size, use_sb=False)

    return ret_c, corners_c, ret_i, corners_i


def run_rgb_only(args, pattern_size, color_dir, output_path):
    """RGB-only 标定：只标定彩色相机，深度用 Kinect 出厂值，外参留待手动对齐"""
    cols, rows = pattern_size

    print("=" * 64)
    print("Kinect v2 RGB-only 标定")
    print("=" * 64)
    print(f"棋盘: {cols}x{rows} 内角点, 格边长 {args.square}mm")
    print(f"输入: {color_dir}")
    print(f"模式: 只标定 RGB（深度用出厂值，外参留待 02b_manual_align.py 手动调）")
    print()

    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    if not color_files:
        print(f"✗ {color_dir} 没有 PNG")
        sys.exit(1)

    print(f"图像数: {len(color_files)}")
    print()

    # 物理坐标
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= args.square

    obj_pts = []
    img_pts = []
    valid_files = []
    color_size = None

    print("[1/2] 检测角点...")
    for f in color_files:
        img = cv2.imread(f)
        if img is None:
            continue
        if color_size is None:
            color_size = img.shape[1::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = find_chessboard(gray, pattern_size, use_sb=True)
        if ret:
            obj_pts.append(objp)
            img_pts.append(corners)
            valid_files.append(os.path.basename(f))
            print(f"  ✓ {os.path.basename(f)}")
        else:
            print(f"  ✗ {os.path.basename(f)}")

    print()
    print(f"有效图: {len(obj_pts)}/{len(color_files)}")
    if len(obj_pts) < 10:
        print(f"✗ 有效图太少，建议至少 15 张")
        sys.exit(1)

    # ---- 标定 ----
    print()
    print(f"[2/2] 张正友法标定...")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, color_size, None, None)

    print(f"  RMS = {rms:.4f} px  "
          f"{'✓ 优秀' if rms < 0.5 else '✓ 可用' if rms < 1.0 else '⚠ 偏高'}")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}")
    print(f"  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  dist={D.ravel()}")

    # ---- 保存 ----
    calib = {
        "pattern_inner_corners": [cols, rows],
        "square_size_mm": args.square,
        "n_valid_pairs": len(obj_pts),
        "mode": "rgb_only",
        "color": {
            "image_size": list(color_size),
            "K": K.tolist(),
            "dist": D.ravel().tolist(),
            "rms_error": float(rms),
            "valid_images": valid_files,
        },
        "depth": {
            "image_size": [512, 424],
            "K": [[365.456, 0.0, 254.878],
                  [0.0, 365.456, 205.395],
                  [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "note": "Kinect v2 factory default (no IR calibration)",
        },
        "stereo_depth_to_color": {
            "R_depth_to_color": [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]],
            "T_depth_to_color": [-52.0, 0.0, 0.0],
            "note": "Kinect v2 factory default. Run 02b_manual_align.py to refine.",
        },
    }

    save_calib(output_path, calib)
    print()
    print("=" * 64)
    print(f"✓ RGB 标定完成: {output_path}")
    print()
    print("深度内参 + 立体外参 = Kinect 出厂默认值")
    print("颜色对齐误差通常 1-2cm (出厂值的固有误差)")
    print()
    print("下一步选一:")
    print("  A. 手动微调外参 (颜色对齐降到 2-3mm):")
    print(f"     python 02b_manual_align.py --calib {output_path}")
    print("  B. 不调，直接用出厂值:")
    print(f"     python 03_capture_scan.py --output capture_xxx --calib {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Kinect v2 标定")
    parser.add_argument("--input", required=True, help="包含 color/ 子目录（联合模式还需要 ir/）")
    parser.add_argument("--pattern", default="7x6", help="内角点 列x行")
    parser.add_argument("--square", type=float, default=25.0, help="棋盘格边长 mm")
    parser.add_argument("--output", default=None,
                        help="输出 JSON（默认 input/calibration.json）")
    parser.add_argument("--min-pairs", type=int, default=15,
                        help="双方都检测到的最少对数")
    parser.add_argument("--rgb-only", action="store_true",
                        help="仅标定 RGB 相机（不需要 IR 棋盘）")
    args = parser.parse_args()

    cols, rows = map(int, args.pattern.lower().split('x'))
    pattern_size = (cols, rows)
    output_path = args.output or os.path.join(args.input, "calibration.json")

    color_dir = os.path.join(args.input, "color")

    if not os.path.isdir(color_dir):
        print(f"✗ 缺少 color/ 子目录")
        sys.exit(1)

    # ============ RGB-only 分支 ============
    if args.rgb_only:
        run_rgb_only(args, pattern_size, color_dir, output_path)
        return

    # ============ 联合标定分支（需要 IR）============
    ir_dir = os.path.join(args.input, "ir")
    if not os.path.isdir(ir_dir):
        print(f"✗ 缺少 ir/ 子目录（联合标定需要）")
        print(f"  如果只想标定 RGB，加 --rgb-only")
        sys.exit(1)

    # 配对（同名文件）
    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.png")))
    color_dict = {os.path.basename(f): f for f in color_files}
    ir_dict = {os.path.basename(f): f for f in ir_files}
    common = sorted(set(color_dict) & set(ir_dict))

    print("=" * 64)
    print("Kinect v2 联合标定（张正友 + 立体）")
    print("=" * 64)
    print(f"棋盘: {cols}x{rows} 内角点, 格边长 {args.square}mm")
    print(f"输入: {args.input}")
    print(f"同名对: {len(common)}")
    print()

    # ---- 物理坐标（棋盘平面 Z=0）----
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= args.square

    # ---- 检测所有对的角点 ----
    print("[1/4] 检测角点...")
    obj_pts = []
    img_pts_c = []
    img_pts_i = []
    valid_pairs = []

    n_only_c = 0
    n_only_i = 0
    n_neither = 0
    n_both = 0

    color_size = None
    ir_size = None

    for fname in common:
        ret_c, corn_c, ret_i, corn_i = detect_pair(
            color_dict[fname], ir_dict[fname], pattern_size)

        if ret_c and color_size is None:
            color_img = cv2.imread(color_dict[fname])
            color_size = color_img.shape[1::-1]  # (w, h)
        if ret_i and ir_size is None:
            ir_img = cv2.imread(ir_dict[fname], cv2.IMREAD_UNCHANGED)
            ir_size = ir_img.shape[1::-1]

        if ret_c and ret_i:
            obj_pts.append(objp)
            img_pts_c.append(corn_c)
            img_pts_i.append(corn_i)
            valid_pairs.append(fname)
            n_both += 1
            print(f"  ✓ {fname}")
        elif ret_c:
            n_only_c += 1
            print(f"  ◐ {fname} (仅 RGB)")
        elif ret_i:
            n_only_i += 1
            print(f"  ◑ {fname} (仅 IR)")
        else:
            n_neither += 1
            print(f"  ✗ {fname}")

    total = len(common)
    print()
    print(f"统计: 双方 {n_both}/{total} ({100*n_both/total:.0f}%) | "
          f"仅 RGB {n_only_c} | 仅 IR {n_only_i} | 都无 {n_neither}")

    if n_both < args.min_pairs:
        print(f"✗ 有效对数不足（{n_both} < {args.min_pairs}），无法标定")
        print(f"  建议：补拍。先看下'仅 RGB'的图像，多数是 IR 检测失败")
        print(f"  常见原因：散斑过强、棋盘太远（>2m）、太小（<画面 1/4）、")
        print(f"           背景反光、棋盘印刷不黑等")
        sys.exit(1)

    # ---- 单相机标定 RGB ----
    print()
    print(f"[2/4] 标定彩色相机（{n_both} 张）...")
    rms_c, K_c, D_c, rvecs_c, tvecs_c = cv2.calibrateCamera(
        obj_pts, img_pts_c, color_size, None, None)
    print(f"  RMS = {rms_c:.4f} px  "
          f"{'✓ 优秀' if rms_c < 0.5 else '✓ 可用' if rms_c < 1.0 else '⚠ 偏高'}")
    print(f"  fx={K_c[0,0]:.2f} fy={K_c[1,1]:.2f}")
    print(f"  cx={K_c[0,2]:.2f} cy={K_c[1,2]:.2f}")
    print(f"  dist={D_c.ravel()}")

    # ---- 单相机标定 IR/深度 ----
    print()
    print(f"[3/4] 标定 IR/深度 相机（{n_both} 张）...")
    rms_i, K_i, D_i, rvecs_i, tvecs_i = cv2.calibrateCamera(
        obj_pts, img_pts_i, ir_size, None, None)
    print(f"  RMS = {rms_i:.4f} px  "
          f"{'✓ 优秀' if rms_i < 0.5 else '✓ 可用' if rms_i < 1.0 else '⚠ 偏高'}")
    print(f"  fx={K_i[0,0]:.2f} fy={K_i[1,1]:.2f}")
    print(f"  cx={K_i[0,2]:.2f} cy={K_i[1,2]:.2f}")
    print(f"  dist={D_i.ravel()}")

    # 跟出厂值比较一下
    print(f"  对比 Kinect 出厂值 (fx=fy=365.46, cx=254.88, cy=205.40):")
    print(f"    fx 误差: {K_i[0,0]-365.456:+.2f}, fy 误差: {K_i[1,1]-365.456:+.2f}")
    print(f"    cx 误差: {K_i[0,2]-254.878:+.2f}, cy 误差: {K_i[1,2]-205.395:+.2f}")

    # ---- 立体标定 ----
    print()
    print(f"[4/4] 立体标定（IR/深度 → 彩色 的外参）...")
    flags = cv2.CALIB_FIX_INTRINSIC  # 锁定内参，只优化 R, T
    ret_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts,
        img_pts_i,    # 源 = IR/深度
        img_pts_c,    # 目标 = RGB
        K_i, D_i, K_c, D_c,
        ir_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6),
    )
    print(f"  立体 RMS = {ret_s:.4f} px")
    print(f"  T (mm) = [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}]")
    print(f"    (Kinect 出厂近似 [-52, 0, 0])")
    # 提取旋转角
    rvec, _ = cv2.Rodrigues(R)
    deg = np.degrees(rvec.ravel())
    print(f"  R 旋转 (deg, rx ry rz) = [{deg[0]:+.3f}, {deg[1]:+.3f}, {deg[2]:+.3f}]")
    print(f"    (理想接近 [0, 0, 0]，几度内是正常的)")

    # ---- 保存 ----
    calib = {
        "pattern_inner_corners": [cols, rows],
        "square_size_mm": args.square,
        "n_valid_pairs": n_both,
        "color": {
            "image_size": list(color_size),
            "K": K_c.tolist(),
            "dist": D_c.ravel().tolist(),
            "rms_error": float(rms_c),
        },
        "depth": {
            "image_size": list(ir_size),
            "K": K_i.tolist(),
            "dist": D_i.ravel().tolist(),
            "rms_error": float(rms_i),
        },
        "stereo_depth_to_color": {
            "R_depth_to_color": R.tolist(),
            "T_depth_to_color": T.ravel().tolist(),
            "rms_error": float(ret_s),
            "essential_matrix": E.tolist(),
        },
    }

    save_calib(output_path, calib)
    print()
    print("=" * 64)
    print(f"✓ 标定完成: {output_path}")
    print()
    print("下一步:")
    print(f"  python 03_capture_scan.py --output capture_花瓶_xxx --calib {output_path}")


if __name__ == "__main__":
    main()
