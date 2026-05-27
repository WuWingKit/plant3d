"""
01b_capture_calib_rgb.py — 仅采集 RGB 棋盘照片（用于纯 RGB 标定）

不依赖 IR 棋盘检测（绕开散斑/低对比度问题）。

用法：
  python 01b_capture_calib_rgb.py --output calib_imgs --pattern 8x6

操作：
  把棋盘举在 Kinect 前 0.8-2.0m 距离，覆盖远近+不同姿态
  绿色提示时按 SPACE 保存
  目标 30-40 张
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import init_kinect, get_color, find_chessboard


def main():
    parser = argparse.ArgumentParser(description="仅 RGB 棋盘采集")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="8x6", help="内角点 列x行 (默认 8x6)")
    args = parser.parse_args()

    cols, rows = map(int, args.pattern.lower().split('x'))
    pattern_size = (cols, rows)

    color_dir = os.path.join(args.output, "color")
    os.makedirs(color_dir, exist_ok=True)

    existing = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(color_dir)
                       if f.endswith('.png') and f[:-4].isdigit()])
    snap_idx = (max(existing) + 1) if existing else 0

    print("=" * 64)
    print(f"RGB 棋盘采集 ({cols}x{rows} 内角点)")
    print("=" * 64)
    print(f"已有 {snap_idx} 张，继续编号...")
    print()
    print("拍摄建议:")
    print("  距离 0.8 / 1.2 / 1.6 / 2.0m 各拍 8-10 张")
    print("  每档姿态多样: 正面/上下倾/左右转/旋转")
    print("  目标 30-40 张")
    print()

    kinect = init_kinect()
    if kinect is None:
        sys.exit(1)

    # 等 Kinect 准备好
    print("等待 Kinect (最多 10 秒)...")
    start = time.time()
    while time.time() - start < 10:
        if get_color(kinect) is not None:
            print("  ✓ Color OK\n")
            break
        time.sleep(0.1)
    else:
        print("  ✗ 拿不到 Color，请跑 test_kinect.py 诊断")
        sys.exit(1)

    cv2.namedWindow("RGB Calib", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB Calib", 960, 600)

    last_detect = 0
    cached_ok = False
    cached_corners = None
    last_color = None

    try:
        while True:
            c = get_color(kinect)
            if c is not None:
                last_color = c
            if last_color is None:
                if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                    break
                continue
            color = last_color

            # 节流检测 (0.2 秒一次)
            now = time.time()
            if now - last_detect > 0.2:
                last_detect = now
                small = cv2.resize(color, (960, 540))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                ok, corners_small = find_chessboard(gray, pattern_size, use_sb=True)
                cached_ok = ok
                cached_corners = corners_small * 2.0 if ok else None

            # 绘制
            disp = cv2.resize(color, (960, 540)).copy()
            if cached_ok:
                corners_disp = cached_corners * (960.0 / 1920)
                cv2.drawChessboardCorners(disp, pattern_size, corners_disp, True)

            # 状态条
            canvas = np.zeros((600, 960, 3), dtype=np.uint8)
            canvas[:540] = disp

            if cached_ok:
                status = f"OK - Press SPACE to save (saved: {snap_idx})"
                color_t = (0, 255, 0)
            else:
                status = "Chessboard NOT detected - adjust position/angle"
                color_t = (0, 0, 255)

            cv2.rectangle(canvas, (0, 540), (960, 600), (30, 30, 30), -1)
            cv2.putText(canvas, status, (10, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_t, 2)
            cv2.putText(canvas, "[SPACE] save  [Q/Esc] quit",
                        (10, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("RGB Calib", canvas)
            key = cv2.waitKey(15) & 0xFF
            if key == ord(' '):
                if not cached_ok:
                    print("  ✗ 未检测到棋盘，不保存")
                    continue
                fname = f"{snap_idx:04d}.png"
                cv2.imwrite(os.path.join(color_dir, fname), color)
                print(f"  ✓ #{snap_idx:04d} 保存")
                snap_idx += 1
                # 闪烁
                flash = canvas.copy()
                cv2.rectangle(flash, (0, 0), (960, 600), (0, 255, 0), 8)
                cv2.imshow("RGB Calib", flash)
                cv2.waitKey(150)
            elif key in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()

    print()
    print(f"采集结束，共 {snap_idx} 张")
    print(f"  → {color_dir}")
    print()
    print("下一步:")
    print(f"  python 02_calibrate.py --input {args.output} --pattern {args.pattern} --square 25 --rgb-only")


if __name__ == "__main__":
    main()
