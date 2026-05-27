"""
01_capture_calib.py — 自动采集 RGB 棋盘标定图（1 fps）

操作流程：
  1. 把棋盘举在 Kinect 前 0.8-2.0m
  2. 程序每秒自动检测 + 保存（仅检测到棋盘时保存）
  3. Q / Esc 退出

使用：
  python 01_capture_calib.py --output calib_imgs --pattern 7x6

输出：
  calib_imgs/color/0000.png 0001.png ...   1920x1080 BGR
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import init_kinect, get_color, find_chessboard


def main():
    parser = argparse.ArgumentParser(description="自动采集 RGB 棋盘标定图")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--pattern", default="7x6",
                        help="棋盘内角点 列x行（默认 7x6）")
    parser.add_argument("--mock", action="store_true", help="无 Kinect 测试模式")
    args = parser.parse_args()

    cols, rows = map(int, args.pattern.lower().split('x'))
    pattern_size = (cols, rows)

    color_dir = os.path.join(args.output, "color")
    os.makedirs(color_dir, exist_ok=True)

    existing = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(color_dir)
                       if f.endswith('.png') and f[:-4].isdigit()])
    snap_idx = (max(existing) + 1) if existing else 0

    print("=" * 64)
    print(f"自动采集棋盘（{cols}x{rows} 内角点）— 1 fps")
    print("=" * 64)
    print(f"输出: {color_dir}")
    print(f"已有 {snap_idx} 张，继续编号...")
    print()
    print("  每秒自动检测，检测到棋盘自动保存")
    print("  Q / Esc 退出")
    print()

    if args.mock:
        kinect = None
    else:
        kinect = init_kinect(need_color=True, need_depth=False, need_ir=False)
        if kinect is None:
            sys.exit(1)
        print("等待 Kinect Color 数据 (最多 10 秒)...")
        warmup_start = time.time()
        while time.time() - warmup_start < 10:
            c = get_color(kinect)
            if c is not None:
                print("  ✓ Color OK\n")
                break
            time.sleep(0.1)
        else:
            print("  ✗ 10 秒内未拿到 Color 帧")
            sys.exit(1)

    cv2.namedWindow("Calib Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calib Capture", 960, 540)

    last_color = None
    last_capture_time = 0
    FPS_INTERVAL = 1.0  # 每秒拍一张

    try:
        while True:
            if args.mock:
                color = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            else:
                color = get_color(kinect)

            if color is None:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            last_color = color

            # 检测棋盘
            color_small = cv2.resize(color, (960, 540))
            gray = cv2.cvtColor(color_small, cv2.COLOR_BGR2GRAY)
            found, corners_small = find_chessboard(gray, pattern_size, use_sb=True)

            # 自动保存（1fps，且检测到棋盘才保存）
            now = time.time()
            if found and (now - last_capture_time >= FPS_INTERVAL):
                last_capture_time = now
                fname = f"{snap_idx:04d}.png"
                cv2.imwrite(os.path.join(color_dir, fname), color)
                print(f"  ✓ #{snap_idx:04d} 保存")
                snap_idx += 1

            # 预览
            disp = cv2.resize(color, (960, 540)).copy()
            if found:
                corners_disp = corners_small  # 已经是 960x540 坐标
                cv2.drawChessboardCorners(disp, pattern_size, corners_disp, True)
                status = f"✓ DETECTED  saved:{snap_idx - 1}"
                status_color = (0, 255, 0)
            else:
                status = "✗ NO CHESSBOARD"
                status_color = (0, 0, 255)

            cv2.rectangle(disp, (0, 0), (960, 40), (30, 30, 30), -1)
            cv2.putText(disp, f"RGB 1fps | {status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(disp, "[Q/Esc] quit", (10, 530),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Calib Capture", disp)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break

    finally:
        cv2.destroyAllWindows()
        if kinect is not None:
            kinect.close()

    print(f"\n采集结束，共 {snap_idx} 张")
    print(f"  color: {color_dir}")
    print(f"\n下一步:")
    print(f"  python 02_calibrate.py --input {args.output} --pattern {args.pattern} --square 25")


if __name__ == "__main__":
    main()
