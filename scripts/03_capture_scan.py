"""
03_capture_scan.py — 扫描花瓶时的数据采集

特点：
  - 每帧同时存：原始 RGB + 原始深度 + 对齐 RGB（关键！）
  - 实时预览对齐效果（深度图上叠加彩色叠加）
  - 实时质量检查（深度有效率、对齐质量）

使用：
  python 03_capture_scan.py --output capture_花瓶_xxx --calib calibration.json
  python 03_capture_scan.py --output capture_花瓶_xxx --calib calibration.json --fps 5

操作：
  - 开 Kinect → 转盘启动 → 等转速稳定
  - 按 SPACE 开始录制 → 录够 1-2 圈 → 按 SPACE 停止
  - 程序问"转了几圈/几秒" → 填入 metadata

输出：
  capture_xxx/
    color/0000.png  ...    1920x1080 BGR (原始)
    depth/0000.png  ...    512x424 uint16 mm
    aligned/0000.png ...   512x424 BGR (彩色对齐到深度坐标系，每像素直接对应)
    timestamps.csv
    metadata.json
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import (
    init_kinect, wait_synced,
    load_calib, get_intrinsics, get_stereo,
    make_aligned_color, ir_to_8bit,
)


def main():
    parser = argparse.ArgumentParser(description="扫描花瓶（带实时对齐）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--fps", type=float, default=8.0,
                        help="目标 FPS（默认 8，覆盖一圈 30s 约 240 帧）")
    parser.add_argument("--duration", type=float, default=0,
                        help="自动停止时长（秒，0=按键停止）")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    # 加载标定
    calib = load_calib(args.calib)
    K_d, dist_d, depth_size = get_intrinsics(calib, "depth")
    K_c, dist_c, color_size = get_intrinsics(calib, "color")
    R_d2c, T_d2c = get_stereo(calib)

    print("=" * 60)
    print("扫描采集（含实时对齐）")
    print("=" * 60)
    print(f"标定: {args.calib}")
    print(f"  RGB RMS={calib['color'].get('rms_error','?')} ")
    print(f"  IR  RMS={calib['depth'].get('rms_error','?')} ")
    print(f"  Stereo RMS={calib.get('stereo_depth_to_color',{}).get('rms_error','?')} ")
    print()

    # 输出目录
    color_dir = os.path.join(args.output, "color")
    depth_dir = os.path.join(args.output, "depth")
    aligned_dir = os.path.join(args.output, "aligned")
    for d in (color_dir, depth_dir, aligned_dir):
        os.makedirs(d, exist_ok=True)

    kinect = None if args.mock else init_kinect()
    if kinect is None and not args.mock:
        sys.exit(1)

    cv2.namedWindow("Scan Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan Capture", 1280, 720)

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    frame_idx = 0
    recording = False
    start_time = None
    last_save = 0
    timestamps = []

    print("操作：")
    print("  把花瓶放在转盘上，启动转盘")
    print("  等转速稳定后按 SPACE 开始 → 录够 1-2 圈 → SPACE 停止")
    print("  Q/Esc 退出")
    print()

    try:
        while True:
            color, depth, ir = wait_synced(kinect, need_ir=True) if not args.mock else (
                np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
                np.random.randint(800, 2500, (424, 512), dtype=np.uint16),
                np.random.randint(100, 2000, (424, 512), dtype=np.uint16),
            )
            if color is None or depth is None:
                cv2.waitKey(5)
                continue

            now = time.time()

            # 计算对齐 RGB（实时算，约 30ms 一次）
            aligned = make_aligned_color(depth, color, K_d, K_c, dist_c, R_d2c, T_d2c)

            # ---- 预览 ----
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

            # 左上：原始彩色
            c_small = cv2.resize(color, (640, 360))
            canvas[0:360, 0:640] = c_small
            cv2.putText(canvas, "Color (1920x1080)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 右上：深度伪彩
            d_vis = np.clip(depth.astype(np.float32) / 3000 * 255, 0, 255).astype(np.uint8)
            d_vis[depth == 0] = 0
            d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
            d_vis_big = cv2.resize(d_vis, (640, 360))
            canvas[0:360, 640:1280] = d_vis_big
            cv2.putText(canvas, "Depth (JET)", (650, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 左下：对齐 RGB（512x424 → 640x530 等比）
            aligned_big = cv2.resize(aligned, (640, 530))
            canvas[360:720, 0:640] = aligned_big[:360]
            cv2.putText(canvas, "Aligned RGB (depth space)", (10, 385),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 右下：对齐质量叠加 (aligned RGB + depth 半透明 - 应该重合)
            aligned_resized = cv2.resize(aligned, (640, 360))
            overlay = cv2.addWeighted(aligned_resized, 0.6, d_vis_big, 0.4, 0)
            canvas[360:720, 640:1280] = overlay
            cv2.putText(canvas, "Alignment Check (RGB+Depth overlay)", (650, 385),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # 信息条
            valid_ratio = np.count_nonzero(depth) / depth.size
            median_depth = np.median(depth[depth > 0]) if np.any(depth > 0) else 0

            if recording:
                elapsed = now - start_time
                actual_fps = frame_idx / elapsed if elapsed > 0 else 0
                info1 = f"REC #{frame_idx}  {elapsed:.1f}s  {actual_fps:.1f}fps"
                info_color = (0, 0, 255)
            else:
                info1 = "READY  Press SPACE to start"
                info_color = (0, 255, 0)

            info2 = f"Depth valid: {valid_ratio*100:.0f}% | median: {median_depth:.0f}mm"

            cv2.rectangle(canvas, (0, 690), (1280, 720), (30, 30, 30), -1)
            cv2.putText(canvas, info1, (10, 712),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
            cv2.putText(canvas, info2, (450, 712),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "[SPACE] start/stop  [Q] quit", (900, 712),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Scan Capture", canvas)

            # 录制保存
            if recording and (now - last_save) >= interval:
                cv2.imwrite(os.path.join(color_dir, f"{frame_idx:04d}.png"), color)
                cv2.imwrite(os.path.join(depth_dir, f"{frame_idx:04d}.png"),
                            depth.astype(np.uint16))
                cv2.imwrite(os.path.join(aligned_dir, f"{frame_idx:04d}.png"), aligned)
                timestamps.append({"id": frame_idx, "timestamp": now - start_time})
                frame_idx += 1
                last_save = now
                if frame_idx % 30 == 0:
                    print(f"  录 {frame_idx} 帧 ({now-start_time:.1f}s)")

            # 自动停止
            if args.duration > 0 and recording and (now - start_time) >= args.duration:
                print(f"达到 {args.duration}s 自动停止")
                break

            key = cv2.waitKey(15) & 0xFF
            if key == ord(' '):
                if not recording:
                    recording = True
                    start_time = time.time()
                    last_save = start_time - interval
                    print(f"▶ 开始 @ {datetime.now().strftime('%H:%M:%S')}")
                else:
                    print(f"⏹ 停止，{frame_idx} 帧")
                    break
            elif key in (ord('q'), 27):
                if not recording:
                    print("取消")
                    sys.exit(0)
                print(f"中止，{frame_idx} 帧")
                break

    finally:
        cv2.destroyAllWindows()

    if frame_idx == 0:
        print("没录到任何帧")
        sys.exit(0)

    # 时间戳
    with open(os.path.join(args.output, "timestamps.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["id", "timestamp_sec"])
        for t in timestamps:
            w.writerow([t["id"], f"{t['timestamp']:.6f}"])

    # 元数据
    print()
    duration_actual = timestamps[-1]["timestamp"]
    print(f"录制完成: {frame_idx} 帧 / {duration_actual:.1f}s "
          f"({frame_idx/duration_actual:.1f} fps)")
    print()
    print("请填写转盘信息（用于估计每帧角度初值）:")

    while True:
        try:
            n_turns = float(input("  共转了多少圈？(可填小数): ").strip())
            if n_turns > 0:
                break
        except ValueError:
            pass

    s = input(f"  总耗时？(秒，回车=记录值 {duration_actual:.1f}): ").strip()
    total_time = float(s) if s else duration_actual

    metadata = {
        "capture_time": datetime.now().isoformat(),
        "n_frames": frame_idx,
        "fps_target": args.fps,
        "fps_actual": frame_idx / duration_actual,
        "n_turns": n_turns,
        "total_time_sec": total_time,
        "rotation_speed_deg_per_sec": n_turns * 360 / total_time,
        "calibration_file": args.calib,
        "has_aligned": True,
        "kinect_version": "v2",
    }
    with open(os.path.join(args.output, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(f"  转速: {metadata['rotation_speed_deg_per_sec']:.2f} °/秒")
    print(f"  目录: {args.output}")
    print()
    print(f"下一步:")
    print(f"  python （已废弃） --input {args.output} --calib {args.calib}")


if __name__ == "__main__":
    main()
