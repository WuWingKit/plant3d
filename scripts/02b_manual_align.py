"""
02b_manual_align.py — 手动对齐 RGB 和 Depth 相机的外参

原理：
  Kinect SDK 出厂值给的 R, T 通常有 1-2cm 误差。
  通过实时调 6 个滑块（tx, ty, tz, rx, ry, rz）观察对齐效果，
  在画面中找到颜色和深度边缘"贴合"的位置。

工作模式：
  默认模式：什么都不调，直接保存出厂值（你之前 02_calibrate 输出的就是出厂值）
  交互模式：弹出窗口实时调

操作：
  1. 启动 → 自动拍一帧（按 R 重拍）
  2. 看预览窗口（左：彩色 + 深度边缘叠加 | 右：当前 R/T）
  3. 调滑块：
     - tx/ty/tz: 平移 (mm)，调到深度边缘和彩色物体边缘对齐
     - rx/ry/rz: 旋转 (0.01° 一档)，通常不用调（出厂 R 接近单位矩阵）
  4. 满意后按 S 保存
  5. Q/Esc 不保存退出

使用：
  # 默认（不调，直接用出厂值，已经在 calibration.json 里）
  python 02b_manual_align.py --calib calibration.json --skip

  # 交互调
  python 02b_manual_align.py --calib calibration.json
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kinect import (
    init_kinect, get_color, get_depth,
    load_calib, get_intrinsics, get_stereo, save_calib,
    make_aligned_color,
)


def take_snapshot(kinect, max_wait=5.0):
    """拍一帧 color + depth"""
    print("  拍摄中（请保持场景静止）...")
    start = time.time()
    color, depth = None, None
    while time.time() - start < max_wait:
        if color is None:
            color = get_color(kinect)
        if depth is None:
            depth = get_depth(kinect)
        if color is not None and depth is not None:
            return color, depth
        time.sleep(0.05)
    return color, depth


def make_overlay(depth, aligned_color, alpha=0.5):
    """
    把对齐 RGB 和深度伪彩色半透明混合显示。
    用深度区域轮廓（非 Canny 边缘）标出物体边界，避免边缘鬼影。
    """
    # 深度伪彩色
    depth_8 = np.clip(depth.astype(np.float32) / 3000 * 255, 0, 255).astype(np.uint8)
    depth_8[depth == 0] = 0
    depth_color = cv2.applyColorMap(depth_8, cv2.COLORMAP_JET)

    # 半透明混合：对齐 RGB + 深度伪彩色
    blended = cv2.addWeighted(aligned_color, 1 - alpha, depth_color, alpha, 0)

    # 深度有效区域轮廓（只画外边界，不用 Canny）
    valid = (depth > 0).astype(np.uint8)
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 1)  # 绿色细线

    return blended


def rotation_from_euler(rx_deg, ry_deg, rz_deg):
    """欧拉角 (XYZ 顺序，单位度) → 旋转矩阵"""
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="calibration.json")
    parser.add_argument("--skip", action="store_true",
                        help="不调，仅打印当前外参就退出")
    args = parser.parse_args()

    # 加载标定
    calib = load_calib(args.calib)
    K_d, _, _ = get_intrinsics(calib, "depth")
    K_c, dist_c, _ = get_intrinsics(calib, "color")
    R_init, T_init = get_stereo(calib)

    print("=" * 64)
    print("手动外参对齐工具")
    print("=" * 64)
    print(f"标定文件: {args.calib}")
    print(f"  RGB K: fx={K_c[0,0]:.2f}, fy={K_c[1,1]:.2f}, cx={K_c[0,2]:.2f}, cy={K_c[1,2]:.2f}")
    print(f"  当前 R (IR→RGB):")
    for row in R_init:
        print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
    print(f"  当前 T (mm): [{T_init[0]:.2f}, {T_init[1]:.2f}, {T_init[2]:.2f}]")
    print()

    if args.skip:
        print("--skip 模式：直接打印当前值，不修改。")
        print("如果你想交互调整，去掉 --skip 重新跑。")
        return

    # 启动 Kinect
    print("启动 Kinect...")
    kinect = init_kinect()
    if kinect is None:
        sys.exit(1)

    # 等准备好
    print("等待 Kinect 准备 (最多 10 秒)...")
    start = time.time()
    while time.time() - start < 10:
        if get_color(kinect) is not None and get_depth(kinect) is not None:
            break
        time.sleep(0.1)
    else:
        print("✗ 拿不到数据")
        sys.exit(1)
    print("  ✓ OK\n")

    # 拍第一帧
    print("拍摄初始帧（请保持场景静止，建议对着花瓶+背景或棋盘）")
    input("按 Enter 拍摄...")
    color, depth = take_snapshot(kinect)
    if color is None or depth is None:
        print("✗ 拍摄失败")
        sys.exit(1)

    print(f"  ✓ Color: {color.shape}, Depth: {depth.shape}")
    print(f"  Depth valid: {np.count_nonzero(depth)/depth.size*100:.0f}%, "
          f"median: {np.median(depth[depth>0]) if np.any(depth>0) else 0:.0f}mm")
    print()

    # 从初始 R 解出欧拉角（以便从滑块编辑）
    # 简化：假设 R 接近单位阵，分解为 ZYX 欧拉角
    sy = np.sqrt(R_init[0, 0]**2 + R_init[1, 0]**2)
    if sy > 1e-6:
        rx_init = np.degrees(np.arctan2(R_init[2, 1], R_init[2, 2]))
        ry_init = np.degrees(np.arctan2(-R_init[2, 0], sy))
        rz_init = np.degrees(np.arctan2(R_init[1, 0], R_init[0, 0]))
    else:
        rx_init = np.degrees(np.arctan2(-R_init[1, 2], R_init[1, 1]))
        ry_init = np.degrees(np.arctan2(-R_init[2, 0], sy))
        rz_init = 0

    # 滑块状态（用 dict 因为闭包要修改）
    # 滑块都是 int，所以做缩放：
    #   tx, ty, tz: 滑块值 = 实际mm + 100 (允许 -100 到 +100，初始 ~52)
    #   rx, ry, rz: 滑块值 = 实际度*100 + 500 (允许 -5° 到 +5°)
    state = {
        "tx": int(T_init[0]) + 100,
        "ty": int(T_init[1]) + 100,
        "tz": int(T_init[2]) + 100,
        "rx": int(rx_init * 100) + 500,
        "ry": int(ry_init * 100) + 500,
        "rz": int(rz_init * 100) + 500,
    }

    win = "Manual Align"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    def on_trackbar(name):
        def cb(val):
            state[name] = val
        return cb

    # 创建滑块
    cv2.createTrackbar("tx (mm) -100", win, state["tx"], 200, on_trackbar("tx"))
    cv2.createTrackbar("ty (mm) -100", win, state["ty"], 200, on_trackbar("ty"))
    cv2.createTrackbar("tz (mm) -100", win, state["tz"], 200, on_trackbar("tz"))
    cv2.createTrackbar("rx (deg*100) -5", win, state["rx"], 1000, on_trackbar("rx"))
    cv2.createTrackbar("ry (deg*100) -5", win, state["ry"], 1000, on_trackbar("ry"))
    cv2.createTrackbar("rz (deg*100) -5", win, state["rz"], 1000, on_trackbar("rz"))

    print()
    print("=" * 64)
    print("操作:")
    print("  调滑块直到深度边缘（黄色）和彩色物体边缘重合")
    print("  R - 重拍一帧（如果场景变了）")
    print("  S - 保存当前 R, T 到 calibration.json")
    print("  Q/Esc - 不保存退出")
    print()
    print("调参建议:")
    print("  - 主要调 tx（左右偏移），通常 -52 ± 3 mm")
    print("  - ty/tz 一般微调即可")
    print("  - 旋转滑块通常不用动（Kinect 两相机几乎平行）")
    print("=" * 64)
    print()

    try:
        while True:
            # 从滑块取当前值
            tx = state["tx"] - 100
            ty = state["ty"] - 100
            tz = state["tz"] - 100
            rx = (state["rx"] - 500) / 100.0
            ry = (state["ry"] - 500) / 100.0
            rz = (state["rz"] - 500) / 100.0

            R = rotation_from_euler(rx, ry, rz)
            T = np.array([tx, ty, tz], dtype=np.float64)

            # 算对齐 RGB
            try:
                aligned = make_aligned_color(depth, color, K_d, K_c, dist_c, R, T)
            except Exception as e:
                aligned = np.zeros((424, 512, 3), dtype=np.uint8)

            # 叠加深度边缘
            overlay = make_overlay(depth, aligned)

            # 拼接画面
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

            # 左：对齐叠加（放大到 768x636 显示）
            disp_overlay = cv2.resize(overlay, (768, 636), interpolation=cv2.INTER_NEAREST)
            canvas[20:656, 20:788] = disp_overlay
            cv2.putText(canvas, "Aligned RGB + Depth Edges (yellow)", (20, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 右：彩色 + 深度（对照）
            color_small = cv2.resize(color, (456, 257))
            canvas[20:277, 808:1264] = color_small
            cv2.putText(canvas, "Original Color", (808, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            depth_vis = np.clip(depth.astype(np.float32) / 3000 * 255, 0, 255).astype(np.uint8)
            depth_vis[depth == 0] = 0
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_small = cv2.resize(depth_vis, (456, 378))
            canvas[290:668, 808:1264] = depth_small
            cv2.putText(canvas, "Depth (JET)", (808, 285),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 底部信息
            info1 = f"T = [{tx:+5d}, {ty:+5d}, {tz:+5d}] mm    R = [{rx:+5.2f}, {ry:+5.2f}, {rz:+5.2f}] deg"
            info2 = "[S] save  [R] reshoot  [Q/Esc] quit"
            cv2.rectangle(canvas, (0, 680), (1280, 720), (30, 30, 30), -1)
            cv2.putText(canvas, info1, (20, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, info2, (820, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                # 保存
                calib["stereo_depth_to_color"]["R_depth_to_color"] = R.tolist()
                calib["stereo_depth_to_color"]["T_depth_to_color"] = T.tolist()
                calib["stereo_depth_to_color"]["note"] = "Manually aligned"
                calib["stereo_depth_to_color"]["manual_align_rxyz_deg"] = [rx, ry, rz]
                save_calib(args.calib, calib)
                print()
                print(f"✓ 已保存到 {args.calib}")
                print(f"  T = [{tx:+5d}, {ty:+5d}, {tz:+5d}] mm")
                print(f"  R = [{rx:+5.2f}, {ry:+5.2f}, {rz:+5.2f}] deg")
                break

            elif key == ord('r'):
                print()
                print("重拍...")
                input("保持场景静止，按 Enter 拍摄...")
                c2, d2 = take_snapshot(kinect)
                if c2 is not None and d2 is not None:
                    color, depth = c2, d2
                    print(f"  ✓ 重拍完成")
                else:
                    print(f"  ✗ 重拍失败，沿用旧帧")

            elif key in (ord('q'), 27):
                print()
                print("退出（未保存）")
                break

    finally:
        cv2.destroyAllWindows()

    print()
    print("下一步:")
    print(f"  python 03_capture_scan.py --output capture_xxx --calib {args.calib}")


if __name__ == "__main__":
    main()
