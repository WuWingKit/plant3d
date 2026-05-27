"""
02c_manual_align_3d.py — 在 3D 点云上手动调外参对齐

不同于 02b（用 2D 边缘叠加），这个工具:
  - 直接显示彩色点云
  - 按键调 R, T，实时重新生成点云
  - 你能旋转/缩放点云从各角度看颜色对齐效果

工作流:
  1. 选一帧已有数据（capture_xxx/depth/XXXX.png + color/XXXX.png）
  2. 按当前外参生成彩色点云显示
  3. 键盘调参（不用 OpenCV 滑块，避免抢焦点）
  4. 调好按 S 保存到 calibration.json

按键:
  W/S       tx ± 1mm   (左右偏移)
  A/D       ty ± 1mm   (上下偏移)
  Q/E       tz ± 1mm   (前后偏移)
  R/F       rx ± 0.1°
  T/G       ry ± 0.1°
  Y/H       rz ± 0.1°
  Shift + 上述键: 步长 ×10
  
  P         打印当前 R, T
  X         保存到 calibration.json
  Z         重置为出厂值
  Esc       退出

用法:
  # 用一帧 capture 数据调
  python 02c_manual_align_3d.py --calib calibration.json --frame capture_xxx/depth/0000.png

  # 或者用 Kinect 现场拍一帧
  python 02c_manual_align_3d.py --calib calibration.json --live
"""

import os
import sys
import json
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
from utils_kinect import (
    load_calib, get_intrinsics, get_stereo, save_calib,
)


def load_frame_from_disk(depth_path, color_path):
    """从磁盘加载一帧 depth + color"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path)
    if depth is None or color is None:
        return None, None
    if depth.dtype != np.uint16:
        print(f"⚠ 深度图不是 uint16 ({depth.dtype})，可能数据损坏")
    return depth, color


def capture_live_frame():
    """从 Kinect 实时抓一帧"""
    from utils_kinect import init_kinect, get_color, get_depth
    print("启动 Kinect...")
    kinect = init_kinect()
    if kinect is None:
        return None, None
    print("等待 Kinect (10秒)...")
    start = time.time()
    while time.time() - start < 10:
        if get_color(kinect) is not None and get_depth(kinect) is not None:
            break
        time.sleep(0.1)
    print("拍一帧（保持场景静止）...")
    input("按 Enter 拍摄...")
    # 多次尝试拿到匹配的一对
    color, depth = None, None
    start = time.time()
    while time.time() - start < 3:
        if color is None:
            color = get_color(kinect)
        if depth is None:
            depth = get_depth(kinect)
        if color is not None and depth is not None:
            break
        time.sleep(0.05)
    return depth, color


def make_pcd(depth, color, K_d, K_c, dist_c, R, T, depth_trunc=3000):
    """用当前 R, T 生成彩色点云（深度坐标系下）"""
    fx, fy = K_d[0, 0], K_d[1, 1]
    cx, cy = K_d[0, 2], K_d[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    mask = (Z > 300) & (Z < depth_trunc)
    if not np.any(mask):
        return None
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)

    # 用 R, T 把 3D 点变到彩色坐标系，再投影到彩色图取颜色
    pts_c = (R @ pts.T + T.reshape(3, 1)).T
    Zc = pts_c[:, 2]
    valid = Zc > 0
    Zc_safe = np.where(valid, Zc, 1)
    xn = pts_c[:, 0] / Zc_safe
    yn = pts_c[:, 1] / Zc_safe

    k1, k2, p1, p2, k3 = (dist_c[0], dist_c[1], dist_c[2], dist_c[3],
                          dist_c[4] if len(dist_c) > 4 else 0)
    r2 = xn*xn + yn*yn
    rad = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = xn * rad + 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
    yd = yn * rad + p1*(r2 + 2*yn*yn) + 2*p2*xn*yn

    u_c = (K_c[0, 0] * xd + K_c[0, 2]).round().astype(np.int32)
    v_c = (K_c[1, 1] * yd + K_c[1, 2]).round().astype(np.int32)
    h_c, w_c = color.shape[:2]
    in_img = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c) & valid

    rgb = np.zeros((len(pts), 3), dtype=np.uint8)
    if np.any(in_img):
        bgr = color[v_c[in_img], u_c[in_img]]
        rgb[in_img] = bgr[:, ::-1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    return pcd


def rotation_from_euler(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def euler_from_rotation(R):
    """从旋转矩阵分解出 XYZ 欧拉角（度）"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0
    return rx, ry, rz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="calibration.json")
    parser.add_argument("--frame", default=None,
                        help="深度图路径，比如 capture_xxx/depth/0000.png（同目录的 color 自动匹配）")
    parser.add_argument("--live", action="store_true",
                        help="不用磁盘文件，现场用 Kinect 拍一帧")
    args = parser.parse_args()

    if not args.frame and not args.live:
        print("✗ 必须指定 --frame 或 --live")
        sys.exit(1)

    # 标定
    calib = load_calib(args.calib)
    K_d, _, _ = get_intrinsics(calib, "depth")
    K_c, dist_c, _ = get_intrinsics(calib, "color")
    R_init, T_init = get_stereo(calib)
    rx0, ry0, rz0 = euler_from_rotation(R_init)
    tx0, ty0, tz0 = T_init[0], T_init[1], T_init[2]

    print("=" * 64)
    print("3D 手动外参对齐")
    print("=" * 64)
    print(f"标定: {args.calib}")
    print(f"初始 T = [{tx0:.2f}, {ty0:.2f}, {tz0:.2f}] mm")
    print(f"初始 R 欧拉角 = [{rx0:.2f}, {ry0:.2f}, {rz0:.2f}] deg")
    print()

    # 加载帧
    if args.live:
        depth, color = capture_live_frame()
    else:
        # 自动匹配 color 路径
        depth_path = args.frame
        color_path = depth_path.replace("depth", "color")
        if not os.path.isfile(color_path):
            # 试试 raw color 路径
            base = os.path.dirname(os.path.dirname(depth_path))
            fname = os.path.basename(depth_path)
            color_path = os.path.join(base, "color", fname)
        if not os.path.isfile(color_path):
            print(f"✗ 找不到对应彩色图: {color_path}")
            sys.exit(1)
        print(f"加载深度: {args.frame}")
        print(f"加载彩色: {color_path}")
        depth, color = load_frame_from_disk(depth_path, color_path)

    if depth is None or color is None:
        print("✗ 帧加载失败")
        sys.exit(1)
    print(f"  深度: {depth.shape} {depth.dtype} median={np.median(depth[depth>0]) if np.any(depth>0) else 0:.0f}mm")
    print(f"  彩色: {color.shape}")
    print()

    # 状态
    state = {
        "tx": float(tx0), "ty": float(ty0), "tz": float(tz0),
        "rx": float(rx0), "ry": float(ry0), "rz": float(rz0),
        "saved": False,
    }

    def get_RT():
        R = rotation_from_euler(state["rx"], state["ry"], state["rz"])
        T = np.array([state["tx"], state["ty"], state["tz"]])
        return R, T

    def print_state(prefix=""):
        print(f"{prefix}T=[{state['tx']:+7.2f}, {state['ty']:+7.2f}, {state['tz']:+7.2f}] mm  "
              f"R=[{state['rx']:+6.2f}, {state['ry']:+6.2f}, {state['rz']:+6.2f}] deg")

    # 初始点云
    R, T = get_RT()
    pcd = make_pcd(depth, color, K_d, K_c, dist_c, R, T)
    if pcd is None:
        print("✗ 点云生成失败")
        sys.exit(1)

    print("=" * 64)
    print("操作（焦点在 Open3D 窗口里时有效）:")
    print()
    print("  平移 (mm):")
    print("    W/S — tx ± 1     (左右)")
    print("    A/D — ty ± 1     (上下)")
    print("    Q/E — tz ± 1     (前后)")
    print()
    print("  旋转 (度):")
    print("    R/F — rx ± 0.1   (绕 X)")
    print("    T/G — ry ± 0.1   (绕 Y)")
    print("    Y/H — rz ± 0.1   (绕 Z)")
    print()
    print("    步长 ×10: Shift + 上述键（即用大写字母）")
    print()
    print("  其他:")
    print("    P — 打印当前 R, T")
    print("    X — 保存到 calibration.json")
    print("    Z — 重置为出厂值")
    print("    鼠标拖动 — 旋转点云视角")
    print("    滚轮 — 缩放")
    print("    Esc — 退出")
    print("=" * 64)
    print()
    print_state("初始: ")

    # 用 VisualizerWithKeyCallback 注册按键回调
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Manual 3D Align", width=1280, height=720)
    vis.add_geometry(pcd)

    # 设置背景为黑色（彩色点云更醒目）
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 2.0

    def update_pcd():
        R, T = get_RT()
        new_pcd = make_pcd(depth, color, K_d, K_c, dist_c, R, T)
        if new_pcd is None:
            return
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        print_state()

    # 注册按键
    def cb_factory(field, delta_small, delta_big):
        def cb(vis):
            # GLFW 没法直接检测 Shift；用大小写字母替代
            state[field] += delta_small
            update_pcd()
            return False
        return cb

    def cb_factory_big(field, delta_big):
        def cb(vis):
            state[field] += delta_big
            update_pcd()
            return False
        return cb

    # 小步：小写字母（GLFW 把所有键当大写传，但我们这里只用 ord('大写字母')）
    # Open3D 的 register_key_callback 接收 GLFW 键码，全部用大写字母对应的 ASCII
    # 我们这里小写当 ±1，大写（Shift+）当 ±10，但 GLFW 区分不开 ——
    # 改用不同按键映射：

    # 平移
    vis.register_key_callback(ord('W'), cb_factory("tx", -1, -10))   # tx-
    vis.register_key_callback(ord('S'), cb_factory("tx", +1, +10))   # tx+
    vis.register_key_callback(ord('A'), cb_factory("ty", -1, -10))   # ty-
    vis.register_key_callback(ord('D'), cb_factory("ty", +1, +10))   # ty+
    vis.register_key_callback(ord('Q'), cb_factory("tz", -1, -10))   # tz-
    vis.register_key_callback(ord('E'), cb_factory("tz", +1, +10))   # tz+

    # 旋转
    vis.register_key_callback(ord('R'), cb_factory("rx", -0.1, -1.0))
    vis.register_key_callback(ord('F'), cb_factory("rx", +0.1, +1.0))
    vis.register_key_callback(ord('T'), cb_factory("ry", -0.1, -1.0))
    vis.register_key_callback(ord('G'), cb_factory("ry", +0.1, +1.0))
    vis.register_key_callback(ord('Y'), cb_factory("rz", -0.1, -1.0))
    vis.register_key_callback(ord('H'), cb_factory("rz", +0.1, +1.0))

    # 大步（数字键 1-6 配对）
    # 1/2: tx ±10  3/4: ty ±10  5/6: tz ±10
    vis.register_key_callback(ord('1'), cb_factory_big("tx", -10))
    vis.register_key_callback(ord('2'), cb_factory_big("tx", +10))
    vis.register_key_callback(ord('3'), cb_factory_big("ty", -10))
    vis.register_key_callback(ord('4'), cb_factory_big("ty", +10))
    vis.register_key_callback(ord('5'), cb_factory_big("tz", -10))
    vis.register_key_callback(ord('6'), cb_factory_big("tz", +10))

    # 打印
    def cb_print(vis):
        print_state(">>> ")
        return False
    vis.register_key_callback(ord('P'), cb_print)

    # 重置
    def cb_reset(vis):
        state["tx"], state["ty"], state["tz"] = float(tx0), float(ty0), float(tz0)
        state["rx"], state["ry"], state["rz"] = float(rx0), float(ry0), float(rz0)
        update_pcd()
        print("(重置)")
        return False
    vis.register_key_callback(ord('Z'), cb_reset)

    # 保存
    def cb_save(vis):
        R, T = get_RT()
        calib["stereo_depth_to_color"]["R_depth_to_color"] = R.tolist()
        calib["stereo_depth_to_color"]["T_depth_to_color"] = T.ravel().tolist()
        calib["stereo_depth_to_color"]["note"] = "Manually aligned (02c_manual_align_3d)"
        calib["stereo_depth_to_color"]["manual_align_rxyz_deg"] = [state["rx"], state["ry"], state["rz"]]
        save_calib(args.calib, calib)
        state["saved"] = True
        print()
        print(f"✓ 已保存到 {args.calib}")
        print_state("最终: ")
        print()
        return False
    vis.register_key_callback(ord('X'), cb_save)

    print()
    print("窗口已打开，请把焦点放在 Open3D 窗口（点一下它）后按键。")
    print()

    vis.run()
    vis.destroy_window()

    if not state["saved"]:
        print()
        print("(未保存退出)")
    else:
        print("下一步:")
        print("  重新跑 Stage 1 看效果:")
        print(f"  python 04_stage1_pcd.py --input capture_xxx --calib {args.calib}")


if __name__ == "__main__":
    main()
