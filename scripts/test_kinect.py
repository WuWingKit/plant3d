"""
test_kinect.py — 最小化诊断：Kinect 能否拿到数据？OpenCV 窗口能否显示？

按顺序测三件事：
  1. PyKinect2 能不能 import
  2. Kinect 能不能初始化
  3. 能不能拿到几帧数据（看数据是不是真的）
  4. OpenCV 窗口能不能弹出来

每一步都打印结果到控制台。
"""

import sys
import time
import numpy as np

print("=" * 60)
print("Kinect v2 最小化诊断")
print("=" * 60)

# ---- Test 1: import ----
print("\n[1/4] 检查 PyKinect2...")
try:
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime
    print("  ✓ PyKinect2 import 成功")
except ImportError as e:
    print(f"  ✗ PyKinect2 import 失败: {e}")
    print("  请先 pip install pykinect2 comtypes")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ 其他错误: {e}")
    print("  可能 PyKinect2 没打 64 位补丁，看 README.md 末尾的补丁说明")
    sys.exit(1)

# ---- Test 2: OpenCV ----
print("\n[2/4] 检查 OpenCV...")
try:
    import cv2
    print(f"  ✓ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    print(f"  ✗ OpenCV import 失败: {e}")
    sys.exit(1)

# ---- Test 3: 初始化 + 取帧 ----
print("\n[3/4] 初始化 Kinect 并尝试取帧...")
try:
    flags = (PyKinectV2.FrameSourceTypes_Color |
             PyKinectV2.FrameSourceTypes_Depth |
             PyKinectV2.FrameSourceTypes_Infrared)
    kinect = PyKinectRuntime.PyKinectRuntime(flags)
    print(f"  ✓ Kinect 初始化成功")
except Exception as e:
    print(f"  ✗ Kinect 初始化失败: {e}")
    sys.exit(1)

print("  等待数据（最多 5 秒）...")
got_color = got_depth = got_ir = False
start = time.time()
while time.time() - start < 5.0:
    if not got_color and kinect.has_new_color_frame():
        c = kinect.get_last_color_frame()
        got_color = c is not None and len(c) > 0
    if not got_depth and kinect.has_new_depth_frame():
        d = kinect.get_last_depth_frame()
        got_depth = d is not None and len(d) > 0
    if not got_ir and kinect.has_new_infrared_frame():
        i = kinect.get_last_infrared_frame()
        got_ir = i is not None and len(i) > 0
    if got_color and got_depth and got_ir:
        break
    time.sleep(0.05)

print(f"  Color: {'✓' if got_color else '✗'}")
print(f"  Depth: {'✓' if got_depth else '✗'}")
print(f"  IR:    {'✓' if got_ir else '✗'}")

if not (got_color and got_depth):
    print("\n  ⚠ Kinect 没拿到数据。可能原因：")
    print("    - USB 不是 3.0（必须蓝色口）")
    print("    - Kinect 没插电源")
    print("    - 其他程序占用了 Kinect（关闭 Kinect Studio 等）")
    sys.exit(1)

# 看一下数据范围
color_raw = kinect.get_last_color_frame()
color = color_raw.reshape((1080, 1920, 4)).astype(np.uint8)
depth_raw = kinect.get_last_depth_frame()
depth = depth_raw.reshape((424, 512)).astype(np.uint16)
ir_raw = kinect.get_last_infrared_frame()
ir = ir_raw.reshape((424, 512)).astype(np.uint16)

print(f"\n  Color shape={color.shape} mean={color.mean():.0f}")
print(f"  Depth shape={depth.shape} valid_ratio={np.count_nonzero(depth)/depth.size*100:.0f}% "
      f"median_mm={np.median(depth[depth>0]) if np.any(depth>0) else 0:.0f}")
print(f"  IR    shape={ir.shape} mean={ir.mean():.0f} max={ir.max()}")

# ---- Test 4: OpenCV 窗口 ----
print("\n[4/4] 测试 OpenCV 窗口（应该弹出一个 800x450 的窗口）...")
print("  窗口标题: 'Kinect Test'")
print("  按任意键关闭窗口")
print()

try:
    cv2.namedWindow("Kinect Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kinect Test", 800, 450)
    print("  ✓ 窗口创建成功")
except Exception as e:
    print(f"  ✗ 窗口创建失败: {e}")
    sys.exit(1)

# 转换显示
color_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
color_small = cv2.resize(color_bgr, (640, 360))

depth_vis = np.clip(depth.astype(np.float32) / 3000 * 255, 0, 255).astype(np.uint8)
depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
depth_vis = cv2.resize(depth_vis, (640, 360))

# 拼接
canvas = np.zeros((400, 800, 3), dtype=np.uint8)
canvas[20:380, 0:640] = cv2.resize(color_bgr, (640, 360))

cv2.putText(canvas, "Press any key to close", (10, 395),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Kinect Test", canvas)
print("  窗口已显示，等待按键...")
key = cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\n  ✓ 按键 {key}，窗口已关闭")
print()
print("=" * 60)
print("✓ 全部测试通过！可以跑 01_capture_calib.py 了")
print("=" * 60)
