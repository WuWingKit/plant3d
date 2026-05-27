"""
utils_kinect.py — Kinect v2 共用工具

包含：
  - PyKinect2 初始化和同步帧获取
  - IR 图像预处理（散斑环境优化）
  - 棋盘检测（同时支持 SB 和经典算法）
  - 标定文件加载和保存
  - 自定义对齐函数（深度像素 → 彩色坐标系）
"""

import os
import sys
import json
import time
import numpy as np
import cv2


# ============================================================
#  PyKinect2 初始化
# ============================================================

try:
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime
    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False


def init_kinect(need_color=True, need_depth=True, need_ir=True):
    """初始化 Kinect runtime。返回 PyKinectRuntime 实例 或 None"""
    if not KINECT_AVAILABLE:
        print("✗ PyKinect2 未安装")
        print("  pip install pykinect2 comtypes")
        print("  并确保 PyKinect2 打了 64 位 Python 补丁")
        return None
    flags = 0
    if need_color: flags |= PyKinectV2.FrameSourceTypes_Color
    if need_depth: flags |= PyKinectV2.FrameSourceTypes_Depth
    if need_ir:    flags |= PyKinectV2.FrameSourceTypes_Infrared
    try:
        return PyKinectRuntime.PyKinectRuntime(flags)
    except Exception as e:
        print(f"✗ Kinect 初始化失败: {e}")
        return None


def get_color(kinect):
    """1080x1920x3 BGR uint8 或 None"""
    if not kinect.has_new_color_frame():
        return None
    raw = kinect.get_last_color_frame()
    img = raw.reshape((1080, 1920, 4)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def get_depth(kinect):
    """424x512 uint16 mm 或 None"""
    if not kinect.has_new_depth_frame():
        return None
    raw = kinect.get_last_depth_frame()
    return raw.reshape((424, 512)).astype(np.uint16)


def get_ir(kinect):
    """424x512 uint16 IR 强度 或 None"""
    if not kinect.has_new_infrared_frame():
        return None
    raw = kinect.get_last_infrared_frame()
    return raw.reshape((424, 512)).astype(np.uint16)


def wait_synced(kinect, max_wait=2.0, need_color=True, need_depth=True, need_ir=True):
    """
    等待所有需要的帧都有新数据再返回。
    PyKinect2 各通道独立异步更新，必须主动等到同一时刻所有通道都更新过。
    """
    start = time.time()
    color, depth, ir = None, None, None
    while time.time() - start < max_wait:
        if need_color and color is None:
            color = get_color(kinect)
        if need_depth and depth is None:
            depth = get_depth(kinect)
        if need_ir and ir is None:
            ir = get_ir(kinect)
        ok = ((not need_color or color is not None) and
              (not need_depth or depth is not None) and
              (not need_ir or ir is not None))
        if ok:
            return color, depth, ir
        time.sleep(0.005)
    return color, depth, ir  # 超时也返回已拿到的


# ============================================================
#  IR 图像预处理（针对散斑环境）
# ============================================================

def ir_to_8bit(ir_uint16, mode="clahe", clip=2.0, tile=8):
    """
    IR 16-bit → 8-bit 灰度图。

    mode:
      'linear'    - 简单线性归一化
      'percent'   - 百分位裁剪后归一化（防过曝）
      'clahe'     - CLAHE 自适应直方图均衡（适合显示）
      'denoised'  - CLAHE + 中值滤波（最适合棋盘角点检测，去散斑）
    """
    if mode == "linear":
        return cv2.normalize(ir_uint16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mode == "percent":
        lo = np.percentile(ir_uint16, 1)
        hi = np.percentile(ir_uint16, 99)
        if hi <= lo:
            hi = lo + 1
        clipped = np.clip(ir_uint16, lo, hi)
        return ((clipped - lo) * 255 / (hi - lo)).astype(np.uint8)

    if mode in ("clahe", "denoised"):
        # 先线性压缩到 8 位（用百分位防过曝）
        lo = np.percentile(ir_uint16, 1)
        hi = np.percentile(ir_uint16, 99.5)
        if hi <= lo:
            hi = lo + 1
        scaled = np.clip((ir_uint16 - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        enhanced = clahe.apply(scaled)

        if mode == "denoised":
            # 中值滤波 3x3 去散斑（散斑是椒盐噪声，中值滤波是首选）
            # 然后双边滤波（保留棋格边缘的同时进一步平滑）
            enhanced = cv2.medianBlur(enhanced, 3)
            enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)

        return enhanced

    raise ValueError(f"unknown mode {mode}")


def ir_quality_score(ir_uint16):
    """
    评估 IR 图质量。返回 dict 含几个指标。
    用于采集时实时反馈"现在能不能拍棋盘"。
    """
    valid = ir_uint16[ir_uint16 > 0]
    if len(valid) < 100:
        return {"ok": False, "reason": "几乎全黑（光线太暗或 Kinect 故障）"}

    mean = float(valid.mean())
    median = float(np.median(valid))
    # IR 16-bit 通常 0-65535，正常室内场景 median 在 200-2000
    if median < 50:
        return {"ok": False, "reason": f"图像太暗 median={median:.0f}（增加照明）",
                "mean": mean, "median": median}
    if median > 30000:
        return {"ok": False, "reason": f"图像过曝 median={median:.0f}（减少照明/反光）",
                "mean": mean, "median": median}

    # 估计对比度：在 8-bit 化的图上算标准差
    img8 = ir_to_8bit(ir_uint16, mode="clahe")
    contrast = float(img8.std())
    if contrast < 25:
        return {"ok": False, "reason": f"对比度太低 std={contrast:.1f}（光线或棋盘问题）",
                "mean": mean, "median": median, "contrast": contrast}

    return {"ok": True, "mean": mean, "median": median, "contrast": contrast}


# ============================================================
#  棋盘检测
# ============================================================

def find_chessboard(img_gray, pattern_size, use_sb=True, refine=True):
    """
    检测棋盘角点。返回 (ret, corners) 或 (False, None)。

    use_sb: True 用 OpenCV 4.0+ 的 SB 算法（散斑场景下显著鲁棒）
            False 用经典 findChessboardCorners
    refine: 是否用 cornerSubPix 做亚像素精化（SB 自带，仅经典模式需要）
    """
    if use_sb and hasattr(cv2, 'findChessboardCornersSB'):
        # SB 自带亚像素精度，不需要再精化
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, 'CALIB_CB_ACCURACY'):
            flags |= cv2.CALIB_CB_ACCURACY
        ret, corners = cv2.findChessboardCornersSB(img_gray, pattern_size, flags=flags)
        return ret, corners

    # 经典算法
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(img_gray, pattern_size, flags=flags)
    if ret and refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


# ============================================================
#  标定文件
# ============================================================

def load_calib(path):
    """加载 calibration.json，返回 dict"""
    with open(path) as f:
        return json.load(f)


def save_calib(path, calib_dict):
    """保存 calibration.json"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(calib_dict, f, indent=2, ensure_ascii=False)


def get_intrinsics(calib_dict, which="color"):
    """从 calib dict 取出 K, dist, size。which='color' or 'depth'"""
    cam = calib_dict[which]
    K = np.array(cam["K"], dtype=np.float64)
    dist = np.array(cam.get("dist", [0.0]*5), dtype=np.float64)
    size = tuple(cam["image_size"])
    return K, dist, size


def get_stereo(calib_dict):
    """取出 IR→RGB 的 R, T"""
    s = calib_dict.get("stereo_depth_to_color", {})
    R = np.array(s.get("R_depth_to_color", np.eye(3).tolist()), dtype=np.float64)
    T = np.array(s.get("T_depth_to_color", [-52.0, 0.0, 0.0]), dtype=np.float64)
    return R, T


# ============================================================
#  深度 → 3D 点云
# ============================================================

def depth_to_xyz(depth_mm, K_depth):
    """
    深度图反投影为 3D 点云（深度相机坐标系，单位 mm）。
    返回 (xyz, mask)：
      xyz  - (H, W, 3) 每像素的 (X, Y, Z)，Z=0 表示无效
      mask - (H, W) bool，True 表示该像素有有效深度
    """
    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_mm.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    xyz = np.stack([X, Y, Z], axis=-1)
    mask = Z > 0
    return xyz, mask


# ============================================================
#  对齐：深度像素 → 彩色像素（自定义实现，用你标定的参数）
# ============================================================

def make_aligned_color(depth_mm, color_bgr, K_depth, K_color, dist_color, R_d2c, T_d2c):
    """
    把彩色图 "映射" 到深度图分辨率，使每个像素直接对应。

    输入：
      depth_mm:     (424, 512) uint16
      color_bgr:    (1080, 1920, 3) uint8
      K_depth:      深度相机内参
      K_color:      彩色相机内参
      dist_color:   彩色相机畸变系数
      R_d2c, T_d2c: 深度→彩色 的外参

    返回：
      aligned_color: (424, 512, 3) uint8，对齐到深度图分辨率
        - 每个像素 [v, u] 的颜色 = 深度图 [v, u] 对应 3D 点投影到彩色图取的颜色
        - 深度无效的像素 = (0, 0, 0)
    """
    h_d, w_d = depth_mm.shape
    h_c, w_c = color_bgr.shape[:2]

    # 1. 深度像素 → 3D 点（深度相机坐标系）
    xyz_d, mask = depth_to_xyz(depth_mm, K_depth)  # (H, W, 3), (H, W)

    # 2. 深度坐标系 → 彩色坐标系
    pts_d = xyz_d.reshape(-1, 3).T  # (3, N)
    pts_c = R_d2c @ pts_d + T_d2c.reshape(3, 1)  # (3, N)

    # 3. 投影到彩色图（带畸变）
    Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]
    valid_z = Zc > 0
    Zc_safe = np.where(valid_z, Zc, 1)
    xn = Xc / Zc_safe
    yn = Yc / Zc_safe

    k1, k2, p1, p2, k3 = (dist_color[0], dist_color[1],
                          dist_color[2], dist_color[3],
                          dist_color[4] if len(dist_color) > 4 else 0.0)
    r2 = xn*xn + yn*yn
    rad = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = xn * rad + 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
    yd = yn * rad + p1*(r2 + 2*yn*yn) + 2*p2*xn*yn

    u_c = K_color[0, 0] * xd + K_color[0, 2]
    v_c = K_color[1, 1] * yd + K_color[1, 2]

    # 4. 取颜色（用线性采样会更平滑，这里用最近邻够用）
    u_int = np.round(u_c).astype(np.int32)
    v_int = np.round(v_c).astype(np.int32)
    in_img = (u_int >= 0) & (u_int < w_c) & (v_int >= 0) & (v_int < h_c) & valid_z & mask.ravel()

    aligned = np.zeros((h_d * w_d, 3), dtype=np.uint8)
    if np.any(in_img):
        aligned[in_img] = color_bgr[v_int[in_img], u_int[in_img]]
    return aligned.reshape(h_d, w_d, 3)


# ============================================================
#  采集预览：构建多面板显示
# ============================================================

def make_preview(color, depth, ir, info_lines, w=1280, h=720):
    """
    创建一个 1280x720 的预览画面，包含彩色/深度/IR 三个面板 + 状态信息。
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 左上：彩色（缩到 640x360）
    if color is not None:
        c_small = cv2.resize(color, (640, 360))
        canvas[0:360, 0:640] = c_small
        cv2.putText(canvas, "Color (RGB)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 右上：IR 可视化（512x424 → 缩到 640x424 但裁掉一点）
    if ir is not None:
        ir_vis = ir_to_8bit(ir, mode="clahe")
        ir_vis = cv2.cvtColor(ir_vis, cv2.COLOR_GRAY2BGR)
        ir_vis = cv2.resize(ir_vis, (640, 360))
        canvas[0:360, 640:1280] = ir_vis
        cv2.putText(canvas, "Infrared (CLAHE)", (650, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 左下：深度伪彩色
    if depth is not None:
        d_vis = np.clip(depth.astype(np.float32) / 4500 * 255, 0, 255).astype(np.uint8)
        d_vis[depth == 0] = 0
        d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
        d_vis = cv2.resize(d_vis, (640, 360))
        canvas[360:720, 0:640] = d_vis
        cv2.putText(canvas, "Depth (JET)", (10, 385),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 右下：信息文本
    info_x, info_y = 660, 400
    for i, line in enumerate(info_lines):
        color_text = (0, 255, 255)
        if line.startswith("✓"):
            color_text = (0, 255, 0)
        elif line.startswith("✗"):
            color_text = (0, 0, 255)
        elif line.startswith("⚠"):
            color_text = (0, 200, 255)
        cv2.putText(canvas, line, (info_x, info_y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_text, 1)

    return canvas
