"""
tool_diagnose_depth.py — 紧急诊断脚本：检查 depth 目录里的图是不是 16 位深度

用法：
  python tool_diagnose_depth.py --dir capture_xxx/depth

输出：
  - 检查前 10 张图的位深、通道、数值范围
  - 判断是否是合法的 Kinect v2 深度图
  - 如果不是，给出补救建议
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2


def diagnose_one(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    info = {
        "path": os.path.basename(path),
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": int(img.min()),
        "max": int(img.max()),
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "is_uint16": img.dtype == np.uint16,
    }

    if img.dtype == np.uint16:
        valid = img[img > 0]
        if len(valid) > 0:
            info["valid_min_mm"] = int(valid.min())
            info["valid_max_mm"] = int(valid.max())
            info["valid_median_mm"] = int(np.median(valid))
            info["valid_ratio"] = float(np.count_nonzero(img) / img.size)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="depth 目录路径")
    parser.add_argument("--n", type=int, default=10, help="检查前 N 张")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.png")))
    if not files:
        print(f"✗ 目录 {args.dir} 找不到 PNG 文件")
        sys.exit(1)

    print(f"目录: {args.dir}")
    print(f"PNG 文件总数: {len(files)}")
    print(f"检查前 {min(args.n, len(files))} 张：\n")

    results = []
    for f in files[:args.n]:
        info = diagnose_one(f)
        if info is None:
            print(f"  ✗ {os.path.basename(f)} 读取失败")
            continue
        results.append(info)

        flag = "✓" if info["is_uint16"] else "✗"
        ch = info["channels"]
        shape_str = f"{info['shape'][1]}x{info['shape'][0]}" + (f" x{ch}ch" if ch > 1 else "")
        print(f"  {flag} {info['path']}: {info['dtype']} {shape_str} "
              f"range [{info['min']}, {info['max']}]")

        if info["is_uint16"] and "valid_median_mm" in info:
            print(f"      有效深度: 中位 {info['valid_median_mm']}mm, "
                  f"范围 [{info['valid_min_mm']}, {info['valid_max_mm']}] mm, "
                  f"覆盖 {info['valid_ratio']*100:.0f}%")

    # 总结
    print("\n" + "=" * 60)
    n_uint16 = sum(1 for r in results if r["is_uint16"])
    n_8bit_3ch = sum(1 for r in results if r["dtype"] == "uint8" and r["channels"] == 3)
    n_8bit_1ch = sum(1 for r in results if r["dtype"] == "uint8" and r["channels"] == 1)

    if n_uint16 == len(results):
        print("✓ 所有图都是合法的 16 位深度图，可以正常使用")
        print("\n  典型 Kinect v2 深度范围 800-4500mm")
        median_mm = np.median([r["valid_median_mm"] for r in results if "valid_median_mm" in r])
        print(f"  你的数据中位深度: {median_mm:.0f}mm")
        if median_mm < 500:
            print("  ⚠ 中位深度过近，可能花瓶离相机太近或数据异常")
        elif median_mm > 3000:
            print("  ⚠ 中位深度过远，可能包含太多背景，建议 --depth-trunc 调小")

    elif n_8bit_3ch == len(results):
        print("🚨 严重问题：所有图都是 8 位 3 通道（BGR），不是真正的深度图！")
        print()
        print("可能原因：02_capture.py 保存时用了：")
        print("    cv2.imwrite('depth/xxx.png', depth)  # 错，会按彩色保存")
        print("应该用：")
        print("    cv2.imwrite('depth/xxx.png', depth.astype(np.uint16))  # 对")
        print()
        print("==== 后果 ====")
        print("16 位深度数据被截断丢失，已采集的 720 帧深度信息基本不可恢复。")
        print()
        print("==== 唯一补救途径 ====")
        print("用相同设置重新采集一次（这次用修复版 02_capture.py）。")
        print("重新采集前先验证 capture 第一帧的深度是 16 位。")

    else:
        print(f"⚠ 混合格式：")
        print(f"  16 位深度图: {n_uint16}")
        print(f"  8 位 3 通道: {n_8bit_3ch}")
        print(f"  8 位 1 通道: {n_8bit_1ch}")


if __name__ == "__main__":
    main()
