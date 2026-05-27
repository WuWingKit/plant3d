# 盆栽 3D 扫描重建

使用 Kinect v2 深度相机 + 电动转盘拍摄盆栽，自动生成 360° 3D 模型。

## 流水线 (5 Stages)

| Stage | 脚本 | 功能 |
|-------|------|------|
| 0 | `01_capture_calib.py` → `02_calibrate.py` | 相机标定 |
| 0 | `03_capture_scan.py` | 转盘拍摄 |
| 1 | `04_stage1_pcd_v5.py` | 深度图→彩色点云，去除背景/地面 |
| 2 | `05_stage2_segment_v3.py` | 裁剪出黑色转盘以上的盆栽 |
| 3 | `06_stage3_register.py` | 多帧配准 (Pose Graph + Color ICP) |
| 4 | `07_stage4_align.py` | 对齐到坐标轴 |
| 5 | `08_stage5_mesh.py` | 泊松网格重建 |

## Stage 3 核心修复 (最重要)

**问题**: ICP 从单位矩阵出发，收敛到局部最优 → 每帧只转 ~0.5°（理论值 ~2.99°），累积仅 55° 而非 360°。

**解决**: 
1. 从 `metadata.json` 读取转速和 fps，算出理论每帧角度
2. 用 `rotation_matrix_y()` 生成角度初始变换 T_init
3. ICP 从 T_init 出发 → 收敛到正确值
4. RANSAC 交叉验证：偏差 <10° 用 RANSAC，>10° 坚持角度初值

```bash
# 推荐用法（自动读 metadata）
python 06_stage3_register.py --input capture_xxx

# 手动指定帧范围（5-123 为一圈，首尾闭合）
python 06_stage3_register.py --input capture_xxx --frame-range 5 123

# 手动指定角度和旋转中心
python 06_stage3_register.py --input capture_xxx --deg-per-frame 2.988 --rotation-center 0 0 0
```

## 关键参数

- **拍摄**: 转速 18.157°/s, fps 6.08, 2圈, 241帧, 总时长 39.65s
- **一圈**: ~119 帧 (帧 5-123 首尾闭合), 每帧理论旋转 2.988°
- **旋转轴**: 之前发现中心在 (-44.2, 907.9), 最新脚本默认原点
- **Y轴**: Y越小越靠上(植物), Y越大越靠下(底座)
- **裁剪**: Y < 130mm (黑色转盘以上)

## 坐标系

- Y轴向上（竖直），旋转绕Y轴
- 点云单位：mm
- 颜色：RGB 0-1 范围

## 目录结构

```
scripts/
├── capture_xxx/            # 拍摄数据 (gitignore: 素材+成果)
│   ├── metadata.json       # ✓ 提交 (拍摄参数)
│   ├── timestamps.csv      # ✓ 提交 (帧时间戳)
│   ├── color/ depth/       # ✗ 原始图
│   ├── pcds/ pcds_seg/     # ✗ 点云
│   └── output_v2/          # ✗ 输出模型
├── calib_imgs/
│   ├── calibration.json    # ✓ 提交
│   └── color/ ir/          # ✗ 标定图
├── utils_kinect.py         # 工具库 (深度对齐、标定加载)
├── 04_stage1_pcd_v5.py     # Stage 1 最终版
├── 05_stage2_segment_v3.py # Stage 2 最终版
├── 06_stage3_register.py   # Stage 3 最终版
├── 07_stage4_align.py      # Stage 4
├── 08_stage5_mesh.py       # Stage 5
└── tool_*.py               # 调试/分析工具脚本
```

## 依赖

Python 3.x, open3d, opencv-python, numpy
