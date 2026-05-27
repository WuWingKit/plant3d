# 盆栽 3D 扫描重建

使用 Kinect v2 深度相机 + 电动转盘拍摄盆栽，自动生成 360° 3D 模型。

## 流水线 (6 Stages)

| Stage | 脚本 | 功能 |
|-------|------|------|
| 0 | `01_capture_calib.py` → `02_calibrate.py` | 相机标定 |
| 0 | `03_capture_scan.py` | 转盘拍摄 |
| 1 | `04_stage1_pcd_v5.py` | 深度图→彩色点云，去除背景/地面 |
| 2 | `05_stage2_segment_v3.py` | 裁剪出黑色转盘以上的盆栽 |
| 3 | `06_stage3_register.py` | 多帧配准 (Pose Graph + Color ICP) |
| 9 | `09_postprocess.py` | 9步后处理: SOR+ROR+降采样+RANSAC+泊松+填洞(pymeshlab)+平滑+法向修正+去碎片 |
| 10 | `10_upsample.py` | 点云上采样(线性插值)，从09的中间结果进一步加密 |

## 09_postprocess 最佳参数

```bash
# 完整流程: 10_upsample → 09_postprocess
python 10_upsample.py --input capture_xxx
python 09_postprocess.py --input capture_xxx --source pcd_upsampled.ply \
    --poisson-depth 10 --density-cut 0.05 --normal-radius 12 \
    --fill-hole-size 200 --smooth-method pymeshlab_laplacian --taubin-iter 25
```

后处理步骤:
1. SOR 统计滤波 (k=20, std=2.0)
2. ROR 半径滤波 (min=5, r=10mm)
3. 体素降采样 (2mm)
4. RANSAC 去平面 (可选)
5. 泊松重建 (depth=10, density_cut=0.05, normal_radius=12)
6. 填洞 pymeshlab (max_hole_size=200) + 法向量修正
7. 平滑 pymeshlab Laplacian 25次 (全局)
8. 花瓶区域追加 HC Laplacian 渐变平滑 (Y>60:30次, Y40-60:15次, Y30-40:5次)
9. 删除孤立碎片

注意:
- pymeshlab 填洞后面片朝向会反转，需要 step_fix_normals 修正
- 花瓶分界面在 Y≈30-44，平滑需渐变过渡避免硬边界
- 依赖: open3d, pymeshlab, opencv-python, numpy

## Stage 3 核心修复 (最重要)

**问题**: ICP 从单位矩阵出发，收敛到局部最优 → 每帧只转 ~0.5°（理论值 ~2.99°），累积仅 55° 而非 360°。

**解决** (三组改进):
1. **角度初值**: 从 metadata.json 算理论角度，ICP 从初值出发 → 收敛到正确值
2. **RANSAC 交叉验证**: 偏差 <10° 用 RANSAC，>10° 坚持角度初值
3. **Pose Graph 结构改进 (A+E+K)**:
   - A: 跳跃边 i↔i+2/3/4（更多冗余约束）
   - E: 闭环 n//8 对均匀分布（消除累积误差堆积）
   - K: 角度偏差 >5° 的坏边直接丢弃
4. **ICP 精度 (C+N)**:
   - C: 第4层精细 ICP (voxel×0.5, 40次迭代)
   - N: 更严格收敛标准 (relative_fitness/rmse=1e-7)
5. **两轮优化 (D+F)**:
   - D: 第二轮用第一轮位姿作 ICP 初值重新配准
   - F: max_correspondence_distance 1.0, preference_loop_closure=5.0

```bash
# 推荐用法（所有改进全开，两轮优化）
python 06_stage3_register.py --input capture_xxx --frame-range 5 123

# 关闭两轮优化（省时间）
python 06_stage3_register.py --input capture_xxx --no-two-pass

# 自定义跳跃步长
python 06_stage3_register.py --input capture_xxx --skip-steps 2 3

# 调整坏边剔除阈值
python 06_stage3_register.py --input capture_xxx --angle-reject 8.0
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
