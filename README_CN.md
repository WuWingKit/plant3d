# Plant3D

[English Version](README.md)

基于 Kinect v2 深度相机 + 电动转盘的盆栽 3D 重建流水线。

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Open3D](https://img.shields.io/badge/Open3D-≥0.15-green)
![OpenCV](https://img.shields.io/badge/OpenCV-≥4.5-blue?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-≥1.20-blue?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-≥1.7-blue?logo=scipy&logoColor=white)
![pymeshlab](https://img.shields.io/badge/pymeshlab-≥2022.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)

拍摄转盘上盆栽的多视角深度图像，通过点云配准、后处理和表面重建，自动生成带颜色的 3D 模型。

## 流水线

| 阶段 | 脚本 | 功能 |
|:----:|------|------|
| 1 | `01_capture_calib.py` → `02_calibrate.py` | 相机标定（张正友法 + 立体标定） |
| 2 | `03_capture_scan.py` | 转盘拍摄，实时 RGB-深度对齐 |
| 3 | `04_stage1_pcd_v5.py` | 深度图 → 彩色点云，去除背景 |
| 4 | `05_stage2_segment_v3.py` | 裁剪黑色转盘以上的盆栽 |
| 5 | `06_stage3_register.py` | 多帧配准（Pose Graph + Color ICP） |
| 6 | `09_postprocess.py` | 9步后处理：SOR+ROR+降采样+泊松+填洞+平滑+法向修正 |
| 7 | `10_upsample.py` | 点云上采样（线性插值） |
| 8 | `12_hull_colored.py` | 凹包包裹点云 + 颜色上色 |

## 快速开始

```bash
# 1. 标定相机（一次性）
python scripts/01_capture_calib.py --output calib_imgs
python scripts/02_calibrate.py --input calib_imgs --pattern 7x6 --square 25

# 2. 扫描盆栽
python scripts/03_capture_scan.py --output capture_xxx --calib calib_imgs/calibration.json

# 3. 构建点云
python scripts/04_stage1_pcd_v5.py --input capture_xxx
python scripts/05_stage2_segment_v3.py --input capture_xxx

# 4. 多帧配准
python scripts/06_stage3_register.py --input capture_xxx --frame-range 5 123

# 5. 后处理生成 mesh
python scripts/10_upsample.py --input capture_xxx
python scripts/09_postprocess.py --input capture_xxx --source pcd_upsampled.ply \
    --poisson-depth 10 --density-cut 0.05 --normal-radius 12

# 6. 替代方案：凹包包裹 + 颜色上色
python scripts/12_hull_colored.py --input capture_xxx --alpha 4.5 --outlier-pct 0.03 \
    --subdivide 2 --smooth 30 --color-radius 20
```

## 文档

| 主题 | 说明 |
|------|------|
| [标定详解](docs/01_calibration.md) | 张正友法、立体标定、质量验证 |
| [扫描详解](docs/02_capture.md) | 转盘设置、采集流程、数据质量检查 |
| [重建详解](docs/03_reconstruction.md) | 逐阶段参数调优指南 |
| [常见问题](docs/troubleshooting.md) | 常见问题排查与解决方案 |

## 依赖

```bash
pip install -r requirements.txt
```

| 包 | 版本 |
|---------|---------|
| numpy | >= 1.20 |
| opencv-python | >= 4.5 |
| open3d | >= 0.15 |
| scipy | >= 1.7 |
| pymeshlab | >= 2022.2 |
| pykinect2 | >= 0.1.0 |
| comtypes | — |

## 硬件

- **深度相机**：Kinect v2（Microsoft）
- **转盘**：电动匀速转盘
- **拍摄参数**：~241 帧/2 圈，6 fps，转速 18.157°/s

## 核心技术

- **标定**：张正友法分别标定 RGB/IR 内参 + 立体标定求外参
- **实时对齐**：用自标定参数做深度到彩色的映射（非 SDK 默认）
- **配准**：Pose Graph（相邻边 + 跳跃边 + 闭环边），LM 优化
- **后处理**：泊松重建 + pymeshlab 填洞 + 分区域渐变平滑
- **凹包**：分区域 alpha（花盆底部大 alpha 保连通，植物小 alpha 保细节）+ 朝向修正

## 09_postprocess 最佳参数

```bash
python 10_upsample.py --input capture_xxx
python 09_postprocess.py --input capture_xxx --source pcd_upsampled.ply \
    --poisson-depth 10 --density-cut 0.05 --normal-radius 12 \
    --fill-hole-size 200 --smooth-method pymeshlab_laplacian --taubin-iter 25
```

后处理步骤：
1. SOR 统计滤波 (k=20, std=2.0)
2. ROR 半径滤波 (min=5, r=10mm)
3. 体素降采样 (2mm)
4. RANSAC 去平面（可选）
5. 泊松重建 (depth=10, density_cut=0.05, normal_radius=12)
6. 填洞 pymeshlab (max_hole_size=200) + 法向量修正
7. 平滑 pymeshlab Laplacian 25次
8. 花瓶区域追加 HC Laplacian 渐变平滑
9. 删除孤立碎片

## Stage 3 配准改进

**问题**：ICP 从单位矩阵出发，收敛到局部最优 → 累积仅 55° 而非 360°。

**解决**：
1. **角度初值**：从 metadata.json 算理论角度，ICP 从初值出发
2. **RANSAC 交叉验证**：偏差 <10° 用 RANSAC，>10° 坚持角度初值
3. **Pose Graph 改进**：跳跃边 i↔i+2/3/4 + 闭环 n//8 + 坏边剔除（>5°）
4. **ICP 精度**：第4层精细 ICP (voxel×0.5, 40次迭代) + 严格收敛标准 (1e-7)
5. **两轮优化**：第二轮用第一轮位姿作 ICP 初值重新配准

## 目录结构

```
scripts/
├── capture_xxx/            # 拍摄数据
│   ├── metadata.json       # 拍摄参数
│   ├── timestamps.csv      # 帧时间戳
│   ├── color/ depth/       # 原始图
│   ├── pcds/ pcds_seg/     # 点云
│   └── output_v2/          # 输出模型
├── calib_imgs/             # 标定图片
│   └── calibration.json    # 标定结果
├── utils_kinect.py         # 工具库
└── 01-12_*.py              # 流水线脚本
```

## 坐标系

- Y 轴向上（竖直），旋转绕 Y 轴
- 单位：mm
- 颜色：RGB 0-1

## License

MIT
