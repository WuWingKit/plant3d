# Kinect v2 转盘扫描 → 三维重建完整工作流

基于张正友标定法 + 立体标定 + 实时对齐 + Pose Graph 6 DOF 配准 + 泊松重建。

## 快速导航

- 📖 **新手开始**：本文件，先看下面的"快速开始"
- 🎯 **标定阶段**：[docs/01_calibration.md](docs/01_calibration.md)
- 📷 **扫描阶段**：[docs/02_capture.md](docs/02_capture.md)
- 🔨 **重建阶段**：[docs/03_reconstruction.md](docs/03_reconstruction.md)
- 🐛 **遇到问题**：[docs/troubleshooting.md](docs/troubleshooting.md)

## 设计思路

- **联合标定**：用张正友法分别标定 RGB / IR 内参，用立体标定求两个相机之间的外参
- **采集时实时对齐**：每帧同时保存 depth + 原始 RGB + **对齐 RGB**（彩色映射到深度坐标系），从源头解决对齐误差
- **重建分 5 阶段**：每个阶段输出可视化文件让你审核，不通过就不进入下一阶段
- **纯 6 DOF Pose Graph 配准**：不依赖"旋转轴"假设，适合转盘也适合手持
- **摆正阶段才用"轴"**：但用的是物理意义上的轴（盆底法向 + PCA 主方向）

## 工作流概览

```
┌──────────────── 标定阶段（拍一次就够，结果可长期复用）───────────────┐
│                                                                  │
│   01_capture_calib.py   →   02_calibrate.py                      │
│      采集 50-60 对           联合标定                              │
│      RGB+IR 同步棋盘         → calibration.json                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────── 扫描阶段（每个花瓶重拍一次）──────────────────────┐
│                                                                  │
│   03_capture_scan.py                                             │
│      录制花瓶绕转盘旋转 1-2 圈                                     │
│      → depth/ + color/ + aligned/                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────── 重建阶段（可反复调参直到满意）─────────────────────┐
│                                                                  │
│   04_stage1_pcd.py     →   05_stage2_segment.py                  │
│      单帧彩色点云           分离前景（去转盘/背景）                  │
│      pcds/0000.ply ...      pcds_seg/0000.ply ...                  │
│                                                                  │
│            ↓                                                     │
│                                                                  │
│   06_stage3_register.py    →   07_stage4_align.py                │
│      Pose Graph 配准合并         摆正（盆底→XY，盆边→X 轴）          │
│      output_v2/merged_*.ply      output_v2/merged_aligned.ply       │
│                                                                  │
│            ↓                                                     │
│                                                                  │
│   08_stage5_mesh.py                                              │
│      泊松重建                                                     │
│      output_v2/plant_model.{obj,stl,ply}                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 文件清单

```
scripts/
├── utils_kinect.py            共用工具（PyKinect2 初始化、IR 预处理、对齐函数）
│
├── 01_capture_calib.py        采集 RGB+IR 同步棋盘对
├── 02_calibrate.py            联合标定（张正友法 + 立体标定）
├── 03_capture_scan.py         扫描花瓶（含实时对齐）
│
├── 04_stage1_pcd.py           Stage 1: 生成单帧彩色点云
├── 05_stage2_segment.py       Stage 2: 纯几何前景分离
├── 06_stage3_register.py      Stage 3: Pose Graph 配准（6 DOF + Color ICP）
├── 07_stage4_align.py         Stage 4: 摆正（盆底找平 + PCA 对齐）
├── 08_stage5_mesh.py          Stage 5: 泊松重建
│
└── tool_diagnose_depth.py     工具：诊断深度图格式（出问题时用）

docs/
├── README.md                  本文件（总览）
├── 01_calibration.md          标定阶段详解
├── 02_capture.md              扫描阶段详解
├── 03_reconstruction.md       重建阶段详解
└── troubleshooting.md         常见问题排查
```

## 环境准备

```bash
pip install -r requirements.txt
```

**PyKinect2 64 位 Python 补丁**（不打补丁会崩）：

1. **`<env>/Lib/site-packages/comtypes/_tlib_version_checker.py`**：
   注释掉 `assert sizeof(tagSTATSTG) == 72` 这一行

2. **`<env>/Lib/site-packages/pykinect2/PyKinectRuntime.py`**：
   全文替换 `time.clock()` → `time.perf_counter()`

## 物理准备

- **棋盘**：7×6 内角点（即 8×7 方格），格边长 25mm，**贴硬纸板/铝板保证平面性**
- **转盘**：表面贴白纸（避免 Kinect 看不到黑色），转一圈 15-30 秒
- **环境**：室内日常照明即可（不需要 IR 光源、不需要盖 IR 投射器）

## 快速开始

**有两种标定方案**（任选一个）：

### 方案 A：完整联合标定（精度最高，但 IR 棋盘检测可能困难）

```bash
# 1. 同步采集 RGB+IR 棋盘对
python 01_capture_calib.py --output calib_imgs --pattern 8x6

# 2. 联合标定（张正友 + 立体）
python 02_calibrate.py --input calib_imgs --pattern 8x6 --square 25
```

### 方案 B：只标定 RGB + 手动对齐（推荐，特别是 IR 棋盘检测失败时）⭐

```bash
# 1. 只采集 RGB 棋盘（不要 IR，散斑/低对比度都不影响）
python 01b_capture_calib_rgb.py --output calib_imgs --pattern 8x6

# 2. 仅标定 RGB（深度内参用 Kinect 出厂值）
python 02_calibrate.py --input calib_imgs --pattern 8x6 --square 25 --rgb-only

# 3. 手动微调外参（可选，不调直接用出厂值也可以）
python 02b_manual_align.py --calib calib_imgs/calibration.json
```

### 然后扫描 + 重建（两个方案都一样）

```bash
# 4. 扫描花瓶
python 03_capture_scan.py --output capture_花瓶_xxx --calib calib_imgs/calibration.json --fps 8

# 5-9. 五阶段重建
python 04_stage1_pcd.py --input capture_花瓶_xxx --calib calib_imgs/calibration.json
python 05_stage2_segment.py --input capture_花瓶_xxx
python 06_stage3_register.py --input capture_花瓶_xxx
python 07_stage4_align.py --input capture_花瓶_xxx
python 08_stage5_mesh.py --input capture_花瓶_xxx
```

## 详细说明请看 docs/

- **[01_calibration.md](docs/01_calibration.md)** — 标定阶段（最关键，一次做好长期受益）
- **[02_capture.md](docs/02_capture.md)** — 扫描阶段（含转盘/光照建议）
- **[03_reconstruction.md](docs/03_reconstruction.md)** — 重建阶段（每个 stage 的参数和验证方法）
- **[troubleshooting.md](docs/troubleshooting.md)** — 常见问题和调参建议

## 输出文件结构（运行完后）

```
calib_imgs/                          # 标定数据
├── color/0000.png ...               # 50-60 张彩色棋盘
├── ir/0000.png ...                  # 50-60 张 IR 棋盘
└── calibration.json                 # ★ 标定结果（长期复用）

capture_花瓶_xxx/                    # 一次扫描
├── color/0000.png ...               # 原始彩色（1920×1080）
├── depth/0000.png ...               # 原始深度（512×424 uint16 mm）
├── aligned/0000.png ...             # 对齐 RGB（512×424，每像素对应 depth）
├── timestamps.csv                   # 每帧时间戳
├── metadata.json                    # 转盘信息（圈数、时长、转速）
│
├── pcds/                            # Stage 1 输出
│   ├── 0000.ply ...                 # 单帧彩色点云
│   ├── pcd_index.json
│   └── quality_report.html
│
├── pcds_seg/                        # Stage 2 输出
│   ├── 0000.ply ...                 # 前景分离后的点云
│   └── segment_log.json
│
└── output_v2/                       # Stage 3-5 输出
    ├── merged_raw.ply               # 配准后原始合并
    ├── merged_clean.ply             # 去噪+降采样
    ├── merged_color_coded.ply       # ★ 每帧不同颜色，看对齐质量
    ├── merged_aligned.ply           # 摆正后
    ├── poses.json                   # 每帧的位姿
    ├── registration_log.csv         # 配准日志（每条边的 fitness/rmse）
    ├── align_transform.json         # 摆正变换矩阵
    └── plant_model.{obj,stl,ply}    # ★ 最终模型
```

## 关键设计原则

### 1. 标定阶段：张正友法（论文：Zhang 2000, IEEE PAMI）

单相机标定核心：让相机从多个姿态观察平面棋盘，每个姿态给出"棋盘平面→图像平面"的单应矩阵 H。H 对内参 K 提供 2 个约束方程，N 张图就有 2N 个方程，足以解出 K。最后用 LM 优化最小化重投影误差。

**联合标定 = 张正友法 × 2 + 立体标定**：
- 用 `cv2.calibrateCamera` 分别标定 RGB 和 IR 内参
- 用 `cv2.stereoCalibrate` + `CALIB_FIX_INTRINSIC` 求两相机外参（R, T）

### 2. 扫描阶段：实时对齐

不用 PyKinect2 自带的 `MapDepthFrameToColorSpace`（它用 SDK 出厂参数），改用你自己标定的 K_d、K_c、dist、R、T 手动计算每个深度像素对应的彩色像素，存到 `aligned/`。

重建时直接 `aligned[v, u]` 就是颜色，零对齐误差。

### 3. 重建阶段：分阶段验证

每个 stage 输出可视化文件，你审核后再进下一步：

| Stage | 验证文件 | 该看什么 |
|---|---|---|
| 1 | `pcds/0000.ply` | 单帧彩色对吗？形状对吗？ |
| 2 | `pcds_seg/0000.ply` | 只剩花瓶+植物了吗？ |
| 3 | `output_v2/merged_color_coded.ply` | 不同颜色的帧叠在一起吗？ |
| 4 | `output_v2/merged_aligned.ply` | 花瓶正立了吗？ |
| 5 | `output_v2/plant_model.obj` | 模型完整吗？ |

### 4. 配准：纯 6 DOF Pose Graph

每对帧用 FPFH+RANSAC 粗配 → Point-to-Plane ICP 几何精配 → Color ICP 颜色精化。

构建 Pose Graph：节点=每帧位姿，边=帧对之间的相对变换。
- 相邻边（i↔i+1）：链接所有帧
- 跳跃边（i↔i+2）：增加冗余约束
- 闭环边（首↔尾、对面位置）：消除累积误差

`global_optimization` 用 LM 同时优化所有位姿，单条边失败不致命。

### 5. 摆正：从物理几何特征出发

不假设转轴朝向，从最终点云的特征推出来：
- 水平校正：取点云最低 15% 区域，RANSAC 拟合"盆底"平面，Rodrigues 旋转让法向 → Z 轴
- 方向校正：盆底点投 XY 平面，PCA 找最大主轴，绕 Z 旋转让主轴 → X 轴

---

**下一步**：看 [docs/01_calibration.md](docs/01_calibration.md) 开始第一步。
