# 重建阶段详解

> 五个 stage 逐步运行，每个 stage 都有可视化中间输出。
> 调参时只需重跑出错的 stage，前面的中间结果可复用。

## 一、整体流程

```
原始数据 (depth/ + aligned/)
       ↓
Stage 1: 单帧彩色点云  →  pcds/0000.ply ...
       ↓ ✓ 用 MeshLab 抽几张检查
Stage 2: 前景分离      →  pcds_seg/0000.ply ...
       ↓ ✓ 检查只剩花瓶+植物
Stage 3: Pose Graph 配准 →  merged_*.ply
       ↓ ✓ 检查 merged_color_coded.ply
Stage 4: 摆正            →  merged_aligned.ply
       ↓ ✓ 检查花瓶正立
Stage 9: 后处理+泊松重建  →  plant_model.{obj,stl,ply}
```

---

## 二、Stage 1: 单帧彩色点云生成

```bash
python 04_stage1_pcd_v5.py --input capture_花瓶_xxx --calib calib_imgs/calibration.json
```

**作用**：
- 读取每帧 `depth/XXXX.png` + `aligned/XXXX.png`
- 用标定的 IR 内参反投影深度像素为 3D 点
- 用对齐的彩色像素直接赋色（零对齐误差）
- 输出独立的 PLY 文件 + 质量报告 HTML

**参数**：
- `--n 36` — 均匀抽取 36 帧（默认）
- `--every 18` — 改为每 18 帧取一帧
- `--all` — 处理所有帧（约 633 帧，5-10 分钟）
- `--depth-trunc 3000` — 深度截断 mm（默认 3000，避免远处噪声）

**验证步骤**：

1. 打开 `pcds/quality_report.html` 看每帧统计
   - 点数应在 30,000-100,000 之间
   - 中位深度应在 1200-1800mm 之间
   - 警告列空白最好
2. MeshLab 打开 `pcds/0000.ply`、`pcds/0018.ply`、`pcds/0035.ply` 各一帧
   - 看花瓶+植物+部分背景
   - 颜色和形状应该一致
   - 没有"鬼影"或漂浮的离群点

**常见问题**：

| 现象 | 原因 | 处理 |
|---|---|---|
| 点数 < 5000 | 距离过近/过远 | 调 `--depth-trunc` |
| 颜色错位 | 标定有问题 | 重新标定 |
| 形状扭曲 | 深度数据损坏 | 跑 `tool_diagnose_depth.py` 诊断 |

---

## 三、Stage 2: 前景分离（纯几何）

```bash
python 05_stage2_segment_v3.py --input capture_花瓶_xxx
```

**作用**：用纯几何方法分离花瓶+植物

算法：
1. RANSAC 找最大水平面 → 桌面/转盘表面
2. 自动判断"哪个轴是竖直方向"（不预设朝向）
3. 平面之上的点 = 前景候选
4. DBSCAN 聚类，取最大簇 = 花瓶+植物
5. HSV 过滤剩余黑色点（残留转盘）

**注意**：这一步**不依赖任何"中心"或"轴"假设**——纯粹基于点云本身的几何结构。

**参数**：
- `--dist-thresh 8` — 平面拟合距离阈值（默认 8mm）
- `--eps 15` — DBSCAN 邻域半径（聚类松紧）
- `--min-points 200` — DBSCAN 最小簇大小
- `--no-hsv` — 跳过 HSV 过滤（仅几何）
- `--drop-dark-v 40` — V < 40 视为黑色，去除
- `--drop-bright-v 252` — V > 252 视为过曝，去除

**输出**：
- `pcds_seg/0000.ply ...` — 每帧分离后的点云
- `pcds_seg/segment_log.json` — 每帧统计 + 警告

**关键检查**：日志里 `vertical_axis` 和 `plane_side` 是否所有帧一致

如果不一致 → 某些帧 RANSAC 找错了平面 → 调 `--dist-thresh` 重试

**验证**：MeshLab 打开 `pcds_seg/0000.ply`
- ✓ 只剩花瓶+植物
- ✓ 没有桌面/转盘/背景
- ✓ 没有杂物簇

**调参建议**：

| 现象 | 处理 |
|---|---|
| 花瓶被切掉一半 | 调大 `--eps 20` 或 `--dist-thresh 12` |
| 还有黑转盘残留 | 调小 `--eps 10` 或 `--drop-dark-v 60` |
| 包含了桌面 | 调小 `--dist-thresh 5` |
| 包含离群小簇 | 调大 `--min-points 500` |

---

## 四、Stage 3: Pose Graph 配准（核心）

```bash
python 06_stage3_register.py --input capture_花瓶_xxx
```

**作用**：把多帧点云配准合并到统一坐标系

**关键设计：纯 6 DOF Pose Graph**：
- 每帧位姿是 6 DOF（3 旋转 + 3 平移）
- **不假设旋转轴**（手持相机自由拍摄也能用）
- 全局优化所有位姿，单条边失败不致命

**配准方法**（每对帧）：
1. 降采样 + 估计法向量 + 计算 FPFH 特征
2. FPFH + RANSAC 粗配（找初始变换）
3. Multi-scale Point-to-Plane ICP 几何精配
4. **Color ICP 颜色精化**（弥补几何对称物体的歧义）

**Pose Graph 边类型**：
- **相邻边** `i ↔ i+1` — 链接所有帧
- **跳跃边** `i ↔ i+2` — 增加冗余约束
- **闭环边** — 首尾、对面位置（消除累积误差）

**全局优化**：用 LM 同时优化所有位姿，最小化所有边的残差。

**参数**：
- `--voxel 8` — 配准用降采样体素 mm（默认 8）
  - 减小（5）→ 更精确但慢
  - 增大（12）→ 更稳但精度下降
- `--loop-closures 3` — 闭环边数量（默认 3）
- `--no-color-icp` — 禁用 Color ICP（仅几何）

**输出**：
- `output_v2/merged_raw.ply` — 配准后原始合并
- `output_v2/merged_clean.ply` — 去噪 + 降采样
- `output_v2/merged_color_coded.ply` — **★ 每帧不同颜色，对齐质量调试用**
- `output_v2/poses.json` — 每帧的最终位姿（4×4 矩阵）
- `output_v2/registration_log.csv` — 每条边的 fitness、RMSE

**验证步骤**：

1. **打开 `merged_color_coded.ply`** — 这是最重要的检查
   - **从俯视图（顶视）看**
   - ✓ 不同颜色的帧叠在一起，形成清晰的花瓶轮廓
   - ✗ 不同颜色形成花瓣状分布 → 配准失败
2. 打开 `merged_clean.ply` 看花瓶完整 360° 形状
3. 看 `registration_log.csv` 配准成功率
   - 总成功率 > 70% → 不错
   - 总成功率 50-70% → 可用
   - < 50% → 重做或调参

**调参建议**：

| 现象 | 处理 |
|---|---|
| 配准失败率 > 50% | 试 `--voxel 12`（更稀疏，特征更稳）|
| 仍然花瓣状 | 试 `--no-color-icp`，看是否变好 |
| 闭环失败 | 试 `--loop-closures 4` 增加冗余 |
| Stage 2 点数太少 | 回 Stage 2 调宽，保留更多上下文（点数太少时 FPFH 失败）|

---

## 五、Stage 4: 摆正

```bash
python 07_stage4_align.py --input capture_花瓶_xxx
```

**作用**：把花瓶摆正——盆底贴 XY 平面，盆边对齐 X 轴

**关键设计**：从最终点云的**物理几何特征**推出方向（不用拍摄时的转轴假设）

**算法**：
1. 取点云最低 15% 区域（盆底候选）
2. RANSAC 拟合盆底平面，得法向 N
3. **水平校正**：Rodrigues 公式让 N → Z 轴
4. 旋转后取盆底点投影到 XY 平面
5. **PCA 找最大主轴**
6. **方向校正**：绕 Z 轴旋转让主轴 → X 轴

**为什么需要这一步**：

Stage 3 的合并点云仍然在相机坐标系下，朝向是任意的。要测量花瓶尺寸/分析对称性都需要先把它**摆到标准朝向**。

**参数**：
- `--low-percent 15` — 取最低 N% 找盆底（默认 15）
- `--dist-thresh 5` — 平面距离阈值 mm
- `--no-orient` — 跳过方向校正（只做水平校正）

**输出**：
- `output_v2/merged_aligned.ply` — 摆正后的点云
- `output_v2/align_transform.json` — 4×4 变换矩阵 + 元信息

**验证**：

MeshLab 打开 `merged_aligned.ply`：
- ✓ 花瓶正立（盆底贴 Z=0 平面，植物向上 +Z）
- ✓ 包围盒的 Z 范围 = 实际花瓶高度
- ✓ 方形花瓶的话，盆边平行于 X 或 Y 轴

**调参建议**：

| 现象 | 处理 |
|---|---|
| 摆得歪了 | 看日志的 `候选平面` 得分，可能选错了。调 `--low-percent 20` 或 `--dist-thresh 8` |
| 倒过来了 | 程序自动判断"物体在哪一侧"，正常不应该出错。看 `align_transform.json` 的 `base_normal` |
| 方向不对（盆边没对齐 X）| `--no-orient` 跳过这步，或者旋转对称物体本来就不该做方向校正 |

---

## 六、后处理：泊松重建 + 9 步清理

```bash
python 10_upsample.py --input capture_花瓶_xxx
python 09_postprocess.py --input capture_花瓶_xxx --source pcd_upsampled.ply \
    --poisson-depth 10 --density-cut 0.05 --normal-radius 12
```

**作用**：从点云生成网格模型，9 步后处理

**流程**：
1. SOR 统计滤波
2. ROR 半径滤波
3. 体素降采样
4. RANSAC 去平面（可选）
5. 泊松重建
6. 填洞（pymeshlab）
7. 平滑（Laplacian / Taubin）
8. 花瓶区域渐变平滑
9. 删除孤立碎片

**参数**：
- `--poisson-depth 10` — 泊松深度（默认 9）
- `--density-cut 0.05` — 低密度裁剪比例（默认 0.02）
- `--normal-radius 12` — 法向估计半径 mm
- `--fill-hole-size 200` — 填洞最大边界边长
- `--smooth-method pymeshlab_laplacian` — 平滑方法
- `--taubin-iter 25` — 平滑迭代次数

**输出**：
- `output_v2/plant_model.obj` — 带顶点色
- `output_v2/plant_model.stl` — 几何
- `output_v2/plant_model.ply` — 带顶点色

**验证**：

MeshLab 打开 `plant_model.obj`：
- ✓ 完整模型，没有大洞
- ✓ 颜色合理（白瓶绿叶）
- ✓ 表面光滑无毛刺

**调参建议**：

| 现象 | 处理 |
|---|---|
| 模型有大洞 | 调大 `--poisson-depth 10`；或调小 `--density-cut 0.05` |
| 模型有毛刺 | 调大 `--density-cut 0.15` |
| 顶部植物缺失 | 输入点云本身就缺，没救（采集时改高度） |
| 颜色错位 | Stage 1 的颜色就错位，回到标定 |

---

## 七、整体调试流程图

```
                ┌──────────┐
                │ Stage 1  │
                └────┬─────┘
                     ↓
                 ┌───┴────┐
                 │ 检查   │── 颜色错？ → 重新标定
                 │ pcds/  │── 形状错？ → 检查深度数据
                 └───┬────┘
                     ↓ OK
                ┌────┴─────┐
                │ Stage 2  │
                └────┬─────┘
                     ↓
                 ┌───┴────┐
                 │ 检查   │── 切多了？ → eps↑ / dist-thresh↑
                 │ seg/   │── 留多了？ → eps↓ / min-points↑
                 └───┬────┘
                     ↓ OK
                ┌────┴─────┐
                │ Stage 3  │
                └────┬─────┘
                     ↓
                 ┌───┴────────┐
                 │ 检查 coded │── 花瓣状？ → 看 log，调 voxel
                 │            │── 失败多？ → 回 Stage 2 留更多上下文
                 └───┬────────┘
                     ↓ OK
                ┌────┴─────┐
                │ Stage 4  │
                └────┬─────┘
                     ↓
                 ┌───┴──────┐
                 │ 检查正立  │── 歪？ → low-percent↑
                 └───┬──────┘
                     ↓ OK
                ┌────┴──────┐
                │ 09 后处理 │
                └────┬──────┘
                     ↓
                ┌────┴──────┐
                │ 检查模型  │── 有洞？ → depth↑
                │           │── 毛刺？ → density-cut↑
                └───────────┘
```

## 八、典型的成功跑通耗时

| Stage | 36 帧 | 120 帧 | 633 帧（--all） |
|---|---|---|---|
| Stage 1 | 30 秒 | 1.5 分钟 | 8 分钟 |
| Stage 2 | 1 分钟 | 3 分钟 | 15 分钟 |
| Stage 3 | 2-5 分钟 | 10-20 分钟 | （不推荐）|
| Stage 4 | 10 秒 | 10 秒 | 10 秒 |
| 09 后处理 | 30 秒 | 1 分钟 | 2 分钟 |
| **总计** | **5-8 分钟** | **15-25 分钟** | — |

**推荐用 36 帧**，调通流程后再考虑加帧。

## 下一步

- 全部跑通 → 用 MeshLab/Blender 打开 `plant_model.obj` 看最终结果
- 遇到问题 → 看 **[troubleshooting.md](troubleshooting.md)**
