# 常见问题排查

按"现象 → 可能原因 → 处理"组织。先用 Ctrl+F 搜你看到的现象。

## 一、环境问题

### PyKinect2 安装后 import 报错

**现象**：`from pykinect2 import PyKinectRuntime` 时崩溃，提示 `sizeof` 断言失败或 `time.clock` 不存在。

**原因**：PyKinect2 在 64 位 Python 3.8+ 上需要打两处补丁。

**处理**：
1. 找到 `<env>/Lib/site-packages/comtypes/_tlib_version_checker.py`，注释掉这一行：
   ```python
   # assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)
   ```
2. 找到 `<env>/Lib/site-packages/pykinect2/PyKinectRuntime.py`，全文替换：
   ```
   time.clock()  →  time.perf_counter()
   ```

### Kinect 没反应

**现象**：脚本启动后 `kinect.has_new_color_frame()` 一直 False。

**检查清单**：
- [ ] Kinect v2 必须插 USB 3.0 接口（蓝色口）
- [ ] 安装了 Kinect SDK 2.0（不只是 PyKinect2）
- [ ] Kinect 电源已插（外接电源那条线）
- [ ] 打开 Kinect Studio 验证硬件能识别
- [ ] 没有其他程序占用 Kinect（关闭 Skype/相机软件）

---

## 二、标定问题

### IR 角点检测成功率低（< 50%）

**原因**：散斑环境下 IR 检测难度大，光线不足/反光会让角点丢失。

**处理**：
1. **加强光线**：开多盏灯，或者把场景挪到窗边
2. **棋盘举近一点**：1.0-1.5m 范围内最准
3. **棋盘平面性**：贴硬纸板，纸张柔性变形会让角点抖
4. **棋盘别太斜**：倾角 < 45°
5. **最后一招**：拿个 LED 强光手电从侧面照棋盘

### 立体标定 RMS > 2px

**原因**：单相机内参误差大，或同步对不够。

**处理**：
1. 看 RGB 单独 RMS 和 IR 单独 RMS，**谁更高谁是元凶**
2. 同步对至少要 30 对有效，少了补拍
3. 检查棋盘是不是真的平整（拿尺子量对角线，应该 ±0.5mm 内）

### T 向量异常（不接近 [-52, 0, 0]）

**原因**：标定算错了，可能棋盘旋转方向被搞反。

**处理**：
- 看 `02_calibrate.py` 的输出，stereoCalibrate 的源/目标顺序应该是 `imgpoints_ir → imgpoints_color`
- 如果 T 是 [+52, 0, 0]，说明源/目标反了。当前代码是对的，如果你改过就改回来
- 如果 T 远离这个范围（比如 [-200, 100, 0]），重做标定

---

## 三、采集问题

### 预览窗口里对齐 RGB 错位

**现象**：右下角的"对齐叠加"图里，颜色和深度边缘明显错开。

**原因**：标定有问题。

**处理**：重新标定。这个错位会被 Stage 1 的彩色点云继承。

### 录制后没有 metadata.json

**现象**：脚本崩溃或被强制关闭，aligned/ 有数据但没 metadata。

**处理**：手动创建 `metadata.json`：

```json
{
  "n_frames": 120,
  "fps_target": 8.0,
  "fps_actual": 8.0,
  "n_turns": 2.0,
  "total_time_sec": 30.0,
  "rotation_speed_deg_per_sec": 24.0
}
```

数值随便填合理的就行——Stage 3 Pose Graph 不依赖这个，只在 Stage 1 旧脚本里用过。

### 深度图都是黑的

**现象**：直接看 `depth/0000.png` 完全是黑的。

**原因**：可能是数据格式问题。

**处理**：跑诊断：
```bash
python tool_diagnose_depth.py --dir capture_花瓶_xxx/depth
```
- 如果 dtype 是 uint16 → 正常（深度图归一化前都看着黑）
- 如果 dtype 是 uint8 → **数据损坏，重新采集**

---

## 四、Stage 1 问题

### 点数太少（< 5000）

**原因**：
- 距离过近/过远（超出 `--depth-trunc`）
- 大面积无效深度（黑色物体太多）

**处理**：
- 调 `--depth-trunc 3500`
- 或者改善拍摄环境

### 颜色错位（看着花瓶白但点云上是红色）

**原因**：标定不准，或者 aligned 计算错。

**处理**：
1. 先看 `aligned/0000.png` 本身，肉眼对一下深度边缘 → 错位严重 = 标定问题
2. 重新标定（看 [01_calibration.md](01_calibration.md)）

### 形状扭曲（花瓶变形）

**原因**：
- 深度数据本身有问题（跑诊断）
- IR 内参不对（用 Kinect 出厂值通常没问题，重标定后该更准）

---

## 五、Stage 2 问题

### "vertical_axis 分布不一致" 警告

**原因**：不同帧 RANSAC 找到了不同的水平面（有的是桌面，有的是墙）。

**处理**：
1. 调大 `--dist-thresh 12` 让 RANSAC 更容忍
2. 检查 stage1 输出，看是不是某些帧距离异常
3. 在脚本里 hardcode 一个 vertical_axis（默认让程序自动判断，但物理上你的相机姿态固定，应该总是某个轴）

### 花瓶被切掉一半

**原因**：DBSCAN 把花瓶切成了两个簇，只保留了大的那个。

**处理**：
- `--eps 20` 让聚类更宽松
- `--dist-thresh 12` 让平面阈值更大，可能少切一些

### 还有黑转盘残留

**原因**：HSV 过滤太宽松。

**处理**：
- `--drop-dark-v 60` 删更多深色
- 注意：阴影部位的 V 也很低，调太狠会切植物的阴影面

### 完全没有输出（segment_log.json 里大部分 stage="..."）

**原因**：
- 点云本身就稀疏（Stage 1 问题）
- 平面找不到（RANSAC 失败）

**处理**：检查 stage1 输出，看 PLY 大小、点数

---

## 六、Stage 3 问题

### 配准失败率 > 50%

**原因**：分割后点云太"小"（FPFH 特征不够），或者花瓶过于光滑（FPFH 描述子退化）。

**处理**：

1. **回到 Stage 2 留更多上下文**：调大 `--eps`，让分割后的点云包含一些花盆周围
2. **试不同 voxel**：
   - `--voxel 6` 更精细
   - `--voxel 12` 更稀疏（特征更稳）
3. **试关闭 Color ICP**：`--no-color-icp` 看是否变好
4. **增加闭环**：`--loop-closures 5`

### merged_color_coded.ply 上不同颜色形成花瓣

**原因**：配准完全失败，每帧都对在了相机原始位置（没有旋转）。

**处理**：
- 看 `registration_log.csv`，如果 RANSAC fitness 全在 0.1 以下，说明 FPFH 完全不工作
- 这种情况下点云特征太弱，需要：
  - 回 Stage 2 留更多点（包括转盘周围背景）
  - 或换成更密集采样（Stage 1 用更多帧）

### Pose Graph 全局优化后位姿崩溃

**原因**：闭环边错位带歪了全局解。

**处理**：
- `--loop-closures 1` 只用首尾闭环
- 看 `registration_log.csv` 的 `loop` 类型边，如果 fitness < 0.3 应该排除
- 当前脚本会自动排除低 fitness 边，问题不大

---

## 七、Stage 4 问题

### 摆得歪了（盆底没贴平面）

**原因**：找到的"盆底"不是真正的盆底。

**处理**：
- 看日志的 `候选平面` 列表，选了哪个 axis/sign
- 如果选错了，调 `--low-percent 20` 取更多盆底点
- 或者 hardcode：用 `--no-orient` 跳过方向校正先看水平校正对不对

### 花瓶倒过来了

**原因**：法向方向判断错了。

**处理**：
- 看 `align_transform.json` 的 `base_normal`
- 在 MeshLab 里手动绕 X 或 Y 轴旋转 180°
- 或者在 stage4 脚本里改 `normal_oriented` 那段（让 if 条件反向）

### 方向校正没效果

**原因**：你的花瓶是圆形对称的，PCA 没有明显主轴。

**处理**：`--no-orient` 跳过这步，圆瓶不需要方向校正

---

## 八、Stage 5 问题

### 模型有大洞

**原因**：泊松重建在缺数据的区域无法填补。

**处理**：
1. `--depth 10` 更精细的网格（但需要点云够密）
2. `--density-cut 0.05` 保留更多顶点
3. 如果是植物顶部 → 采集时改高度（相机往上抬）
4. 实在不行 → MeshLab 手动 "Close Holes"

### 模型一团毛刺

**原因**：低密度顶点没裁干净。

**处理**：`--density-cut 0.20` 更激进裁剪

### 模型表面有彩色噪点

**原因**：顶点颜色映射用最近邻，有些远点的颜色被采样到表面。

**处理**：Stage 3 后处理时调小 `voxel_final`，让点云更稀疏统一

---

## 九、整体调试套路

如果某一步出问题，**别死磕参数**。按这个顺序往回查：

1. **诊断深度数据**（`tool_diagnose_depth.py`）
2. **看标定 RMS**（`calibration.json`）
3. **抽几帧看 aligned 图**（肉眼判断对齐质量）
4. **抽几帧看 stage1 输出**（点云本身对不对）
5. **看 stage2 日志**（vertical_axis 一致吗）
6. **看 stage3 日志**（fitness 分布）

90% 的问题都能在前 3 步发现。

## 十、还是搞不定？

把这些发给我（或者贴到 issue）：
- 控制台完整输出
- 出问题的 stage 的输出文件截图（MeshLab 截图最好）
- `tool_diagnose_depth.py` 的输出
- `calibration.json` 的 RMS 部分

绝大多数问题都能从这些信息定位。
