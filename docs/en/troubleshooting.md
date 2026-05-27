# Troubleshooting

Organized as "Symptom → Possible cause → Fix". Use Ctrl+F to search for your symptom.

---

## 1. Environment Issues

### PyKinect2 import error after install

**Symptom**: Crash when importing PyKinect2, `sizeof` assertion failure or `time.clock` not found.

**Fix**:
1. Find `<env>/Lib/site-packages/comtypes/_tlib_version_checker.py`, comment out:
   ```python
   # assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)
   ```
2. Find `<env>/Lib/site-packages/pykinect2/PyKinectRuntime.py`, replace all:
   ```
   time.clock()  →  time.perf_counter()
   ```

### Kinect not responding

**Symptom**: `kinect.has_new_color_frame()` always returns False.

**Checklist**:
- [ ] Kinect v2 must be on USB 3.0 port (blue)
- [ ] Kinect SDK 2.0 installed (not just PyKinect2)
- [ ] Power cable connected
- [ ] Kinect Studio can see the device
- [ ] No other program using Kinect (close Skype/camera apps)

---

## 2. Calibration Issues

### IR corner detection success rate low (< 50%)

**Cause**: Difficult IR detection under speckle, insufficient lighting, or glare.

**Fix**:
1. Add more lighting (multiple lights, near window)
2. Move board closer (1.0-1.5m range)
3. Ensure board is flat (mount on cardboard)
4. Keep tilt angle < 45°
5. Last resort: LED flashlight from the side

### Stereo RMS > 2px

**Cause**: Poor single-camera intrinsics or insufficient pairs.

**Fix**:
1. Check which camera (RGB or IR) has higher RMS — that's the culprit
2. Need at least 30 valid pairs
3. Verify board flatness (diagonal measurement ±0.5mm)

### T vector abnormal (not near [-52, 0, 0])

**Cause**: Calibration error, possibly board rotation direction reversed.

**Fix**: Check that stereoCalibrate source/target order is `imgpoints_ir → imgpoints_color`. If T is [+52, 0, 0], the order is reversed.

---

## 3. Capture Issues

### Aligned RGB misaligned in preview

**Symptom**: Bottom-right overlay shows color and depth edges clearly offset.

**Cause**: Bad calibration. **Fix**: Re-calibrate. This error propagates to Stage 1.

### No metadata.json after capture

**Fix**: Create manually:
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
Values can be approximate — Pose Graph doesn't depend on them.

### Depth images all black

**Cause**: Likely a data format issue.

**Fix**: Quick check:
```bash
python -c "import cv2; img=cv2.imread('capture_xxx/depth/0000.png',cv2.IMREAD_UNCHANGED); print(f'dtype={img.dtype}, shape={img.shape}')"
```
- uint16 → Normal (depth looks black before normalization)
- uint8 → **Corrupted data, re-capture**

---

## 4. Stage 1 Issues

### Too few points (< 5000)

**Cause**: Distance out of range or too many black objects.

**Fix**: Adjust `--depth-trunc 3500` or improve capture environment.

### Color misalignment

**Cause**: Calibration issue or aligned computation error.

**Fix**: Check `aligned/0000.png` visually. If misaligned → re-calibrate.

---

## 5. Stage 2 Issues

### "vertical_axis inconsistent" warning

**Cause**: Different frames RANSAC found different horizontal planes.

**Fix**: Increase `--dist-thresh 12`, check stage1 output for outlier frames.

### Plant cut in half

**Cause**: DBSCAN split plant into two clusters.

**Fix**: `--eps 20` for looser clustering, `--dist-thresh 12` for larger plane threshold.

### Black turntable residue

**Cause**: HSV filter too loose.

**Fix**: `--drop-dark-v 60` removes more dark points. Warning: too aggressive cuts plant shadows.

---

## 6. Stage 3 Issues

### Registration failure rate > 50%

**Cause**: Point clouds too sparse (weak FPFH features) or plant too smooth.

**Fix**:
1. Go back to Stage 2, use `--eps 20` to keep more context
2. Try `--voxel 6` (finer) or `--voxel 12` (sparser, more stable features)
3. Try `--no-color-icp` to check if color ICP hurts
4. Increase `--loop-closures 5`

### merged_color_coded.ply shows petals

**Cause**: Complete registration failure — each frame stayed at camera origin.

**Fix**: If RANSAC fitness is all below 0.1, FPFH is not working. Keep more points from Stage 2 or use denser sampling.

### Pose Graph optimization crashes

**Cause**: Loop closure edges misaligned, pulling global solution off.

**Fix**: `--loop-closures 1` (head-to-tail only). Current script auto-excludes low-fitness edges.

---

## 7. Stage 4 Issues

### Plant tilted

**Fix**: Adjust `--low-percent 20` to use more pot base points.

### Plant upside down

**Fix**: Check `align_transform.json` `base_normal`. Manually rotate 180° in MeshLab.

---

## 8. Post-processing Issues

### Model has large holes

**Fix**: Increase `--poisson-depth 10`, decrease `--density-cut 0.05`.

### Model has fuzz/bristles

**Fix**: Increase `--density-cut 0.15` for more aggressive pruning.

### Color noise on surface

**Cause**: Nearest-neighbor color sampling picks up distant noisy points.

**Fix**: Use denser point cloud (reduce voxel size in earlier stages).

---

## 9. Debugging Workflow

Don't brute-force parameters. Follow this order:

1. **Check depth data** (dtype, shape, valid pixel %)
2. **Check calibration RMS** (`calibration.json`)
3. **Visual check aligned images** (alignment quality)
4. **Spot-check stage1 output** (point cloud correctness)
5. **Check stage2 log** (vertical_axis consistency)
6. **Check stage3 log** (fitness distribution)

90% of problems are found in the first 3 steps.

## Still Stuck?

Provide these (or post as an issue):
- Full console output
- Screenshot of the problematic stage output (MeshLab screenshot preferred)
- Depth image dtype/shape/valid pixel %
- `calibration.json` RMS section
