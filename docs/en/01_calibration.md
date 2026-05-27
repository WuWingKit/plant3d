# Camera Calibration

> Calibration is the foundation of the entire pipeline. Calibration errors are amplified by every subsequent step. **Investing time here pays off**.

## Why Joint Calibration?

Kinect v2 has **two independent cameras**:

| Camera | Position | Resolution | FOV |
|--------|----------|------------|-----|
| RGB Color | Left side | 1920x1080 | 84° x 54° |
| IR Depth | Right side | 512x424 | 70° x 60° |

The physical distance between them is ~52mm. Two images taken simultaneously have **no pixel correspondence**. To align depth with color (or vice versa), you need:

1. **RGB intrinsics** (focal length, principal point, distortion)
2. **IR intrinsics** (same)
3. **Relative pose** (R rotation, T translation) between the two cameras

Kinect v2 SDK's `MapDepthFrameToColorSpace` uses **factory defaults** — each device varies slightly. **Custom calibration reduces alignment error from 1-2cm to sub-millimeter**.

## Zhang's Method

Zhang Zhengyou's 2000 paper [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/) is the algorithm behind OpenCV's `calibrateCamera`.

**Core idea**: Observe a planar checkerboard from multiple poses. Use the homography H (board plane → image plane) at each pose to solve for intrinsics K, then refine everything with nonlinear optimization.

**Stereo calibration**: Given N synchronized image pairs, find each camera's extrinsics [R_c, t_c] and [R_d, t_d], then compute the inter-camera relationship:

```
R = R_c * R_d^(-1)       # IR → RGB rotation
T = t_c - R * t_d        # IR → RGB translation
```

OpenCV's `cv2.stereoCalibrate` performs least-squares + LM optimization across all pairs.

## Step 1: Prepare the Checkerboard

- **7x6 inner corners** (8x7 squares)
- **Square size: 25mm** (measure precisely, pass via `--square`)
- **Flat**: Mount on rigid cardboard/aluminum. Paper flex ruins calibration.
- **Matte print**: Glossy surfaces cause glare; IR corners disappear.

## Step 2: Capture Checkerboard Pairs

```bash
python scripts/01_capture_calib.py --output calib_imgs --pattern 7x6
```

**Operation**:
1. Start Kinect, hold board in front
2. Real-time detection:
   - **Green cross** = both RGB and IR detected
   - **Yellow hint** = only RGB detected (IR may be too dark/tilted)
   - **Red hint** = neither detected
3. Wait 1 second for stabilization + both detected → press SPACE
4. Adjust board position/pose, repeat

**Capture strategy** (50-60 pairs total):

| Distance | Count | Purpose |
|----------|-------|---------|
| 0.8m | 12-15 | Near-field distortion |
| 1.2m | 12-15 | Medium distance |
| 1.6m | 12-15 | Close to scanning distance |
| 2.0m | 12-15 | Far-field distortion |

At each distance, **vary the pose**: frontal, tilted left/right, up/down, rotated 0°/45°/90°, at different screen positions.

## Step 3: Run Joint Calibration

```bash
python scripts/02_calibrate.py --input calib_imgs --pattern 7x6 --square 25
```

**Output**: `calib_imgs/calibration.json`

**What it does**:
1. Detect RGB and IR corners in each pair (OpenCV `findChessboardCornersSB`)
2. Keep only pairs where both sides detected (usually 65-80%)
3. RGB intrinsics calibration (`cv2.calibrateCamera`)
4. IR intrinsics calibration (same)
5. Stereo calibration (`cv2.stereoCalibrate` + `CALIB_FIX_INTRINSIC`)

## Step 4: Verify Quality

| Metric | Value | Rating |
|--------|-------|--------|
| RGB RMS | < 0.3 px | Excellent |
| RGB RMS | 0.3 - 0.5 px | Very good |
| RGB RMS | 0.5 - 1.0 px | Usable |
| RGB RMS | > 1.0 px | Redo |
| IR RMS | < 0.5 px | Excellent (for speckle) |
| IR RMS | 0.5 - 0.8 px | Very good |
| IR RMS | 0.8 - 1.2 px | Usable |
| IR RMS | > 1.2 px | Redo |
| Stereo RMS | < 0.8 px | Excellent |
| Stereo RMS | 0.8 - 1.5 px | Usable |
| Stereo RMS | > 1.5 px | Redo |

**Extrinsic check**: T (mm) should be near **[-52, 0, 0]** (Kinect physical layout). T_x in [-55, -49] is normal. T_y, T_z in [-5, 5] is normal.

## Common Failure Modes

**RGB detection works, IR fails** → Increase lighting, move board closer (< 1.5m), use matte print.

**Both fail** → Board not flat enough, too tilted (> 60°), too far (> 2m), or too small (< 1/4 of frame).

**RMS too high (> 1px)** → Board damaged, camera moved during capture, add more close-range samples.

## Calibration is Reusable

`calibration.json` captures your device's "factory parameters". **As long as you don't drop, disassemble, or swap the Kinect**, this calibration lasts a year or more. Re-calibrate only if the device has been physically disturbed.

## Next

Once calibration passes, see [02_capture.md](02_capture.md) to start scanning.
