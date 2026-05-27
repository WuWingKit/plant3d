# Scanning

> After calibration, each new plant only needs one scan.
> Scan quality determines the reconstruction ceiling — effort saved here costs 10x downstream.

## Physical Setup

### Turntable Preparation

**Kinect v2 cannot see black objects** — IR light is absorbed, depth returns 0.

| Modification | Necessity | Time |
|-------------|-----------|------|
| Cover turntable surface with A3 white paper | Required | 3 min |
| Draw crosshair on paper (for centering) | Recommended | 2 min |
| Raise paper edge with small cardboard | Optional | 1 min |

### Lighting

- **Diffuse lighting** (overhead lights)
- **Avoid direct light** causing harsh shadows
- **Avoid reflections** (bright floor/tabletop)

### Plant Placement

- Center on turntable
- Distance from Kinect: **1.2-1.8m** (optimal depth range)
- Plant height should be at or slightly below Kinect eye level

### Rotation Speed

- **15-30 seconds per rotation** is ideal
- Too fast → motion blur
- Too slow → redundant data, long capture time

## Capture

```bash
python scripts/03_capture_scan.py --output capture_xxx --calib calib_imgs/calibration.json --fps 8
```

### Parameters

- `--fps 8` — 8 frames/sec. 15s/rotation → 120 frames. Sufficient.
- `--fps 10` — For denser sampling
- `--fps 5` — Lightweight (60 frames still works)

### Operation

1. Start Kinect, let it warm up for 10 seconds
2. Launch the script, **wait for preview window**
3. **Preview window panels**:
   - **Top-left**: Raw color (1920x1080)
   - **Top-right**: Depth pseudo-color (red=near, blue=far)
   - **Bottom-left**: Aligned RGB (color mapped to depth coordinates)
   - **Bottom-right**: Aligned overlay (depth edges + color) — **real-time alignment quality check**
4. Start turntable rotation
5. **Wait for stable rotation** → press SPACE to start recording
6. Record 1-2 full rotations
7. Press SPACE to stop
8. Program asks "how many rotations / how many seconds" — estimate is fine

### Real-time Quality Check

The bottom-right panel is critical:

- **Aligned RGB and depth edges should match** (check plant outline)
- If color and shape misalign by > 2cm, **calibration is wrong** — stop and re-calibrate
- Info bar shows:
  - `Depth valid: XX%` — should be 30-60% (rest is black turntable/background, normal)
  - `median: XXmm` — should match your camera-to-plant distance (e.g., 1500mm)

## Post-capture Data Structure

```
capture_xxx/
├── color/        # Raw color (1920x1080 BGR uint8 PNG)
│   ├── 0000.png
│   └── ...
├── depth/        # Raw depth (512x424 uint16 mm PNG)
│   ├── 0000.png
│   └── ...
├── aligned/      # ★ Aligned RGB (512x424 BGR uint8 PNG)
│   ├── 0000.png  # Each pixel corresponds to depth at same coordinate
│   └── ...
├── timestamps.csv
└── metadata.json
```

`aligned/` is **the core innovation of this pipeline**. Reconstruction reads `aligned[v, u]` directly for color — zero alignment error.

## Post-capture Health Check

Quick Python check:

```bash
python -c "
import cv2, numpy as np, glob
files = sorted(glob.glob('capture_xxx/depth/*.png'))
print(f'Frames: {len(files)}')
img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
print(f'Format: dtype={img.dtype}, shape={img.shape}')
d = img.astype(float); d[d==0] = np.nan
print(f'Median depth: {np.nanmedian(d):.0f}mm, valid: {np.count_nonzero(~np.isnan(d))/d.size*100:.1f}%')
"
```

| Metric | Normal | Action if abnormal |
|--------|--------|--------------------|
| Frame count | 120+ per rotation | Record longer |
| Median depth | 1200-1800mm | Adjust camera position |
| Valid pixels | 30-60% | < 20%: too many black objects; > 80%: too empty |
| Format | uint16 1ch | 8-bit/3ch: corrupted data, re-capture |

## Typical Scenarios

### Standard (Recommended)
- Distance: 1.5m, Speed: 22s/rotation, FPS: 8, 2 rotations = 44s = ~350 frames

### Quick
- Distance: 1.2m, Speed: 15s/rotation, FPS: 8, 1 rotation = 120 frames

### High Density
- Distance: 1.5m, Speed: 30s/rotation, FPS: 10, 2 rotations = 600 frames

## Post-scan Checklist

Open `aligned/0000.png` and verify:

- [ ] Plant + pot clearly visible
- [ ] No large black areas (except turntable)
- [ ] Colors match reality (not shifted)
- [ ] No strong glare / overexposure at edges

If any check fails, **don't proceed** — fix and re-capture.

## Next

Once data looks good, see [03_reconstruction.md](03_reconstruction.md) for the reconstruction stage.
