# Plant3D

[中文版 / Chinese Version](README_CN.md)

3D plant reconstruction pipeline using Kinect v2 depth camera and electric turntable.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Open3D](https://img.shields.io/badge/Open3D-≥0.15-green)
![OpenCV](https://img.shields.io/badge/OpenCV-≥4.5-blue?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-≥1.20-blue?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-≥1.7-blue?logo=scipy&logoColor=white)
![pymeshlab](https://img.shields.io/badge/pymeshlab-≥2022.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Captures multi-view depth images of a potted plant on a rotating turntable, then automatically generates a colored 3D model through point cloud registration, post-processing, and surface reconstruction.

## Pipeline

| Stage | Script | Description |
|:-----:|--------|-------------|
| 1 | `01_capture_calib.py` → `02_calibrate.py` | Camera calibration (Zhang's method + stereo calibration) |
| 2 | `03_capture_scan.py` | Turntable scanning with real-time RGB-depth alignment |
| 3 | `04_stage1_pcd_v5.py` | Depth map → colored point cloud, background removal |
| 4 | `05_stage2_segment_v3.py` | Crop plant above turntable |
| 5 | `06_stage3_register.py` | Multi-frame registration (Pose Graph + Color ICP) |
| 6 | `09_postprocess.py` | 9-step post-processing: SOR + ROR + downsample + Poisson + hole fill + smooth + normal fix |
| 7 | `10_upsample.py` | Point cloud upsampling (linear interpolation) |
| 8 | `12_hull_colored.py` | Concave hull wrapping with color projection |

## Quick Start

```bash
# 1. Calibrate camera (one-time setup)
python scripts/01_capture_calib.py --output calib_imgs
python scripts/02_calibrate.py --input calib_imgs --pattern 7x6 --square 25

# 2. Scan plant on turntable
python scripts/03_capture_scan.py --output capture_xxx --calib calib_imgs/calibration.json

# 3. Build point cloud
python scripts/04_stage1_pcd_v5.py --input capture_xxx
python scripts/05_stage2_segment_v3.py --input capture_xxx

# 4. Register multi-frame point clouds
python scripts/06_stage3_register.py --input capture_xxx --frame-range 5 123

# 5. Post-process and generate mesh
python scripts/10_upsample.py --input capture_xxx
python scripts/09_postprocess.py --input capture_xxx --source pcd_upsampled.ply \
    --poisson-depth 10 --density-cut 0.05 --normal-radius 12

# 6. Alternative: concave hull with color
python scripts/12_hull_colored.py --input capture_xxx --alpha 4.5 --outlier-pct 0.03 \
    --subdivide 2 --smooth 30 --color-radius 20
```

## Documentation

| Topic | Description |
|-------|-------------|
| [Calibration](docs/en/01_calibration.md) | Zhang's method, stereo calibration, quality verification |
| [Scanning](docs/en/02_capture.md) | Turntable setup, capture workflow, data quality checks |
| [Reconstruction](docs/en/03_reconstruction.md) | Stage-by-stage guide with parameter tuning |
| [Troubleshooting](docs/en/troubleshooting.md) | Common problems and solutions |

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---------|---------|
| numpy | >= 1.20 |
| opencv-python | >= 4.5 |
| open3d | >= 0.15 |
| scipy | >= 1.7 |
| pymeshlab | >= 2022.2 |
| pykinect2 | >= 0.1.0 |
| comtypes | — |

## Hardware

- **Depth Camera**: Kinect v2 (Microsoft)
- **Turntable**: Electric turntable with constant rotation speed
- **Scanning**: ~241 frames per 2 full rotations at 6 fps

## Key Techniques

- **Calibration**: Zhang's method for RGB/IR intrinsics + stereo calibration for extrinsics
- **Real-time Alignment**: Custom calibration-based depth-to-color mapping (not SDK defaults)
- **Registration**: Pose Graph with adjacent + skip + loop-closure edges, optimized via LM
- **Post-processing**: Multi-step pipeline with Poisson reconstruction, hole filling, and region-selective smoothing
- **Concave Hull**: Variable alpha shape (larger alpha for pot base, smaller for plant details) with orientation correction

## Project Structure

```
scripts/
├── capture_xxx/            # Scan data
│   ├── metadata.json       # Capture parameters
│   ├── timestamps.csv      # Frame timestamps
│   ├── color/ depth/       # Raw images
│   ├── pcds/ pcds_seg/     # Point clouds
│   └── output_v2/          # Output models
├── calib_imgs/             # Calibration images
│   └── calibration.json    # Calibration result
├── utils_kinect.py         # Utility library
└── 01-12_*.py              # Pipeline scripts
```

## Coordinate System

- Y-axis: vertical (up)
- Rotation: around Y-axis
- Units: millimeters (mm)
- Colors: RGB 0-1

## License

MIT
