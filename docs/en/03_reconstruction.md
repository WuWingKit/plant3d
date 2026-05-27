# Reconstruction

> Run stages sequentially. Each stage has visual intermediate outputs.
> When tuning parameters, only re-run the failed stage — previous intermediates are reusable.

## Overview

```
Raw data (depth/ + aligned/)
       ↓
Stage 1: Single-frame colored point cloud  →  pcds/0000.ply ...
       ↓ ✓ Spot-check in MeshLab
Stage 2: Foreground separation             →  pcds_seg/0000.ply ...
       ↓ ✓ Verify only plant + pot remains
Stage 3: Pose Graph registration           →  merged_*.ply
       ↓ ✓ Check merged_color_coded.ply
Stage 4: Alignment                          →  merged_aligned.ply
       ↓ ✓ Verify plant is upright
Stage 5: Post-processing + Poisson         →  plant_model.{obj,stl,ply}
```

---

## Stage 1: Single-frame Point Cloud

```bash
python scripts/04_stage1_pcd_v5.py --input capture_xxx --calib calib_imgs/calibration.json
```

**Purpose**:
- Read each frame's `depth/XXXX.png` + `aligned/XXXX.png`
- Back-project depth pixels to 3D points using calibrated IR intrinsics
- Assign color from aligned RGB (zero alignment error)
- Output per-frame PLY files + quality report HTML

**Parameters**:
- `--n 36` — Uniformly sample 36 frames (default)
- `--every 18` — Take every 18th frame
- `--all` — Process all frames (~633 frames, 5-10 min)
- `--depth-trunc 3000` — Depth cutoff in mm (default 3000)

**Verification**:
1. Open `pcds/quality_report.html` — point count should be 30,000-100,000
2. Open `pcds/0000.ply`, `pcds/0018.ply`, `pcds/0035.ply` in MeshLab
   - Should show plant + pot + some background
   - Colors and shapes should be consistent

---

## Stage 2: Foreground Separation

```bash
python scripts/05_stage2_segment_v3.py --input capture_xxx
```

**Purpose**: Pure geometric separation of plant + pot

Algorithm:
1. RANSAC finds the largest horizontal plane → tabletop/turntable surface
2. Auto-detect vertical axis
3. Points above plane = foreground candidates
4. DBSCAN clustering, keep largest cluster
5. HSV filter for remaining black points (turntable residue)

**Parameters**:
- `--dist-thresh 8` — Plane fitting distance threshold (default 8mm)
- `--eps 15` — DBSCAN neighborhood radius
- `--min-points 200` — Minimum cluster size
- `--drop-dark-v 40` — V < 40 treated as black

**Verification**: Open `pcds_seg/0000.ply`:
- Only plant + pot remains
- No table/turntable/background

---

## Stage 3: Pose Graph Registration (Core)

```bash
python scripts/06_stage3_register.py --input capture_xxx
```

**Purpose**: Register multi-frame point clouds into a unified coordinate system

**Key design: Pure 6-DOF Pose Graph**:
- Each frame has 6-DOF pose (3 rotation + 3 translation)
- **No rotation axis assumption** (works with handheld capture too)
- Global optimization of all poses; single edge failure is not fatal

**Registration method** (per frame pair):
1. Downsample + estimate normals + compute FPFH features
2. FPFH + RANSAC coarse alignment
3. Multi-scale Point-to-Plane ICP geometric refinement
4. **Color ICP** color refinement

**Pose Graph edge types**:
- **Adjacent** `i ↔ i+1` — links all frames
- **Skip** `i ↔ i+2` — adds redundancy
- **Loop closure** — head-to-tail, opposing positions

**Parameters**:
- `--voxel 8` — Downsampling voxel for registration (mm)
- `--loop-closures 3` — Number of loop closure edges
- `--no-color-icp` — Disable Color ICP (geometry only)

**Verification**:
1. Open `merged_color_coded.ply` — **most important check**
   - **View from top (plan view)**
   - Different-colored frames should overlap into a clear plant outline
   - If petals → registration failed
2. Check `registration_log.csv` — success rate > 70% is good

---

## Stage 4: Alignment

```bash
python scripts/07_stage4_align.py --input capture_xxx
```

**Purpose**: Orient the plant — pot base on XY plane, pot edge aligned to X axis

**Parameters**:
- `--low-percent 15` — Lowest N% for pot base detection
- `--dist-thresh 5` — Plane distance threshold (mm)
- `--no-orient` — Skip orientation correction (only level)

**Verification**: Open `merged_aligned.ply`:
- Plant is upright (pot base on Z=0, plant grows in +Z)
- Bounding box Z range = actual plant height

---

## Stage 5: Post-processing + Poisson Reconstruction

```bash
python scripts/10_upsample.py --input capture_xxx
python scripts/09_postprocess.py --input capture_xxx --source pcd_upsampled.ply \
    --poisson-depth 10 --density-cut 0.05 --normal-radius 12
```

**Purpose**: Generate mesh from point cloud with 9-step post-processing

**Pipeline**:
1. SOR statistical outlier removal
2. ROR radius outlier removal
3. Voxel downsampling
4. RANSAC plane removal (optional)
5. Poisson reconstruction
6. Hole filling (pymeshlab)
7. Smoothing (Laplacian / Taubin)
8. Region-selective gradient smoothing
9. Remove isolated fragments

**Parameters**:
- `--poisson-depth 10` — Poisson depth (default 9)
- `--density-cut 0.05` — Low-density vertex cut ratio (default 0.02)
- `--normal-radius 12` — Normal estimation radius (mm)
- `--fill-hole-size 200` — Max hole boundary edge count
- `--smooth-method pymeshlab_laplacian` — Smoothing method
- `--taubin-iter 25` — Smoothing iterations

**Output**:
- `output_v2/plant_model.obj` — With vertex colors
- `output_v2/plant_model.stl` — Geometry only
- `output_v2/plant_model.ply` — With vertex colors

**Verification**: Open `plant_model.obj` in MeshLab:
- Complete model, no large holes
- Reasonable colors
- Smooth surface, no fuzz

---

## Debugging Flowchart

```
           ┌──────────┐
           │ Stage 1  │
           └────┬─────┘
                ↓
            ┌───┴────┐
            │ Check  │── Color wrong? → Re-calibrate
            │ pcds/  │── Shape wrong? → Check depth data
            └───┬────┘
                ↓ OK
           ┌────┴─────┐
           │ Stage 2  │
           └────┬─────┘
                ↓
            ┌───┴────┐
            │ Check  │── Cut too much? → eps↑ / dist-thresh↑
            │ seg/   │── Left too much? → eps↓ / min-points↑
            └───┬────┘
                ↓ OK
           ┌────┴─────┐
           │ Stage 3  │
           └────┬─────┘
                ↓
            ┌───┴────────┐
            │ Check      │── Petals? → Check log, adjust voxel
            │ coded      │── Many failures? → Go back to Stage 2
            └───┬────────┘
                ↓ OK
           ┌────┴──────┐
           │ 09 Post   │
           └────┬──────┘
                ↓
           ┌────┴──────┐
           │ Check     │── Holes? → depth↑
           │ model     │── Fuzz? → density-cut↑
           └───────────┘
```

## Typical Runtimes

| Stage | 36 frames | 120 frames | 633 frames (--all) |
|-------|-----------|------------|---------------------|
| Stage 1 | 30 sec | 1.5 min | 8 min |
| Stage 2 | 1 min | 3 min | 15 min |
| Stage 3 | 2-5 min | 10-20 min | (not recommended) |
| Stage 4 | 10 sec | 10 sec | 10 sec |
| Post-process | 30 sec | 1 min | 2 min |
| **Total** | **5-8 min** | **15-25 min** | — |

**Recommended: start with 36 frames**, validate the workflow, then increase.
