# Cleanup Plan (for review — do NOT execute until approved)

## 1. Delete KITTI zip files (free ~39 GB)

The raw zip archives are no longer needed — the data is extracted to
`data/kitti/training/{image_2,velodyne,calib,label_2}/`.

```bash
rm data/kitti/image_2.zip      # 12 GB
rm data/kitti/velodyne.zip      # 27 GB
rm data/kitti/calib.zip         # 26 MB
rm data/kitti/label_2.zip       # 5 MB
```

**Saves ~39 GB.** The extracted data in `data/kitti/training/` stays.

### Also delete the intermediate `testing/` split (if not needed)

The script downloaded both training and testing splits. Testing split has no labels
(labels are withheld for the KITTI leaderboard). If you only train/eval on the training
split, you can delete testing:

```bash
rm -rf data/kitti/testing/      # ~saves 13 GB (images + velodyne)
```

**Total disk freed: ~52 GB** (39 GB zips + 13 GB testing split)

### Verify before deleting

```bash
# confirm training data is intact (should show 7481 each)
ls data/kitti/training/image_2/ | wc -l    # expect 7481
ls data/kitti/training/velodyne/ | wc -l   # expect 7481
ls data/kitti/training/calib/ | wc -l      # expect 7481
ls data/kitti/training/label_2/ | wc -l    # expect 7481
```

## 2. Clean up temporary files

```bash
# training logs (keep for reference or delete)
rm data/early_2d_train.log
rm data/mid_2d_train.log
rm data/late_2d_train.log
rm data/fusion_3d_train.log
rm data/kitti_download.log

# temporary scripts (already removed, verify)
ls notebooks/_fetch_kitti_sample.py 2>/dev/null   # should not exist
ls notebooks/_build_05.py 2>/dev/null              # should not exist
ls notebooks/_edit_04.py 2>/dev/null               # should not exist
ls notebooks/_viz_3d.py 2>/dev/null                 # still exists — delete or keep?
```

## 3. Git: commit the new work

### Uncommitted changes (from git status)

**Modified files:**
- `common/geometry/boxes2d.py` — int(labels) fix
- `common/geometry/boxes3d.py` — int(labels) fix + docstring/code mismatch note
- `config/base.yaml` — pretrained_backbone, epochs, lr, batch_size, val_fraction, early stopping, data_root
- `session_1_1906.md` — session notes update

**New files (untracked):**
- `LEARNING_FUSION.md` / `LEARNING_FUSION.html` — learning material
- `future_reworks.md` — 3D improvement roadmap
- `notebooks/02_frames_calibration_projection.ipynb`
- `notebooks/03_early_mid_late_fusion.ipynb`
- `notebooks/04_bev_fusion_3d.ipynb` — reviewed + fixed + 2D-vs-3D comparison added
- `notebooks/05_point_cloud_basics.ipynb` — new, real KITTI point cloud basics
- `notebooks/fusion_3d_preprocessing.png` — viz
- `notebooks/viz_bev_vs_perspective.py` / `.png`
- `deploy/` — full deploy directory (demo_onnx.py, cpp/, build_engines.sh, deploy_report.md, runtime_comparison.md)
- `data/kitti_sample/` — 5 KITTI frames for notebook 05
- `data/early_2d_train.log` etc. — training logs

### Suggested commit structure

```bash
# Stage modified files
git add common/geometry/boxes2d.py common/geometry/boxes3d.py config/base.yaml

# Commit the training infra + config changes
git commit -m "Add pretrained backbone, grad clip, val split, early stopping, best checkpoint saving"

# Stage new model + trainer changes
git add fusion/early_2d/model.py fusion/mid_2d/model.py fusion/late_2d/model.py fusion/fusion_3d/model.py
git add train/trainer.py export/onnx_export.py eval/benchmark.py

# Commit the deploy pipeline
git add deploy/
git commit -m "Add late_2d deploy: ONNX export, C++ TensorRT pipeline, Python demo, deploy report"

# Stage notebooks + learning material
git add notebooks/02_frames_calibration_projection.ipynb notebooks/03_early_mid_late_fusion.ipynb
git add notebooks/04_bev_fusion_3d.ipynb notebooks/05_point_cloud_basics.ipynb
git add notebooks/fusion_3d_preprocessing.png notebooks/viz_bev_vs_perspective.py notebooks/viz_bev_vs_perspective.png
git add LEARNING_FUSION.md LEARNING_FUSION.html future_reworks.md
git commit -m "Add notebooks 02-05, learning material, future reworks doc"

# Stage sample data
git add data/kitti_sample/
git commit -m "Add 5-frame KITTI sample for notebook 05"

# Stage the rest
git add session_1_1906.md
git commit -m "Update session notes"
```

### .gitignore additions

```
# Add to .gitignore — don't commit large/temp files
data/kitti/
data/kitti_sample/   # or keep this one, it's only 41 MB
checkpoints/*.pt      # or keep best checkpoints (55-109 MB each)
*.log
onnx/
__pycache__/
.pytest_cache/
```

## 4. Checkpoint management

| Checkpoint | Size | Keep? |
|---|---|---|
| early_2d_best.pt | 55 MB | ✅ best model for early fusion |
| early_2d.pt | 55 MB | ❌ final epoch, redundant (delete) |
| mid_2d_best.pt | 55 MB | ✅ best model for mid fusion |
| mid_2d.pt | 55 MB | ❌ redundant (delete) |
| late_2d_best.pt | 109 MB | ✅ **deploy model** — keep |
| late_2d.pt | 109 MB | ❌ redundant (delete) |
| fusion_3d_best.pt | 54 MB | ✅ best 3D model |
| fusion_3d.pt | 54 MB | ❌ redundant (delete) |

```bash
# Delete redundant final checkpoints (keep _best only)
rm checkpoints/early_2d.pt
rm checkpoints/mid_2d.pt
rm checkpoints/late_2d.pt
rm checkpoints/fusion_3d.pt
# Saves ~273 MB
```

## 5. Jetson cleanup (optional)

On the Jetson, the ONNX files (106 MB total) can be deleted after engines are built:

```bash
ssh vee@192.168.55.1 "rm ~/fusion_deploy/onnx/*.onnx"   # saves 106 MB on Jetson
```

## Summary of what gets cleaned

| Item | Space freed |
|---|---|
| KITTI zips | ~39 GB |
| KITTI testing split | ~13 GB |
| Redundant checkpoints | ~273 MB |
| Training logs | ~few MB |
| Jetson ONNX files | ~106 MB |
| **Total** | **~52 GB** |

## What gets kept

| Item | Size | Why |
|---|---|---|
| data/kitti/training/ | ~39 GB | real KITTI data for training/eval |
| data/kitti_sample/ | 41 MB | notebook 05 sample data |
| checkpoints/*_best.pt | ~273 MB | trained model weights |
| deploy/ | ~few MB | deploy code + report |
| notebooks/ | ~few MB | learning material |
| onnx/ | 106 MB | ONNX engines (for laptop demo) |