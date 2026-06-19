# Session 1 — 19 June 2026

Camera + LiDAR fusion benchmark project. New-grad scoped learning roadmap with
all four fusion variants implemented and runnable end-to-end on synthetic data.

## What we did this session

### 1. Guide rewrite
Rewrote `../fusion-benchmark-guide.md` from an aspirational senior delivery plan
into a new-grad-scoped learning roadmap:
- Removed all `§` symbols (`§N` → `Section N`).
- Kept all 3 two-dimensional fusion modes (early / mid / late) + the 3D BEV variant intact.
- Added a build-vs-study-vs-defer honesty table (Section 1) — what to build yourself
  vs reuse commodity libs (OpenPCDet, KITTI devkit) vs defer to senior infra.
- Understanding-gated milestones (Section 8) with a "stop-after-7 = shippable checkpoint".
- Drone on-ramp (Section 13): VIO / VINS-Fusion / OpenVINS on EuRoC as phase 2/3,
  framed as the missing temporal/state-estimation piece for drone perception.

### 2. Full implementation (6 dependency-ordered layers)
Everything under `fusion-benchmark/`. Runnable WITHOUT KITTI via a synthetic loader,
so the whole pipeline (train / eval / export / benchmark) can be verified before
downloading 12 GB of data.

- `requirements.txt`, `README.md` (synthetic smoke + real-KITTI flow + coordinate
  convention + scope honesty), `.gitignore`.
- `common/config.py` (load_config merges base + variant yaml; set_seed).
- `config/` — base.yaml + per-variant yamls (early_2d, mid_2d, late_2d, fusion_3d).
- `common/sensors/calibration.py` — `Calib` (from_file / from_arrays / synthetic,
  velo_to_cam, cam_to_image, torch_matrices with per-device cache).
- `common/sensors/projection.py` — numpy + torch projection / depth-image / BEV helpers.
- `common/geometry/` — boxes2d (CenterNet encode/decode, iou, nms), boxes3d
  (CenterPoint encode/decode, rotated BEV iou via shapely, rotated nms), bev.
- `common/backbones/` — image_backbone (ResNet-18 + FPN, in_channels param),
  lidar_encoder (PointMLP, LidarImageGridEncoder, PillarBEVEncoder).
- `fusion/` — base.py (FusionModel abstract + blind flags), heads.py (CenterHead2D/3D),
  factory.py, common_2d.py, common_3d.py, and the 4 variant models.
- `train/` — trainer.py (one trainer, AMP, AdamW, cosine), losses_2d.py, losses_3d.py.
- `data/loaders/` — collate.py, synthetic_loader.py, paired_loader.py (KITTI),
  kitti_download.sh.
- `eval/` — infer.py, metrics_2d.py, metrics_3d.py (KITTI-devkit wrapper),
  robustness.py (blind eval), benchmark.py (one table + CSV).
- `export/onnx_export.py`.
- `tests/` — test_projection.py, test_smoke.py.
- `notebooks/01_lidar_visual_sanity_check.ipynb` (milestone-1 visual tool).

### 3. Environment setup
Cloned conda env: `conda create -n fusion --clone gpu_base` (torch 2.5.1 + CUDA,
RTX 4060). Installed requirements + pytest.

### 4. Debugging — three AMP dtype bugs (the key learning of the session)
All three only surfaced under autocast (the trainer path), NOT in the no-AMP unit
tests. Lesson: always smoke-test the trainer under AMP, not just forward passes.

1. `early_2d.export_onnx` — `torch.onnx.export(self, dummy)` passed a raw tensor to
   a dict-expecting forward. Fixed with a `_Wrap` module (backbone + head).
2. `lidar_encoder` scatter_reduce dtype mismatch — MLP output was fp16 under autocast
   while scatter grids were fp32. Fixed with `feats = self.mlp(feat_in).float()`
   in both LidarImageGridEncoder and PillarBEVEncoder.
3. early/late depth-image fill — `lidar_to_image_torch` matmuls got promoted to fp16
   under autocast, so depth was fp16 while the depth image was fp32. Fixed by wrapping
   the projection core in `torch.amp.autocast(..., enabled=False)` and forcing fp32
   (it is fixed geometry / preprocess).
Plus: mid_2d and fusion_3d `cat` dtype mismatches (`f_lid.to(f_cam.dtype)`,
bev_lid cast to bev_cam dtype); late_2d malformed walrus line removed.

### 5. Verified final state
- Unit tests: 6 passed, 1 skipped (KITTI overlay — no data).
- All 4 variants train + save checkpoints under AMP.
- ONNX export works for all 4 variants.
- Benchmark table renders with real latency (mid_2d ~24ms, fusion_3d ~32ms,
  late_2d ~47ms on RTX 4060). AP = 0.000 on synthetic is expected (random noise,
  untrained / 1-epoch).

## What to do next session

### Immediate (verify on real data)
- [ ] Download KITTI Object Detection (12 GB) via `data/loaders/kitti_download.sh`,
      confirm `paired_loader.py` loads a real frame, and re-run the KITTI-overlay
      test (currently skipped).
- [ ] Train mid_2d (the likely deploy winner) on real KITTI for the full 60 epochs
      and get a non-zero AP@0.5. This is the first real milestone number.
- [ ] Sanity-check `notebooks/01_lidar_visual_sanity_check.ipynb` on a real frame —
      confirm projection overlay and BEV look correct before trusting any AP.

### Then (the actual benchmark)
- [ ] Train all 4 variants on real KITTI with identical settings; record AP + latency
      in `benchmark_results.csv` (already wired via `eval/benchmark.py`).
- [ ] Run robustness eval (`eval/robustness.py`): cam-blinded vs lidar-blinded
      graceful degradation per variant — this is the whole point of comparing
      fusion modes.
- [ ] Compare early vs mid vs late vs 3D — write up which mode wins on AP, latency,
      and robustness, and why (this is the learning outcome).

### Deploy on-ramp (drone-relevant)
- [ ] Export the best 2D variant to ONNX and run on Jetson via TensorRT (matches
      your existing Jetson C++ / TensorRT / INT8 strengths). This is the Tier-1
      deploy skill and the bridge from this project to your drone work.
- [ ] Document the deploy-friendly input signatures: mid_2d takes a pre-scattered
      lidar feature map (scatter done in C++/CUDA at deploy); fusion_3d exports
      only fuse + head (lift-splat / pillarize are CUDA preprocess).

### Open / offered but not confirmed
- [ ] Git repo + baseline commit (offered end of session, not yet done — user was
      focused on understanding the tests / context settings). Confirm before
      starting to change things so there's a clean versioned starting point.

### Carry-over notes
- `lr_scheduler "step before optimizer.step"` warning in trainer — benign, not fixed.
- Simplified lift-splat (fusion_3d) is for learning, not publication numbers; for
  real 3D-AP reuse the KITTI devkit via `eval/metrics_3d.py::evaluate_official`.
- Conda env name: `fusion`. Run everything in it.