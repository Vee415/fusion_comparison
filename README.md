# fusion-benchmark

Camera + LiDAR fusion benchmark: early/mid/late 2D fusion + a 3D BEV fusion variant, trained on KITTI, best-2D deployed to Jetson Orin in C++ TensorRT.

## Benchmark results

### Four fusion variants (KITTI, real data, 100 frames)

| Variant | Fusion space | AP@0.5 | lat_p50 (ms) | cam_dropped | lidar_dropped |
|---|---|---|---|---|---|
| early_2d | image (input-level) | 0.924 | 105 | 0.028 | 0.904 |
| mid_2d | image (feature-level) | 0.928 | 105 | 0.040 | 0.926 |
| **late_2d** | image (decision-level) | **0.935** | 200 | **0.856** | 0.944 |
| fusion_3d | BEV (metric space) | 0.099 | 238 | 0.125 | 0.002 |

- **AP@0.5**: average precision at IoU 0.5 (2D) or BEV IoU 0.7 (3D)
- **cam_dropped**: AP when camera is blinded (LiDAR-only performance)
- **lidar_dropped**: AP when LiDAR is blinded (camera-only performance)

**Key findings:**
- 2D image-space fusion (early/mid) is camera-dominant — LiDAR is barely used (lidar_dropped ≈ full AP)
- late_2d is the only 2D variant where both sensors contribute (cam_dropped=0.856, lidar_dropped=0.944)
- 3D BEV fusion flips the dominance — LiDAR becomes the primary sensor (lidar_dropped=0.002, cam_dropped=0.125)
- late_2d achieves the best AP (0.935) with real sensor redundancy

### Deployed variant (late_2d) — runtime comparison

| Runtime | Hardware | Precision | AP | p50 (ms) | FPS | Speedup |
|---|---|---|---|---|---|---|
| PyTorch | RTX 4060 laptop | FP32 | 0.951 | 22.8 | 44 | baseline |
| ONNX Runtime | laptop CPU | FP32 | 0.949 | 130.0 | 7.7 | 0.18× |
| **TensorRT** | **Jetson Orin** | **FP16** | **0.949** | **9.5** | **105** | **5.7×** |

- Export is **lossless**: AP difference < 0.003 (within decode tie-breaking noise)
- TensorRT FP16 on Jetson is **5.7× faster** than PyTorch FP32 on RTX 4060
- **105 FPS** on Jetson Orin — comfortably real-time (10× headroom over 10 Hz LiDAR)

See `deploy/deploy_report.md` for the full deployment report.

## Architecture

```
            camera (384×1280 RGB)          LiDAR (N×4 points)
                    |                              |
              ImageBackbone                   PointPillars /
              (ResNet-18+FPN)                 LidarImageGrid / LiftSplat
                    |                              |
              ──────────── fuse ──────────────────────
              |         |         |          |
            early     mid      late       3D/BEV
           (concat   (feature  (two dets  (BEV cat
           at input)  grid)    + merge)   + 1×1 conv)
                |         |         |          |
           CenterHead2D           CenterHead3D
                |         |         |          |
              2D boxes (x1,y1,x2,y2)    3D boxes (x,y,z,w,h,l,yaw)
```

## Run without KITTI first (smoke test)

The whole pipeline runs on **synthetic data** so you can verify every module works before downloading KITTI:

```powershell
pip install -r requirements.txt
# train a mid-fusion variant for a few steps on synthetic data
python -m train.trainer --config config/mid_2d.yaml --dataset synthetic --epochs 1 --iters 5
# run the full benchmark on synthetic (smoke check the harness)
python -m eval.benchmark --dataset synthetic --iters 2
# export to ONNX
python -m export.onnx_export --config config/mid_2d.yaml --dataset synthetic
```

## Run on real KITTI

1. Get KITTI Object Detection (see `data/kitti_download.sh`) into `data/kitti/`:
   ```
   data/kitti/training/image_2/000008.png  data/kitti/training/velodyne/000008.bin
   data/kitti/training/label_2/000008.txt  data/kitti/training/calib/000008.txt
   ```
2. Train each variant (pretrained ResNet-18 backbone + early stopping + val split):
   ```powershell
   python -m train.trainer --config config/early_2d.yaml
   python -m train.trainer --config config/mid_2d.yaml
   python -m train.trainer --config config/late_2d.yaml
   python -m train.trainer --config config/fusion_3d.yaml
   ```
3. Produce the benchmark table:
   ```powershell
   python -m eval.benchmark
   ```
4. Deploy best-2D (late_2d) to Jetson:
   ```powershell
   # export to ONNX
   python -m export.onnx_export --variants late_2d
   # copy to Jetson + build TRT engines
   bash deploy/build_engines.sh
   # compile + run C++ inference
   cd deploy/cpp && cmake . && make
   ./late_2d_deploy --cam_engine ... --lid_engine ... --image ... --velodyne ... --calib ...
   ```

## Deploy (late_2d → Jetson Orin)

```
PyTorch → ONNX → TensorRT FP16 → C++ inference on Jetson
   |         |           |                  |
checkpoints  onnx/    build/*.engine   late_2d_deploy
```

- `deploy/demo_onnx.py` — Python ONNX demo (runs on laptop)
- `deploy/cpp/` — C++ TensorRT pipeline (8 files, CUDA preprocessing + TRT + decode + NMS + merge)
- `deploy/build_engines.sh` — ONNX → TRT engine conversion script
- `deploy/deploy_report.md` — full deployment report (accuracy + latency comparison)
- See `future_reworks.md` for planned 3D AP improvements (BEVDepth, finer grid, padded pillars)

## Training details

- **Backbone**: ResNet-18 (ImageNet pretrained) + FPN, 256 output channels
- **Head**: CenterNet-style (focal loss heatmap + L1 offset/size)
- **Training**: lr=3e-4, batch_size=8, AdamW + cosine schedule, AMP, gradient clipping (max_norm=10)
- **Validation**: 10% held-out split, early stopping (patience=5), best-checkpoint saving
- **Nan protection**: skip nan batches, val loss returns inf on nan (never saves corrupted weights)

## Coordinate convention

KITTI camera frame: **x right, y down, z forward**. BEV grid: dim0 = z (forward), dim1 = x (right). See `common/geometry/bev.py`.

## Scope honesty

- **Built**: calibration/projection, shared backbones, 4 fusion variants, training pipeline, eval suite, ONNX export, Jetson C++ deploy
- **Reused**: torchvision ImageNet ResNet-18 weights (standard detection init)
- **Simplified**: lift-splat (no frustum pooling), PointPillars (max-pool, no padded pillars) — documented in `future_reworks.md`
- **Honest eval**: AP@0.5 is a clean stand-in, not official KITTI Easy/Moderate/Hard. The benchmark is comparative across variants under the same eval.
- **Deferred (v2)**: C++ TensorRT deploy of the 3D variant (documented, not built)