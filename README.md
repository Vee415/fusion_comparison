# fusion-benchmark

Camera + LiDAR fusion benchmark: early/mid/late 2D fusion + a 3D BEV fusion variant, trained on a laptop, best-2D deployed to Jetson in C++ TensorRT. See `../fusion-benchmark-guide.md` for the full design, scope, and learning roadmap.

## Run without KITTI first (smoke test)

The whole pipeline runs on **synthetic data** so you can verify every module works before downloading 12 GB of KITTI:

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

1. Get KITTI Object Detection (see `data/kitti_download.sh`) into `data/kitti/` with the standard layout:
   ```
   data/kitti/image_2/000008.png  data/kitti/velodyne/000008.bin
   data/kitti/label_2/000008.txt  data/kitti/calib/000008.txt
   ```
2. Train each variant (one config each):
   ```powershell
   python -m train.trainer --config config/early_2d.yaml
   python -m train.trainer --config config/mid_2d.yaml
   python -m train.trainer --config config/late_2d.yaml
   python -m train.trainer --config config/fusion_3d.yaml
   ```
3. Produce the one benchmark table + robustness + scatter:
   ```powershell
   python -m eval.benchmark
   ```
4. Export and deploy best-2D to Jetson (see `export/` and `deploy_cpp/best_2d/`).

## Coordinate convention (assert everywhere)

KITTI camera frame: **x right, y down, z forward**. BEV grid: dim0 = z (forward), dim1 = x (right). Pick this and assert it — see `common/geometry/bev.py`.

## Scope honesty (from the guide)

Build: calibration/projection, shared backbones, the interface, variants A/B/C, best-2D Jetson deploy.
Reuse: OpenPCDet ops + KITTI devkit for 3D-AP (do not reinvent 3D eval).
Defer (v2): C++ TensorRT deploy of the 3D variant. Document it, don't half-build it.