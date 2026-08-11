# Future Reworks

Documented improvements for a v2 of the fusion-3d variant. The current 3D AP (0.099)
is honest but low — these are the changes that would move it meaningfully.

## Priority order (expected gain vs effort)

### 1. Train longer (80+ epochs instead of 20)
- **Expected gain:** +5-10% AP
- **Effort:** just time (~8-10 hrs at batch_size=8)
- **Why:** CenterPoint-style heads converge slowly. 20 epochs is not enough for the
  heatmap to sharpen + the 5 regression fields (off, height, size, yaw) to stabilize.
  Real PointPillars/CenterPoint train 80-160 epochs on KITTI.
- **Config change:** `epochs: 80` in `config/base.yaml` (or a fusion_3d-specific override)

### 2. Finer BEV grid (0.2m resolution → 320x320 instead of 160x160)
- **Expected gain:** +5-8% AP
- **Effort:** config change (`res_m: 0.2` in `config/fusion_3d.yaml`)
- **Why:** at 0.4m resolution, a car at 30m forward spans only 2-3 cells. The head
  can't localize the box center precisely. 0.2m doubles the grid to 320x320, giving
  4x finer spatial resolution.
- **Tradeoff:** 4x more memory + compute for the BEV convs. Still fits in 8GB VRAM
  (current usage is ~2GB at 160x160, so 320x320 ~8GB — tight but feasible).
- **Also:** update `common/geometry/bev.py` `grid_shape` — it auto-computes from
  range/res, so no code change needed.

### 3. Deeper PointMLP (4 layers instead of 2)
- **Expected gain:** +3-5% AP
- **Effort:** ~10 lines in `common/backbones/lidar_encoder.py`
- **Why:** the current PointMLP is `Linear(6→64) → ReLU → Linear(64→64) → ReLU`.
  Real PointPillars uses a deeper per-point network. More capacity per point = better
  per-cell features after max-pool.
- **Change:**
  ```python
  # current (lidar_encoder.py line 20-27)
  self.net = nn.Sequential(
      nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
      nn.Linear(hidden, out_dim), nn.ReLU(inplace=True),
  )
  # proposed
  self.net = nn.Sequential(
      nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
      nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
      nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
      nn.Linear(hidden, out_dim), nn.ReLU(inplace=True),
  )
  ```

### 4. BEVDepth (camera depth supervision)
- **Expected gain:** +2-5% AP (improves the camera BEV branch)
- **Effort:** ~30 lines in `fusion/fusion_3d/model.py` + `train/losses_3d.py`
- **Why:** the current lift-splat learns depth *indirectly* (only gradient is from the
  detection loss). BEVDepth adds a direct depth supervision loss: render LiDAR depth
  to image, compare to the predicted depth distribution. This makes the camera→BEV
  mapping much more accurate.
- **Implementation:**
  ```python
  # in Fusion3D.forward or _lift_splat: save the depth distribution
  depth_dist = self.depth_head(feat)  # (B, K, Hf, Wf) — already computed
  # in losses_3d.compute_3d_loss: add depth loss
  gt_depth = render_depth_from_lidar(points, calib)  # sparse depth map (Hf, Wf)
  pred_depth = (depth_dist.softmax(-1) * bin_centers).sum(-1)  # expected depth
  depth_loss = F.l1_loss(pred_depth[mask], gt_depth[mask])
  total_loss = detection_loss + 0.1 * depth_loss
  ```
- **Note:** this requires passing the depth distribution through to the loss function,
  which means changing the forward() return signature or storing it as an attribute.

### 5. Padded pillars (instead of max-pool only)
- **Expected gain:** +3-5% AP
- **Effort:** moderate — needs a point-to-pillar scatter kernel
- **Why:** max-pool throws away all but the strongest point per cell. Padded pillars
  keep up to N points per cell (e.g. 32 or 100), giving the MLP more context.
- **Implementation:** either use OpenPCDet's scatter kernel (CUDA, needs Linux) or
  implement a simpler version with `torch.scatter_add_` + a padding mask.
- **Laptop-friendly workaround:** use `torch.zeros(N_max, P, features)` + a loop
  to fill it. Slower but doesn't need custom CUDA.

### 6. All combined
- **Expected total gain:** +15-25% AP (from 0.10 to ~0.25-0.35)
- **Note:** even with all fixes, the simplified architecture has limits vs production
  systems (BEVFusion uses Swin transformers, sparse conv, attention fusion, trains on
  nuScenes for days with 8x A100). The goal is a *better* educational result, not SOTA.

## Non-3D improvements (for the 2D variants)

### Copy-paste augmentation
- **Expected gain:** +3-5% AP for all 2D variants
- **Effort:** moderate (data loader changes)
- **Why:** KITTI has ~7.5k frames. Copy-paste augmentation (paste car instances from
  other frames into the current frame) effectively multiplies the dataset and is
  standard for detection.

### Multi-scale training
- **Expected gain:** +1-3% AP
- **Effort:** small (random image resize in the loader)
- **Why:** makes the model robust to object scale, important for detection.

### Larger backbone (ConvNeXt-Tiny or EfficientNet-B0)
- **Expected gain:** +2-5% AP
- **Effort:** small (swap `ImageBackbone` to use torchvision's ConvNeXt)
- **Why:** ResNet-18 is a 2015 backbone. Modern backbones give better features.
  ConvNeXt-Tiny is ~28M params (fits in 8GB), gives richer features than ResNet-18.

## Things explicitly NOT worth doing on a laptop

- **Sparse convolutions (spconv):** needs Linux + custom CUDA build, not Windows-friendly
- **Swin Transformer backbone:** heavy, slow to train, needs more data than KITTI
- **Training on nuScenes (84GB):** too much data + compute for one RTX 4060
- **Multi-GPU training:** you have one GPU
- **DETR-style query head:** needs 100+ epochs to converge, memory-heavy

## References

- BEVDepth: https://arxiv.org/abs/2206.10040 (depth supervision for lift-splat)
- PointPillars: https://arxiv.org/abs/1812.05093 (padded pillars + scatter)
- CenterPoint: https://arxiv.org/abs/2006.11223 (3D detection head + 2-stage refinement)
- BEVFusion: https://arxiv.org/abs/2205.13542 (camera+LiDAR BEV fusion, SOTA)
- Copy-paste augmentation: https://arxiv.org/abs/2012.07177