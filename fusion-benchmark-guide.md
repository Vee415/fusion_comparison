# Fusion Benchmark Project Guide
**Benchmark 2D early/mid/late fusion + a 3D fusion variant on camera+LiDAR, laptop-trainable, then deploy best-2D in C++ TensorRT on Jetson. New-grad-scoped: depth and understanding over breadth, reuse the commodity ops, build the fusion yourself, and reuse your existing Jetson stack.**

**Date:** 2026-06-19 (revised for new-grad scope + transferable-skills-first framing)
**Author context:** Varun Nelluri — recent grad. Existing strengths: Jetson C++ TensorRT, YOLOv8n, QAT/PTQ + INT8 calibration, radar range-Doppler, measured latency/power/RAM tradeoffs. This project adds the *missing* skill — multi-modal fusion + 3D geometry — on top of the deploy strength you already have. Long-term interest: autonomous drone perception; this project is built as a **transferable-skills gym** whose skills carry to drones (see §13).

---

## 0. The two rules this guide runs on

**Rule 1 (makes it a benchmark, not demos): hold everything fixed except the fusion strategy.**
Same sensor pair, same dataset split, same backbones, same input resolution, same epochs, same seed. The only thing that varies across the four variants is *where and how* camera and LiDAR are fused. "Fusion gain" = variant AP minus the best single-sensor baseline under the *same* constraints. Keep models small and identical-capacity across variants (ResNet-18 everywhere) — this is laptop-friendly *and* makes the comparison fair. You are measuring *which fusion level trades accuracy/latency/robustness how*, under fixed compute. That framing is the whole value of the artifact.

**Rule 2 (new-grad rule): depth and understanding over breadth. Optimize for "can I explain every line cold," not "did I build everything."**
The senior-scope version of this project (full 3D export with custom CUDA kernels, four trained variants, sparse-conv point nets, all deployed) is 4–6 months for someone learning the domain, and the trap is shipping something sprawling you can't fully explain — which reads *worse* in an interview than something smaller you own completely. So this guide keeps the four variants and the deploy, but it tells you **what to build, what to study-and-reuse, and what to defer**, and it sequences the work so each concept clicks before the next. Build the fusion glue yourself (nobody hands you that, and it's what the job tests). Reuse reference code for the heavy commodity ops (PointPillars scatter, KITTI 3D eval). Understand both. That allocation is how you go deep without drowning.

---

## 1. What you build vs study vs defer (the honesty table — read first)

The single most important section for keeping this new-grad-scoped. Be honest about which is which in your README; reviewers respect a stated scope.

| Piece | Verdict | Why |
|---|---|---|
| Calibration + projection (LiDAR→image, image→LiDAR) | **Build** (and visually verify) | The universal geometry muscle; transferable to *every* fusion system including drones. Build it, test it, trust it. |
| Shared image backbone (ResNet-18 FPN), LiDAR point encoder, box utils | **Build** | Small, reusable, teaches the encoders. |
| The `FusionModel` interface + one trainer for all variants | **Build** | This modularity is what makes it a benchmark, not four demos. |
| Variants A (early-2D), B (mid-2D), C (late-2D) | **Build all three** | The core skill you came for. Each is a small model. Understand each cold. |
| Variant D (3D BEV: PointPillars + lift-splat + CenterPoint) | **Build the fusion glue; reuse OpenPCDet for ops/eval** | Understand it deeply (whiteboard the data flow); don't hand-write the CUDA scatter or the 3D-AP eval. Reuse `OpenPCDet`/KITTI devkit for those commodity parts. |
| Eval harness (AP + latency + robustness) | **Build** 2D; **reuse KITTI devkit** for 3D-AP | 2D AP you build; 3D-AP you do *not* reinvent (pitfall #5). |
| ONNX export of all four | **Build** (light) | Cheap, transferable, do all four. |
| C++ TensorRT deploy of **best-2D** | **Build, full strength** | Your existing Jetson/QAT/INT8 toolchain, applied to a fusion model. This is your standout new-grad signal. |
| C++ TensorRT deploy of the 3D variant | **Defer to v2 (state as planned, don't build now)** | Custom CUDA pillarize/scatter + 3D head export is mid/senior edge work. Document it as v2 scope; don't under-deliver by half-building it. |
| Sparse-conv / full lift-splat C++ export | **Study, never build** | Least transferable, most painful. Read production PointPillars-TRT pipelines to speak about them; don't implement. |

This table is the difference between a project you finish and can explain, and one you don't. Come back to it whenever scope creeps.

---

## 2. Sensor pair + dataset

**Pair: camera (RGB) + LiDAR (3D point cloud).** Chosen because it's the cleanest *classroom* for learning transferable fusion + 3D geometry — public paired data with 2D+3D annotations and calibration, small enough to train on a laptop. Not because it's "the drone stack" (see §13 for the drone on-ramp). The *skills* you learn here (calibration, projection, feature fusion, 3D geometry, edge deploy) transfer; the *sensor pair* you'll swap later for drone work.

**Dataset: KITTI Object Detection (primary).**
- Single front camera + LiDAR, registered, with 2D + 3D boxes and per-frame calibration files (`calib/*.txt`: camera intrinsics `P2`, LiDAR→camera `velo_to_cam`). Small enough to train a tiny model on a laptop GPU. Use the **Car** class; subset to ~1,500 train / ~400 val if needed.
- nuScenes mini (10 scenes) — only if KITTI feels too easy and you have ≥8 GB VRAM. Heavier and multi-camera complicates the 2D variants.

**Calibration is the project's #1 risk** (see Pitfalls §10). Build and **visually verify** projection (overlay LiDAR points on the image, confirm they land on car bodies) before training anything. This is the cheapest place to learn the geometry that underpins all fusion.

---

## 3. Module structure (designed to plug variants in)

```
fusion-benchmark/
  README.md                      # results table + design tradeoffs + build-vs-study scope honesty (the deliverable)
  config/
    base.yaml                    # shared: backbone, resolution, epochs, seed, split
    early_2d.yaml  mid_2d.yaml  late_2d.yaml  fusion_3d.yaml   # only fusion-specific deltas
  data/
    kitti_download.sh
    kitti/                       # raw + calibration (gitignored)
    loaders/
      paired_loader.py           # yields (image, points, calib, boxes2d, boxes3d)
  common/
    sensors/
      calibration.py             # load P2, velo_to_cam; Camera + Lidar objects
      projection.py              # lidar->image, image->lidar (with depth), bev transforms
    backbones/
      image_backbone.py          # shared ResNet-18 FPN (used by every variant)
      lidar_encoder.py           # shared point encoder (pillar/voxel -> BEV/2D features)
    geometry/
      boxes2d.py  boxes3d.py     # encode/decode, IoU2D, IoU3D/BEV-IoU, rotated NMS
      bev.py                     # point->pillar, BEV grid, lift-splat-shoot (camera->BEV)
  fusion/
    base.py                      # FusionModel interface (see §4)
    early_2d/model.py
    mid_2d/model.py
    late_2d/model.py
    fusion_3d/model.py
  train/
    trainer.py                   # one trainer for all variants via the interface
    losses_2d.py  losses_3d.py
  eval/
    metrics_2d.py                # AP (KITTI 40-point), P/R, AP_small  -- BUILD
    metrics_3d.py                # thin wrapper over KITTI devkit for AP-3D/AP-BEV -- REUSE
    benchmark.py                 # accuracy + latency + robustness table
    robustness.py                # re-eval with camera-blinded and lidar-removed
  export/
    onnx_export.py               # per-variant ONNX + shape check (all four)
  deploy_cpp/                    # Jetson side
    best_2d/                     # BUILD (your strength)
    fusion_3d/                   # v2 — document, don't build now
  tests/
    test_projection.py           # overlay sanity + numeric round-trip  (DO THIS FIRST)
    test_boxes.py  test_interface.py
  notebooks/
    01_lidar_visual_sanity_check.ipynb   # milestone 1, doubles as your learning tool
```

Everything above `fusion/` is **shared**; each variant is one folder implementing the same interface. Adding a 5th variant later = one folder + one config.

---

## 4. The shared interface (this is what makes it modular)

Every variant implements one class so the trainer/eval/export code is written **once**:

```python
# fusion/base.py
class FusionModel(nn.Module):
    def forward(self, batch) -> dict:
        """
        batch: {image: (B,3,H,W), points: list[(N_i,4)] xyz+int, calib: ...,
                bev_map: optional}
        returns: {boxes2d or boxes3d, scores, classes}  # variant decides output space
        """
        raise NotImplementedError
    def output_space(self) -> str:  # "2d" | "3d"
        ...
    def export_onnx(self, path):  # variant-specific input signature
        ...
```

- `trainer.py` calls `model(batch)`, computes the loss based on `model.output_space()`, steps. One trainer, four variants.
- `benchmark.py` calls `model.output_space()` to pick the right metric suite.
- `export_onnx` lives in the variant because early/mid/late take different inputs.

This is the single most important design choice — it's what lets you swap variants and produce one comparison table from one harness, and it's reusable engineering that reads well on a CV.

---

## 5. The four variants — concrete specs (keep all four; scope the 3D honestly)

All share: ResNet-18 image backbone, AdamW, cosine schedule, AMP, identical epochs. Image input **384×1280** (KITTI aspect); BEV grid **±32 m × ±32 m, 0.2 m cells** for the 3D variant (small grid = laptop-friendly).

### Baselines (build these first — they're the reference the fusion gain is measured against)
- **Camera-only 2D:** shared image backbone + 2D head, RGB only.
- **LiDAR-only 2D:** project points to image, render a 3-channel "depth/height/intensity" image, same 2D head.
- **LiDAR-only 3D:** PointPillars-lite → BEV → 3D head, no camera.
- **Camera-only 3D:** (weak) monocular 3D head — include only if time; mainly to show why you fuse.

### Variant A — Early-2D (data-level fusion) — BUILD
- **Fuse at input:** project LiDAR into the camera frame and append a 4th channel = depth (optionally HHA). Input `(H, W, 4)`.
- **Model:** shared image backbone with first conv → 4-in channels → 2D head.
- **Output:** 2D boxes.
- **What it tests:** max raw info, but the network must learn to use a geometry channel from scratch. Sensitive to projection/calib errors and missing-depth pixels (interpolate or mask).

### Variant B — Mid-2D (feature-level fusion) — BUILD
- **Fuse at features:** image backbone → multi-scale feature map `F_cam`. LiDAR → tiny encoder producing a feature map **aligned to the image grid** → `F_lid`. Fuse: `concat([F_cam, F_lid]) + 1×1 conv` (optional cross-attention block) → 2D head.
- **Output:** 2D boxes.
- **What it tests:** the modern sweet spot in 2D — each modality gets a suited encoder, fusion where representations are comparable. This is the variant most likely to win and the one you'll deploy.
- **Transfer note:** this exact architecture maps to camera+radar by swapping the LiDAR branch for a range-Doppler feature branch — directly relevant to your radar background and to drones. Understand it cold.

### Variant C — Late-2D (decision-level fusion) — BUILD
- **Two independent detectors:** camera-only 2D → `D_cam`; LiDAR-only 2D → `D_lid`.
- **Merge:** associate by IoU2D (Hungarian), fuse confidence (weighted avg or max), NMS the merged list. Unmatched high-conf boxes kept.
- **Output:** 2D boxes.
- **What it tests:** robustness/modularity — easiest to debug, degrades gracefully, but can't recover from a hard miss by one detector.

### Variant D — Fusion-3D (BEV feature fusion) — BUILD THE GLUE, REUSE OpenPCDet FOR OPS/EVAL
- **Camera branch:** image backbone → features → **lift-splat-shoot** (predict per-pixel depth, lift to 3D frustum, splat into BEV) → `BEV_cam`.
- **LiDAR branch:** **PointPillars-lite** (points → pillars → 2D pseudo-image via PointNet + scatter) → `BEV_lid`.
- **Fuse in BEV:** `concat([BEV_cam, BEV_lid]) + conv` → BEV feature map.
- **Head:** CenterPoint-style (heatmap + offset + size + yaw + velocity) → 3D boxes `(x,y,z,w,h,l,yaw)`.
- **Output:** 3D boxes.
- **Scope for new-grad:** understand the full data flow deeply enough to whiteboard it. **Reuse `OpenPCDet` for the pillarize/scatter ops and the 3D-AP eval** — do not hand-write the CUDA scatter or the KITTI 3D eval (commodity, painful, least-transferable). Build the *fusion* (the concat + conv in BEV, the two-branch glue) yourself; that's the part you can't copy and the part that's transferable. Use a *tiny* lift-splat (small depth bins, coarse BEV) and PointPillars (not VoxelNet/sparse-conv — painful to install/export). Goal: a working, honest 3D-fusion pipeline you can explain, not SOTA NDS.

> If Variant D is eating your timeline: ship A/B/C + baselines + the 2D deploy first (the grad-appropriate core), then add D as the stretch. A complete A/B/C benchmark + deploy is a strong artifact on its own; D is the depth cherry on top, not the foundation.

---

## 6. Benchmark harness — what to measure, one table

`eval/benchmark.py` produces **one** table across all variants + baselines:

| Variant | Output | AP | AP_small | Latency p50/p95 (laptop) | Cam-dropped AP | LiDAR-dropped AP |
|---|---|---|---|---|---|---|

- **Accuracy:** 2D → KITTI 40-point AP (Easy/Moderate/Hard) — **you build**; 3D → AP-3D/AP-BEV at IoU 0.7 Car Moderate — **reuse KITTI devkit** (don't reinvent eval; pitfall #5).
- **Cost:** p50/p95 latency on your laptop GPU (batch 1, warmup, fixed input) — **you build** (it's your discipline). Params/FLOPs/mem are optional if time-pressed; latency + AP + robustness are the core three.
- **Robustness (the differentiator):** for each fusion variant, re-eval with **camera blinded** (zero image / drop camera branch) and **LiDAR removed** (drop lidar branch). The point of fusion is graceful degradation — a mid/3D variant that keeps ~60% AP when one sensor dies beats a late variant that craters. This row is what makes the writeup say "I understand *why* fuse, not just *how*."

**Also produce:** (1) accuracy-vs-latency scatter (all variants on one plot), (2) a written failure analysis — which objects/regimes each variant wins/loses on (distant cars, occluded, night-ish via augmentation). Numbers + analysis = the artifact.

---

## 7. Laptop reality-check

- Use **AMP (fp16)**, batch 1–2, ResNet-18, ~1.5k KITTI samples, ≤60 epochs. A 6–8 GB GPU is enough for A–C; D needs care (small BEV grid, PointPillars not sparse-conv).
- **Cache projections**: precompute LiDAR→image / →BEV tensors once, save to disk; don't reproject every epoch.
- Subsample LiDAR points (random 15k per frame) to bound memory.
- If D OOMs: shrink BEV grid to ±20 m, coarser cells (0.4 m), fewer lift depth bins. Document the constraint — the benchmark is comparative, so it's fine.
- **Train 2D variants first** (fast, validates the whole pipeline + projection), then 3D.

---

## 8. Milestones — ordered as a learning roadmap with understanding gates

Sequenced so each concept clicks before the next. **Do not skip the gates** — if a gate isn't solid, the later work is built on sand and you won't be able to explain it.

1. **Data + calibration + projection + visual check (GATE: you can overlay LiDAR on the image and points land on cars; round-trip test passes).** Download KITTI, build `calibration.py` + `projection.py`, write `test_projection.py`, build `notebooks/01_lidar_visual_sanity_check.ipynb`. *This is also how you learn LiDAR from scratch — see each concept on real data.*
2. **Shared parts (GATE: one trainer runs a dummy variant end-to-end).** Image backbone, LiDAR encoder, box utils (2D + 3D), BEV ops, paired loader, the `FusionModel` interface + one trainer.
3. **Baselines + eval harness (GATE: AP-2D and AP-3D numbers are sane on baselines).** Camera-only 2D, LiDAR-only 2D, LiDAR-only 3D. Get 2D AP (build) + 3D AP (KITTI devkit) working here. Now the benchmark harness exists.
4. **Variant A (early-2D) → numbers (GATE: you can explain the 4th channel and why missing-depth matters).**
5. **Variant B (mid-2D) → numbers (GATE: you can explain why the grids must align and what the 1×1 conv does).** Most likely your deploy winner.
6. **Variant C (late-2D) → numbers (GATE: you can explain the merge/NMS and graceful degradation).**
7. **Robustness pass** (cam-dropped / lidar-dropped) on A/B/C + the one benchmark table + scatter + failure analysis → write README. **At this point you have a complete, grad-appropriate, deployable artifact.** Ship-ready checkpoint.
8. **Variant D (fusion-3D) → numbers (GATE: whiteboard the full data flow: points→pillars→scatter→BEV and image→lift→splat→BEV, then concat+conv).** Reuse OpenPCDet ops + KITTI devkit eval; build the fusion glue. This is the stretch/depth piece — budget the most time here, but it's *on top of* an already-complete project, not blocking it.
9. **Export:** ONNX for all four (light, do all).
10. **C++ Jetson deploy of best-2D** (see §9) — your existing strength, now applied to a fusion model. This is the standout new-grad signal.

> Note the ordering: you can **stop after milestone 7 with a strong, complete artifact** and still have a legit new-grad project (A/B/C benchmark + robustness + README). D and the deploy are the depth that makes it stand out — but the project doesn't collapse if life happens and you stop at 7. That's the safety net that keeps this finishable.

---

## 9. C++ TensorRT deploy on Jetson — best-2D (your strength; build this fully)

This is where your existing stack becomes the differentiator. Reuse your Jetson C++ TensorRT toolchain (ONNX→engine, CUDA preprocess, `enqueueV3`, NMS/decode, FPS/power/RAM across FP16/INT8) on the best-2D variant (likely B). One port, done properly:

- ONNX → TensorRT **FP16** and **INT8** (with calibration set — apply your QAT-vs-PTQ experience; you've done exactly this analysis on Keyword Spotting, so you know to watch for accuracy collapse on small heads and to fall back to QAT if PTQ hurts).
- Input = whatever the variant needs (4-channel for early; two streams for mid; two networks + a C++ merge/NMS for late). Pick the variant that exports cleanest *and* scores best on your benchmark — measure, don't assume.
- CUDA letterbox preprocess, CPU/GPU NMS, decode boxes.
- Measure FPS, p50/p95 latency, W (watts), RAM at 15W vs MAXN — your usual axes.
- Produce the **accuracy-cost frontier** (AP vs FPS vs W, FP16 vs INT8). That table is your standout signal: a new grad who can show a measured fusion-model deploy frontier is rare.

**3D deploy is v2 — document it, don't build it now.** State in the README: "v1 deploys best-2D; v2 would deploy the 3D variant by doing pillarize+scatter in CUDA so the BEV head exports as a plain CNN (the standard production split)." That sentence shows you *understand* the senior path without under-delivering by half-building it. Full lift-splat in C++ = v3 / never — study production PointPillars-TRT pipelines to speak about it, don't implement.

---

## 10. Pitfalls (read before starting)

1. **Calibration/projection errors** make every fusion variant look bad *and* make it look like fusion doesn't help. Verify projection visually first (LiDAR points land on car bodies). Write the round-trip test (point→pixel→point). This is milestone 1's gate for a reason.
2. **Unfair comparison** — the moment you let backbone/epochs/resolution drift between variants, the benchmark is dead. One config, one seed.
3. **Missing-depth pixels in early fusion** — LiDAR doesn't cover the whole image. Mask or interpolate; don't feed zeros silently.
4. **BEV coordinate convention** — KITTI uses camera coords (x right, y down, z forward); nuScenes uses ego/global. Mix these up and 3D boxes fly off into nowhere. Pick one, assert it.
5. **Rotated NMS + 3D eval** — use the official KITTI devkit / `pylot` / OpenPCDet eval for 3D AP so your numbers are credible. Do *not* reinvent 3D eval — that's reuse, not build (see §1).
6. **Latency measurement** — warmup, batch 1, lock clocks, many iterations, p50/p95 not mean. You already do this; keep the discipline.
7. **Overclaiming** — with small models you will *not* hit SOTA AP. Frame as "fusion-level tradeoff benchmark under fixed compute," not "I beat BEVFusion." Honesty is the signal — especially as a new grad, a crisply-honest scoped project beats a puffed-up one every time.
8. **Scope creep into senior infrastructure** — every time you're about to hand-write a CUDA kernel or a 3D eval from scratch, check §1's table. If it says reuse/study, reuse/study. Your time goes into the fusion and the deploy, not commodity plumbing.

---

## 11. Transferable skills you earn (the real point for a new grad)

This project is a **transferable-skills gym**, not a "car detector." The skills map across automotive, drones, robotics:

| Skill (earned here) | Transfers to | How you show it |
|---|---|---|
| Calibration + multi-sensor projection/geometry | Every fusion system, incl. drones | Milestone 1 + round-trip test |
| Multi-modal fusion mechanics (early/mid/late, feature-level) | All multi-sensor stacks | Variants A/B/C + comparison table |
| 3D geometry / BEV / point networks (understood) | Inspection drones, robotics, automotive | Variant D + whiteboard explanation |
| Detection + metrics + benchmarking discipline | All perception roles | The one-table harness |
| Edge deploy of a *fused* model (FP16/INT8, measured frontier) | All realtime perception | best-2D Jetson deploy |
| Reading + reusing senior reference code (OpenPCDet) | Day-one of any job | Build-vs-study scope in README |

After this you are a *transferable* perception engineer who can fuse sensors anywhere — not someone who can only do camera+LiDAR 3D car detection.

---

## 12. CV bullet to earn (new-grad, honest, decision-mode)

> Benchmarked camera–LiDAR fusion at every level — early (depth-as-channel), mid (feature-map fusion), late (detection-list merge) 2D, plus a 3D BEV variant (PointPillars + lift-splat-shoot, CenterPoint head) — on KITTI under fixed backbone/compute so only fusion strategy varied; measured AP, graceful-degradation AP under single-sensor failure, and p95 latency. Reused my Jetson C++ TensorRT + QAT/INT8 toolchain to deploy the best 2D variant with an FP16/INT8 accuracy-cost frontier; reused OpenPCDet ops and the KITTI devkit for the 3D path and documented the build-vs-study scope honestly.

Not: "Built a multi-modal fusion model using PyTorch." The verbs that land for a new grad: *benchmarked under fixed compute, measured graceful degradation, reused my deploy toolchain, documented scope honestly.*

---

## 13. On-ramp to drones (your long-term interest — read this last)

This project is deliberately *not* a drone stack, and that's fine: drones rarely carry LiDAR (weight/power), and their dominant fusion is **visual-inertial (camera + IMU) over time**, not camera+LiDAR over single frames. But the transferable skills you build here are exactly the on-ramp:

- **Calibration + projection + geometry** → directly underpins VIO/SLAM sensor alignment.
- **Feature-level fusion (Variant B)** → the *same architecture* transfers to camera+radar (swap the LiDAR branch for a range-Doppler branch — your radar strength) and to camera+IMU feature fusion.
- **BEV + 3D geometry (Variant D, understood)** → transfers to inspection/mapping drones that *do* carry LiDAR/radar.
- **Edge deploy discipline** → drones run on Jetson under hard realtime + battery; your deploy strength is the scarcest skill in drone perception.

The one skill this project *doesn't* build, that drone perception lives on: **temporal fusion / state estimation** (Kalman/EKF/UKF, tracking, VIO back-ends). That's the natural **phase-2 project** after this one: add temporal fusion to one variant (track boxes across frames with an EKF), then specialize to VIO with `VINS-Fusion` / `OpenVINS` on the **EuRoC MAV** dataset. Because you'll already own geometry + fusion + edge-deploy, phase 2 is "learn the VIO back-end + aerial data," not "learn everything at once" — a small, finishable next step rather than a fresh mountain.

**Sequencing for your goal:** v1 = this benchmark (transferable skills, A/B/C + best-2D deploy, with D as stretch) → v2 = add temporal fusion → v3 = drone-specialize (VIO, EuRoC). Each phase is a complete, explainable artifact; none is a cliff.

---

## 14. Suggested libraries
- **Backbones/ops:** PyTorch, `torchvision` (ResNet-18), **OpenPCDet** (reference for PointPillars/CenterPoint ops + eval — read it, reuse the ops/eval, build the fusion glue yourself), MMDetection3D (reference only — heavy).
- **Projection/calib:** `pykitti` or raw file parsing; **Open3D** (visualization + point sanity — use it in the milestone-1 notebook).
- **Metrics:** official **KITTI devkit** for 3D AP (reuse, don't reinvent); `pycocotools` for 2D AP (adapt); `motmetrics` not needed here (no tracking — that's phase 2).
- **Cost:** `fvcore`/`ptflops` for FLOPs/params (optional).
- **Export:** `torch.onnx`, `onnx-simplifier` (onnxsim), TensorRT 10.x (matches your Jetson stack).
- **C++:** your existing TensorRT C++ binary + CUDA preprocess scaffold. 3D CUDA pillarize/scatter = v2 scope (document, don't build now).
- **Phase 2 (drone on-ramp):** `VINS-Fusion` / `OpenVINS`, EuRoC MAV dataset, GTSAM for the back-end.