# Learning Fusion — From Your Own Code

> **Purpose:** teach multi-modal sensor fusion by reading what you *already built*,
> not a textbook. Every concept below maps to a real file/function in this repo so you
> can open it, see it, and explain it cold in an interview.
>
> **How to use this doc:** read one section, open the linked file, convince yourself the
> code matches the explanation, then move on. When you're done, do the **Self-Test**
> at the bottom *without* looking at the answers. Come back days later and re-test.
>
> **Your project in one sentence:** a *benchmark* of four fusion strategies
> (early-2D, mid-2D, late-2D, 3D-BEV) on camera + LiDAR, where **everything is held fixed
> except where/how the two sensors are fused** — so the only thing you measure is the
> *effect of the fusion strategy*.

---

## 0. Why fuse sensors at all?

A single sensor has hard failure modes:
- **Camera** fails at night, in glare, in fog, and gives **no distance** (a pixel tells you
  nothing about how far an object is — that's lost the moment light hits the sensor).
- **LiDAR** gives precise 3D distance but no color/texture, is sparse far away, and can be
  fooled by dark/glass surfaces.

Fusion's promise: each sensor covers the other's failure mode, so the combined detector is
**more accurate *and* more robust** (degrades gracefully when one sensor drops out) than
either alone. The whole project exists to *measure* that claim honestly.

**The benchmark's single rule (from `fusion-benchmark-guide.md` §0):** hold the sensor pair,
dataset split, backbones (ResNet-18 everywhere), input resolution, epochs, and seed **fixed**.
The *only* thing that varies across the four variants is *where and how* you fuse.
"Fusion gain" = variant AP − best single-sensor baseline under the same constraints. That
framing is what makes this a benchmark, not four demos.

---

## 1. Coordinate frames & calibration — the universal geometry muscle

This is the #1 thing reviewers care about and the #1 place new-grads hand-wave. Your repo
treats it correctly — `common/sensors/calibration.py`.

### The frames
KITTI uses three coordinate frames, and fusion = moving data between them:

```
velo (LiDAR frame)  --Tr_velo_to_cam-->  cam (unrectified)  --R0_rect-->  cam (rectified)  --P2-->  image (u,v,depth)
```

- **velo** = LiDAR's own frame (where points live when the sensor spits them out).
- **cam (rectified)** = the camera's frame after stereo rectification. This is the
  *reference* frame your code works in.
- **image** = 2D pixels (u,v) + a depth value.

### The matrices (`calib.py:12-23`)
A `Calib` object stores three matrices as homogeneous 4×4 so composition is just matmul:
- **`V2C`** = `Tr_velo_to_cam` (3×4) — LiDAR → camera.
- **`R0`** = `R0_rect` (3×3) — rectify the camera frame.
- **`P`** = `P2` (3×4) — camera intrinsics (focal length fx,fy, principal point cx,cy).

### The convention your code asserts (look at every docstring)
> **cam frame: x right, y down, z forward. BEV grid: dim0(r)=z (forward), dim1(c)=x (right),
> centered at ego.**

This is KITTI-specific and *different* from the more common "x forward, y left, z up" you'll
see in ROS/nuScenes. Your code asserts it and assumes points are already in cam frame for
the 3D branch — so always convert velo→cam first (`lidar_encoder.py` does this at lines 52/92).

### Why this matters for the interview
If you can't name the frames and the matrix chain, a sensing interviewer will assume the rest
is copy-paste. If you can say "LiDAR points live in the velo frame; I compose
`R0 @ V2C` to bring them into the rectified camera frame, then `P2` to project to pixels,
keeping depth as the third coordinate" — that's the whole game.

---

## 2. Projection — turning 3D LiDAR points into 2D pixels (and back)

This is the bridge that makes *all* 2D fusion possible. `common/sensors/projection.py`.

### Forward: LiDAR point → pixel (`projection.py:59-76`, `lidar_to_image_torch`)
```python
h = [x, y, z, 1]                          # homogeneous velo point
cam = (R0 @ V2C @ h.T).T[:, :3]           # velo -> rectified cam (3D still)
img = (P @ [cam, 1].T).T[:, :3]           # apply intrinsics -> (X', Y', Z') where Z' = depth
depth = img[:, 2]                         # the forward distance survives as depth
uv = img[:, :2] / depth                   # perspective divide: pixel = image_xyz / depth
```
The key insight: projection = **matrix multiply, then divide by depth (the "perspective divide")**.
`depth` is not a separate measurement — it falls out of the `Z` coordinate after the
intrinsics. `valid` keeps only points in front of the camera (`depth > 0`) and inside the
image bounds.

### Rendering depth to an image (`projection.py:79-93`, `render_depth_torch`)
Project every point, write its depth into the pixel it lands on. Tricky detail: multiple
LiDAR points can hit the same pixel. Your code sorts **far-first** so the **nearest point
overwrites** — the closest surface wins, which is what you want for a depth map.

### Back-projection: pixel + depth → 3D point (`calibration.py:83-91`, used by lift-splat)
This is the *reverse* trick and it's the heart of the 3D camera branch:
```python
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
z = depth
```
A single pixel has no depth, so it can't become a 3D point. But a **pixel + a predicted depth**
*can*. This is exactly what lift-splat-shoot does (Section 5): the camera branch *predicts* a
depth distribution for every pixel, then back-projects each pixel to 3D using that depth.

### The 3-channel LiDAR image (`projection.py:96-112`, `render_lidar_3ch_torch`)
For the late-fusion LiDAR-only detector, you render LiDAR as a fake 3-channel image
`[depth, height(z), intensity]`. This is how you turn sparse 3D points into something a normal
2D CNN can eat — you "camera-ify" the LiDAR.

---

## 3. The three fusion levels — the core concept, mapped to your three model files

This is **the** thing every fusion JD asks about. Three levels, defined by *where in the
pipeline* you combine the sensors:

| Level | Where you fuse | What you combine | Your file |
|---|---|---|---|
| **Early** | at the **input** | raw RGB + raw LiDAR-as-depth → 4-channel image | `fusion/early_2d/model.py` |
| **Mid** | at the **features** | camera feature map + LiDAR feature map | `fusion/mid_2d/model.py` |
| **Late** | at the **decision** | two separate sets of detections | `fusion/late_2d/model.py` |

### Early fusion (data-level) — `early_2d/model.py:26-41`
```python
d = render_depth_torch(pts, calib, H, W, ...)   # LiDAR -> (H,W) depth
d = (d / 80.0).clamp(0, 1)                      # normalize to [0,1]
x = torch.cat([imgs, depth], dim=1)             # (B,4,H,W): RGB + depth
feat = self.backbone(x)[stride]                 # first conv takes 4 channels
```
- **What's fused:** the raw signals, before any learning. The network sees a 4th channel.
- **Why simple:** the depth rendering is *preprocessing*, so the network is just a
  4-channel detector — clean to export. At deploy you do the depth render in C++/CUDA
  and feed a 4-channel tensor.
- **Weakness:** the network has to learn to use a depth channel from scratch, and a sparse
  depth map wastes most of that channel as zeros.

### Mid fusion (feature-level) — `mid_2d/model.py:32-46` — the modern sweet spot
```python
f_cam = self.image_backbone(img)[stride]                    # (B,C,Hg,Wg)
f_lid = self.lidar_encoder(points, calib, H, W, device)      # (B,Cl,Hg,Wg) ALIGNED to cam grid
fused = self.fuse(torch.cat([f_cam, f_lid], dim=1))          # concat + 1x1 conv
return self.head(fused)
```
- **What's fused:** each sensor is *encoded into features first*, then the feature maps are
  concatenated and mixed with a 1×1 conv.
- **The critical word — ALIGNED:** the LiDAR encoder (`lidar_encoder.py:31-67`)
  projects points to the *camera feature grid* (same stride, same Hg×Wg cells), so cell
  (i,j) in `f_cam` and cell (i,j) in `f_lid` describe the **same physical region**. Without
  alignment, concatenation is meaningless.
- **The PointNet atom:** per-point MLP → max-pool per cell (`scatter_reduce_` with `amax`).
  Max-pool is permutation-invariant and handles variable point counts per cell — that's why
  PointNet-style encoders work on unordered point sets.
- **Why it's the sweet spot:** each sensor gets a learned representation, fusion is cheap
  (one conv), and it usually beats early/late. Likely your deploy winner.
- **Deploy note:** the LiDAR scatter is awkward to export to ONNX, so you export the net to
  take a *pre-scattered* LiDAR feature map and do the scatter in C++/CUDA at deploy.

### Late fusion (decision-level) — `late_2d/model.py:34-87`
```python
cam_pred = self.cam_head(self.cam_backbone(img)[stride])      # detector A on RGB
lid_pred = self.lid_head(self.lid_backbone(lid_img)[stride])  # detector B on LiDAR-as-image
# decode -> merge by IoU association -> fuse conf (mean or max) -> NMS
```
- **What's fused:** the *outputs* — two independent detectors each produce boxes, then you
  merge them.
- **The merge (`_merge`, lines 61-87):** match camera boxes to LiDAR boxes by **IoU**
  (geometric association), fuse confidences (mean or max), then **NMS** the combined list.
  This is the classic "track/detect-then-merge" pattern.
- **Training (`loss`, line 50-51):** sum of *both* detectors' losses against the *same* GT,
  each sees the same ground truth. `custom_loss=True` (set in `base.py`) so the trainer knows
  not to apply the standard loss path.
- **Cost:** needs **two engines + a C++ merge/NMS** at deploy — heavier than mid.
- **Strength:** most robust by design (each sensor has its own detector, so one failing
  doesn't kill the pipeline), and easiest to reason about.

### The one-line mental model
> **Early** fuses data, **mid** fuses features, **late** fuses decisions. The later you fuse,
> the more independent the sensors and the more robust you are, but the less the sensors can
> *help each other learn*. The earlier you fuse, the more the network can jointly exploit
> both signals, but the harder the learning and the export.

---

## 4. Robustness — the real reason fusion beats single-sensor

This is what turns "I fused two sensors" into "I can *prove* fusion helps." `fusion/base.py:32-37`.

Every variant has blind-mode hooks:
```python
def set_cam_blind(self, flag=True): self._cam_blind = flag
def set_lidar_blind(self, flag=True): self._lidar_blind = flag
```
When blind, the branch is *zeroed but the forward pass still runs* (see e.g.
`mid_2d/model.py:35-41`). So you can measure **AP with both sensors**, then **AP with camera
dropped**, then **AP with LiDAR dropped**. A good fused model should:
- beat both single-sensor baselines with both sensors present (fusion gain), **and**
- when one sensor is dropped, degrade gracefully — not collapse to zero.

The robustness sweep lives in `eval/robustness.py`. This single experiment is the most
interview-impressive part of the project: most candidates show accuracy; you show
*graceful degradation under sensor failure*, which is exactly what safety-critical
perception roles ask for.

---

## 5. BEV & 3D fusion — `fusion/fusion_3d/model.py` + `common/geometry/bev.py`

The 2D variants detect boxes in the image. The 3D variant detects boxes in the world
`(x, y, z, w, h, l, yaw)` — the canonical automotive approach.

### Why BEV (bird's-eye view)?
Detecting in the image plane is bad for driving because perspective distorts distances
(close objects are huge, far ones tiny) and objects overlap. BEV is a **top-down grid**
where one cell = one patch of ground; distances and sizes are physically uniform. It's the
natural space for 3D detection.

### Your BEV convention (`bev.py:1-6`)
```
cam frame: x right, y down, z forward
BEV grid: dim0 (r) = z (forward), dim1 (c) = x (right), centered at ego
z = -range + (r+0.5)*res,  x = -range + (c+0.5)*res
```
So a point's BEV cell is found from its **x** (→column) and **z** (→row); **y** (height) is
the *thing you encode inside the cell*, not where the cell is. That's why it's called
"bird's-eye" — you collapse height into the cell's features.

### The two branches (both build a BEV feature map, then fuse in BEV)
**Camera branch — simplified lift-splat-shoot (`fusion_3d/model.py:48-79`):**
1. Image backbone → per-pixel features.
2. `depth_head` predicts a **depth distribution** over K bins for every pixel (Section 2's
   back-projection needs a depth — here the network *predicts* one).
3. Back-project each pixel at each candidate depth to 3D: `x=(u-cx)*d/fx, y=(v-cy)*d/fy, z=d`.
4. Map to BEV cells, weight each contribution by its softmax depth probability, and
   **scatter-add** into the BEV grid. (The `scatter_add_` at line 77 is the "splat".)
Result: `BEV_cam` — the camera's view lifted into the top-down world.

**LiDAR branch — simplified PointPillars (`lidar_encoder.py:70-108`):**
1. Convert points to cam frame.
2. Assign each point to a BEV cell ("pillar").
3. Per-point MLP on `[x,y,z,intensity, dx,dz]` (point coords + cell-relative offsets).
4. **Max-pool per cell** → `BEV_lid`, a top-down pseudo-image.

**Fusion + head (`fusion_3d/model.py:96-97`):**
```python
fused = self.fuse(torch.cat([bev_cam, bev_lid], dim=1))   # concat in BEV + 1x1 conv
return self.head(fused)                                   # CenterPoint 3D head
```
Both branches speak the *same* BEV language, so concatenation is geometrically meaningful
(same as mid-2D's alignment, but in BEV instead of the image grid).

### Honesty note (already in your code, `fusion_3d/model.py:8-13`)
Your lift-splat is **simplified** (direct splat, no frustum pooling). For real 3D-AP numbers
you reuse the KITTI devkit (`eval/metrics_3d.py`) — you do *not* trust this simplified pipeline
for publication numbers, and you don't hand-write the CUDA scatter. That stated scope is
what makes the project credible.

---

## 6. The deploy story (where your Jetson strength plugs in)

This is your standout new-grad signal — you can take a *fused* model to edge hardware, not
just a single detector. Each variant has a different deploy shape:

| Variant | Export | C++ preprocess | Engines |
|---|---|---|---|
| Early | 4-channel detector | render LiDAR→depth in C++/CUDA | 1 |
| Mid | net takes pre-scattered LiDAR feat map | scatter points→grid in C++/CUDA | 1 |
| Late | two backbones | merge boxes + NMS in C++ | **2** |
| 3D | only BEV-fuse + head | lift-splat & pillarize in C++/CUDA (v2) | 1 |

The pattern across all of them: **the messy geometry (projection, scatter, splat) is moved
out of the network into C++/CUDA preprocessing**, so the exported ONNX is a clean
feature→boxes net you can run through TensorRT — exactly the toolchain you already own from
P7/QAT. Mid-2D is the likely deploy winner (one engine, clean export, best accuracy).

---

## 7. Interview quick-reference (memorize these seven)

1. **Frames:** velo →(`V2C`)→ cam unrect →(`R0`)→ cam rect →(`P2`)→ image. Convention x-right, y-down, z-forward.
2. **Projection:** matrix multiply then **perspective divide** by depth; depth survives as the Z coordinate.
3. **Early/mid/late** = fuse at **input / features / decisions**. The trade is joint-learning vs robustness vs export cost.
4. **Mid needs alignment:** LiDAR features must land on the *same grid cells* as camera features before concat.
5. **PointNet atom:** per-point MLP + max-pool = permutation-invariant, handles variable points per cell.
6. **BEV:** top-down grid, cell = (x,z), height y goes *inside* the cell. Both branches must speak BEV to fuse there.
7. **Robustness:** blind-mode sweeps prove graceful degradation — the *real* argument for fusion, not just higher AP.

---

# SELF-TEST

> Do this **without** opening the answer appendix or the code. Jot answers on paper, then
> check against **APPENDIX A** at the very bottom. Re-take this in 3 days and again in a week.
> Goal: every question answered in one or two sentences, from memory.

### A. Concepts

1. Name the three coordinate frames your code uses and the matrix chain that connects them.
2. What is the "perspective divide" and why does depth survive as a separate value after projection?
3. In one sentence each: what does early, mid, and late fusion actually combine?
4. Why must the LiDAR feature map be **aligned** to the camera feature grid in mid fusion? What breaks if it isn't?
5. Why does the late-fusion model set `custom_loss = True`? What would go wrong if the trainer applied the standard single-output loss?
6. Late fusion merges two detection sets. Name the three steps your `_merge` does, in order.
7. Why does `render_depth_torch` sort points **far-first** before writing to the image?
8. What does `set_cam_blind(True)` actually do at runtime, and why is the forward pass *not* skipped?
9. What two things must a good fused model demonstrate in the robustness sweep (not just one)?

### B. BEV / 3D

10. In your BEV convention, which two of (x, y, z) determine a point's *cell*, and what happens to the third?
11. A camera pixel alone cannot become a 3D point. How does the camera branch of fusion-3D get around this? Name the mechanism.
12. Why detect in BEV rather than the image plane for driving? Give two reasons.
13. Both branches of fusion-3D produce a BEV feature map. Why is concatenating *those* meaningful, but concatenating camera-image-features with raw LiDAR points would not be?

### C. Deploy / scope honesty

14. For each variant, say how many ONNX engines you export and what moves to C++/CUDA preprocessing.
15. Why does your `README`/guide say the lift-splat in this repo is "simplified" and that 3D-AP should use the KITTI devkit rather than this pipeline? What would be dishonest about claiming otherwise?
16. Why is mid-2D the likely deploy winner over late-2D, even if late is more robust?

### D. "Explain it to me like a reviewer" (say it out loud)

17. "Walk me through what happens to one LiDAR point from the moment it enters mid-2D until it contributes to a box." (Cover: frame convert → project → cell → MLP → pool → concat → conv → head.)
18. "Your project holds backbones and data fixed. Why is that the whole point?" (Fusion gain = variant AP − baseline; if backbones differ you can't attribute gains to fusion.)
19. "When would you *not* want fusion?" (When one sensor is unreliable/degenerate and adds noise/cost with no upside; or when latency/edge budget can't afford the second branch.)

---
---

# APPENDIX A — ANSWERS

> Read only after attempting all questions. Keep it terse; the point is the *concept*, not
> matching my wording.

1. **velo (LiDAR) → cam unrect → cam rect → image.** Chain: `R0 @ V2C` brings velo into the rectified cam frame, then `P2` projects to pixels. Convention: x right, y down, z forward.
2. **Perspective divide** = dividing `(X', Y', Z')` by `Z'` (depth) to get pixel `(u,v)`. Depth survives because it *is* the `Z'` coordinate you divided by — projection doesn't destroy it, it uses it as the divisor.
3. **Early** combines raw RGB + raw LiDAR-as-depth (a 4-channel input). **Mid** combines learned camera features + learned LiDAR features. **Late** combines two independent sets of detections/boxes.
4. Concatenation only makes sense if channel (i,j) of both maps describes the **same physical region**. If unaligned, you'd be mixing "what the camera saw here" with "what LiDAR saw *somewhere else*" — the conv has no meaningful correspondence to learn.
5. Late fusion has **two** heads (cam + lid), each producing its own prediction against the *same* GT. The loss is the **sum of both** detectors' losses. If the trainer applied the standard single-output loss path, it wouldn't know how to handle a dict `{"cam":..., "lid":...}` of predictions.
6. **(a)** compute IoU between camera and LiDAR boxes, **(b)** associate matches above the IoU threshold and fuse their confidences (mean or max), **(c)** NMS the merged list.
7. Multiple points can land on the same pixel. Sorting **far-first** means the **nearest** point is written last and overwrites — so the depth map holds the closest surface, which is the physically correct "what's there."
8. It **zeros** the camera branch's input/features but the forward pass still runs (so the LiDAR branch and the head still produce output). This lets you measure AP with one sensor *removed* — graceful degradation — rather than just crashing.
9. (a) **Fusion gain**: beat both single-sensor baselines with both sensors present. (b) **Graceful degradation**: when one sensor is dropped, AP stays reasonable instead of collapsing.
10. **x** (→column c) and **z** (→row r) determine the cell; **y** (height) is encoded *inside* the cell as a feature (e.g. height channel, or the per-point MLP input), not as a cell coordinate.
11. Back-projection needs a depth, which a pixel lacks. The camera branch **predicts a per-pixel depth distribution** (`depth_head` → softmax over K bins) and back-projects each pixel at each candidate depth, weighting by the predicted probability.
12. (a) Perspective distorts sizes/distances in the image, making 3D box regression hard. (b) Objects occlude/overlap in the image but are separated in top-down BEV, so association and distance estimation are cleaner.
13. Both BEV maps share the **same coordinate system and grid**, so cell (i,j) in each refers to the same patch of ground → concat is meaningful. Camera image features live in the image plane (pixels) while raw LiDAR points live in 3D — different spaces, no cell correspondence, so concat would be geometrically meaningless.
14. **Early:** 1 engine (4-channel detector); depth render in C++. **Mid:** 1 engine (takes pre-scattered LiDAR feat map); point scatter in C++. **Late:** 2 engines + a C++ merge/NMS. **3D:** 1 engine (BEV fuse+head only); lift-splat & pillarize in C++ (v2).
15. The lift-splat here is a simplified direct-splat without frustum pooling, not a faithful LSS. Claiming publication-grade 3D-AP from it would overstate what was built. The honest move: reuse the KITTI devkit/reference 3D-AP eval for real numbers and *state* that this branch is for understanding the data flow, not for final metrics.
16. Late needs **two engines + a C++ merge/NMS** — more latency, memory, and deploy complexity on edge hardware. Mid is one clean engine with the scatter pushed to C++ preprocess, better accuracy-per-cost — even though late's independence is theoretically more robust, that robustness usually isn't worth the edge cost.
17. Point (velo frame) → `R0 @ V2C` to cam frame → project to image pixel + depth (perspective divide) → divide by stride to get a camera-grid **cell** → feed point + intensity into a per-point **MLP** → **max-pool** all points in that cell (scatter-reduce amax) → that's `f_lid`, aligned to the camera grid → **concat** with `f_cam` → **1×1 conv** fuses → **CenterHead2D** predicts heat/offset/size → decode to boxes.
18. If backbones/data/epochs differ, any AP difference could be from the backbone or data, not the fusion. Holding everything fixed means **the only variable is the fusion strategy**, so "fusion gain = variant AP − baseline" is a clean, attributable measurement. That's what makes it a benchmark.
19. When the second sensor is unreliable or degenerate (adds noise with no information gain); when latency/power/memory budget can't afford a second branch; or when the task doesn't benefit from the other modality (e.g. pure 2D texture classification needs no LiDAR). Fusion has a real cost — only worth it when the upside in accuracy/robustness exceeds it.