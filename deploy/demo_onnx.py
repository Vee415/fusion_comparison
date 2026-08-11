"""ONNX deploy demo for late_2d fusion model.

Loads the two ONNX engines (cam + lid), runs inference on a real KITTI frame,
decodes 2D boxes, merges cam + lid detections, and visualizes the result.

This is the deploy reference implementation — it proves the export + decode +
merge pipeline works outside PyTorch, using only onnxruntime + numpy + OpenCV.

Usage:
    python deploy/demo_onnx.py --image data/kitti/training/image_2/000001.png \\
        --velodyne data/kitti_sample/velodyne/000001.bin \\
        --calib data/kitti_sample/calib/000001.txt \\
        --cam-onnx onnx/late_2d_cam.onnx --lid-onnx onnx/late_2d_lid.onnx \\
        --output deploy/demo_output.png
"""
import argparse
import numpy as np
import onnxruntime as ort

# ---- preprocessing ----

def parse_calib(path):
    """Parse a KITTI calib txt -> P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, vals = line.split(":", 1)
            out[key.strip()] = np.array([float(v) for v in vals.split()], np.float64)
    P2 = out["P2"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = out["R0_rect"].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :] = out["Tr_velo_to_cam"].reshape(3, 4)
    return P2, R0, Tr


def preprocess_camera(image_path, H=384, W=1280):
    """cv2 read -> BGR2RGB -> resize -> /255 -> CHW -> (1,3,H,W)."""
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))  # cv2 resize takes (width, height)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img[None]  # add batch dim


def render_lidar_3ch(points, P2, R0, Tr, H=384, W=1280):
    """Render LiDAR points to a 3-channel image [depth, height(z), intensity].

    Mirrors common/sensors/projection.py:render_lidar_3ch_torch.
    Points are in velo frame; project to cam frame then to image pixels.
    """
    n = len(points)
    velo_h = np.hstack([points[:, :3], np.ones((n, 1))])
    cam = (R0 @ Tr @ velo_h.T).T[:, :3]  # (N, 3) in cam frame

    # project to image
    cam_h = np.hstack([cam, np.ones((n, 1))])
    img_pts = (P2 @ cam_h.T).T  # (N, 3)
    depth = img_pts[:, 2]
    valid = depth > 0
    uv = img_pts[valid, :2] / depth[valid, None]

    u = uv[:, 0]
    v = uv[:, 1]
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_idx = np.where(valid)[0][in_bounds]

    u = u[in_bounds].astype(int)
    v = v[in_bounds].astype(int)
    d = depth[valid][in_bounds]
    h = points[valid_idx, 2]      # velo z = height
    inten = points[valid_idx, 3] if points.shape[1] >= 4 else np.zeros(len(valid_idx))

    # far-first so near points overwrite (closest wins per pixel)
    order = np.argsort(-d)
    u, v, d, h, inten = u[order], v[order], d[order], h[order], inten[order]

    img = np.zeros((3, H, W), np.float32)
    img[0, v, u] = d.astype(np.float32)       # depth
    img[1, v, u] = h.astype(np.float32)       # height (velo z)
    img[2, v, u] = inten.astype(np.float32)   # intensity
    return img[None]  # add batch dim


# ---- decode + NMS + merge (mirrors common/geometry/boxes2d.py) ----

def decode_head(heat, off, size, stride=16, k=40, thresh=0.1):
    """Decode CenterNet head outputs -> (boxes, scores).

    heat: (1,1,24,80) raw logits, off: (1,2,24,80), size: (1,2,24,80).
    Returns boxes (N,4) [x1,y1,x2,y2] in pixels, scores (N,).
    """
    heat = 1.0 / (1.0 + np.exp(-heat[0, 0]))  # sigmoid, (24,80)
    flat = heat.flatten()
    if k > len(flat):
        k = len(flat)
    topk_idx = np.argpartition(flat, -k)[-k:]
    scores = flat[topk_idx]

    Hg, Wg = heat.shape
    ys = topk_idx // Wg
    xs = topk_idx % Wg

    # gather offsets and sizes
    off = off[0]  # (2,24,80)
    size = size[0]  # (2,24,80)
    off_y = off[0, ys, xs]
    off_x = off[1, ys, xs]
    h = size[0, ys, xs]
    w = size[1, ys, xs]

    cy = (ys + off_y) * stride
    cx = (xs + off_x) * stride
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    mask = scores > thresh
    return boxes[mask], scores[mask]


def iou2d(a, b):
    """IoU between (N,4) and (M,4) boxes -> (N,M)."""
    N, M = len(a), len(b)
    if N == 0 or M == 0:
        return np.zeros((N, M))
    a, b = a[:, None], b[None]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_thresh=0.45):
    """Greedy NMS -> kept indices."""
    if len(boxes) == 0:
        return np.array([], int)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ovs = iou2d(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][ovs < iou_thresh]
    return np.array(keep, int)


def merge(cam_boxes, cam_scores, lid_boxes, lid_scores, merge_iou=0.5):
    """Late fusion merge: greedy IoU match, mean score, cam box retained."""
    if len(cam_boxes) == 0:
        return lid_boxes, lid_scores
    if len(lid_boxes) == 0:
        return cam_boxes, cam_scores

    iou = iou2d(cam_boxes, lid_boxes)  # (Nc, Nl)
    used = set()
    boxes, scores = [], []

    for i in range(len(cam_boxes)):
        jv = np.where(iou[i] > merge_iou)[0]
        matched = None
        for j in jv:
            if j not in used:
                used.add(j)
                matched = j
                break
        if matched is None:
            boxes.append(cam_boxes[i])
            scores.append(cam_scores[i])
        else:
            boxes.append(cam_boxes[i])
            scores.append((cam_scores[i] + lid_scores[matched]) / 2)

    for j in range(len(lid_boxes)):
        if j not in used:
            boxes.append(lid_boxes[j])
            scores.append(lid_scores[j])

    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = nms(boxes, scores, merge_iou)
    return boxes[keep], scores[keep]


# ---- main pipeline ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="KITTI image .png")
    ap.add_argument("--velodyne", required=True, help="KITTI .bin point cloud")
    ap.add_argument("--calib", required=True, help="KITTI calib .txt")
    ap.add_argument("--cam-onnx", default="onnx/late_2d_cam.onnx")
    ap.add_argument("--lid-onnx", default="onnx/late_2d_lid.onnx")
    ap.add_argument("--output", default="deploy/demo_output.png")
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()

    H, W = 384, 1280

    print("=== loading ONNX engines ===")
    cam_sess = ort.InferenceSession(args.cam_onnx, providers=["CPUExecutionProvider"])
    lid_sess = ort.InferenceSession(args.lid_onnx, providers=["CPUExecutionProvider"])
    print(f"  cam engine: {cam_sess.get_inputs()[0].name} -> {[o.name for o in cam_sess.get_outputs()]}")
    print(f"  lid engine: {lid_sess.get_inputs()[0].name} -> {[o.name for o in lid_sess.get_outputs()]}")

    print("=== loading KITTI frame ===")
    P2, R0, Tr = parse_calib(args.calib)
    cam_input = preprocess_camera(args.image, H, W)
    print(f"  camera image: {cam_input.shape}")
    points = np.fromfile(args.velodyne, dtype=np.float32).reshape(-1, 4)
    print(f"  points: {points.shape}")
    lid_input = render_lidar_3ch(points, P2, R0, Tr, H, W)
    print(f"  lidar image: {lid_input.shape}")

    print("=== running inference ===")
    cam_out = cam_sess.run(None, {"image": cam_input})
    lid_out = lid_sess.run(None, {"lidar_image": lid_input})
    cam_heat, cam_off, cam_size = cam_out[0], cam_out[1], cam_out[2]
    lid_heat, lid_off, lid_size = lid_out[0], lid_out[1], lid_out[2]
    print(f"  cam heat: {cam_heat.shape}, off: {cam_off.shape}, size: {cam_size.shape}")
    print(f"  cam heat sigmoid max: {1/(1+np.exp(-cam_heat)).max():.4f}")
    print(f"  lid heat sigmoid max: {1/(1+np.exp(-lid_heat)).max():.4f}")

    print("=== decoding ===")
    cam_boxes, cam_scores = decode_head(cam_heat, cam_off, cam_size, args.stride, thresh=args.thresh)
    lid_boxes, lid_scores = decode_head(lid_heat, lid_off, lid_size, args.stride, thresh=args.thresh)
    print(f"  cam detections: {len(cam_boxes)} (after thresh {args.thresh})")
    print(f"  lid detections: {len(lid_boxes)} (after thresh {args.thresh})")

    print("=== NMS per stream ===")
    cam_keep = nms(cam_boxes, cam_scores, 0.45)
    lid_keep = nms(lid_boxes, lid_scores, 0.45)
    cam_boxes, cam_scores = cam_boxes[cam_keep], cam_scores[cam_keep]
    lid_boxes, lid_scores = lid_boxes[lid_keep], lid_scores[lid_keep]
    print(f"  cam after NMS: {len(cam_boxes)}")
    print(f"  lid after NMS: {len(lid_boxes)}")

    print("=== merging cam + lid ===")
    merged_boxes, merged_scores = merge(cam_boxes, cam_scores, lid_boxes, lid_scores, 0.5)
    print(f"  merged detections: {len(merged_boxes)}")

    print("=== visualizing ===")
    import cv2
    img = cv2.imread(args.image)
    img = cv2.resize(img, (W, H))

    # draw cam detections (blue)
    for box, score in zip(cam_boxes, cam_scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"cam {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # draw lid detections (green)
    for box, score in zip(lid_boxes, lid_scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"lid {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # draw merged detections (red, thicker)
    for box, score in zip(merged_boxes, merged_scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"merge {score:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(args.output, img)
    print(f"saved {args.output}")
    print(f"\n=== summary ===")
    print(f"  cam detections:  {len(cam_boxes)}")
    print(f"  lid detections:  {len(lid_boxes)}")
    print(f"  merged:          {len(merged_boxes)}")
    print(f"\nLegend: blue=cam, green=lidar, red=merged (final output)")


if __name__ == "__main__":
    main()