"""3D boxes (KITTI cam frame: x right, y down, z forward).

Box = (x, y, z, w, h, l, yaw): center (x,y,z), extents (w=x, h=y(down), l=z(forward)),
yaw = heading in the BEV (x,z) plane measured from +x axis CCW.

BEV grid: dim0 (r) = z (forward), dim1 (c) = x (right). cell center:
    z = -range + (r+0.5)*res,  x = -range + (c+0.5)*res
CenterPoint-style targets: heatmap (C,Hg,Wg) at (z->r, x->c); offset (2) sub-cell (dz,dx);
height (1) = y; size (3) = (w,h,l); yaw (2) = (sin,cos).
"""
import numpy as np
import torch

try:
    from shapely.geometry import Polygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# ---------------- encode / decode ----------------

def encode_boxes3d(boxes, labels, Hg, Wg, res, range_m, num_classes, device, min_overlap=0.7):
    """boxes: (M,7) [x,y,z,w,h,l,yaw] cam frame -> target dict."""
    from common.geometry.boxes2d import _draw_gaussian, gaussian_radius
    heat = torch.zeros((num_classes, Hg, Wg), device=device)
    off = torch.zeros((2, Hg, Wg), device=device)     # dz, dx
    height = torch.zeros((1, Hg, Wg), device=device)
    size = torch.zeros((3, Hg, Wg), device=device)     # w, h, l
    yawc = torch.zeros((2, Hg, Wg), device=device)     # sin, cos
    if boxes.shape[0] == 0:
        return {"heat": heat, "off": off, "height": height, "size": size, "yaw": yawc}
    x, y, z, w, h, l, yaw = [boxes[:, i] for i in range(7)]
    r = ((z + range_m) / res).long().clamp(0, Hg - 1)   # forward index
    c = ((x + range_m) / res).long().clamp(0, Wg - 1)   # right index
    off_z = (z + range_m) / res - r.float()
    off_x = (x + range_m) / res - c.float()
    for i in range(boxes.shape[0]):
        # gaussian radius from BEV footprint (l x w)
        rad = gaussian_radius(max(h[i].item(), 1.0), max(l[i].item(), 1.0), min_overlap)
        _draw_gaussian(heat[labels[i]], (r[i].item(), c[i].item()), rad)
    off[0, r, c] = off_z; off[1, r, c] = off_x
    height[0, r, c] = y
    size[0, r, c] = w; size[1, r, c] = h; size[2, r, c] = l
    yawc[0, r, c] = torch.sin(yaw); yawc[1, r, c] = torch.cos(yaw)
    return {"heat": heat, "off": off, "height": height, "size": size, "yaw": yawc}


def decode_boxes3d(pred, res, range_m, k=40, thresh=0.1):
    """pred: heat (B,C,Hg,Wg), off (B,2,..), height (B,1,..), size (B,3,..), yaw (B,2,..)
    -> per-batch list of {boxes:(N,7), scores:(N,), labels:(N,)} in cam frame."""
    from common.geometry.boxes2d import _topk_heatmap
    heat = pred["heat"].sigmoid()
    B, C, Hg, Wg = heat.shape
    scores, labels, ys, xs = _topk_heatmap(heat, k)   # ys=r (forward), xs=c (right)
    def gather(t):
        t = t.permute(0, 2, 3, 1).contiguous().view(B, -1, t.shape[1])
        idx = ys * Wg + xs
        return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
    off = gather(pred["off"]); height = gather(pred["height"])
    size = gather(pred["size"]); yaw = gather(pred["yaw"])
    z = (ys.float() + off[..., 0]) * res - range_m
    x = (xs.float() + off[..., 1]) * res - range_m
    y = height[..., 0]
    w, h, l = size[..., 0], size[..., 1], size[..., 2]
    angle = torch.atan2(yaw[..., 0], yaw[..., 1])
    out = []
    for b in range(B):
        m = scores[b] > thresh
        s = scores[b][m]
        box = torch.stack([x[b][m], y[b][m], z[b][m], w[b][m], h[b][m], l[b][m], angle[b][m]], dim=-1)
        out.append({"boxes": box, "scores": s, "labels": labels[b][m]})
    return out


# ---------------- rotated BEV IoU + NMS (shapely) ----------------

def _bev_polygon(x, z, w, l, yaw):
    """Rectangle footprint in (x,z) BEV. w=x-extent, l=z-extent, yaw from +x CCW."""
    dx = np.array([w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([l / 2, l / 2, -l / 2, -l / 2])
    cos, sin = np.cos(yaw), np.sin(yaw)
    rx = x + dx * cos - dz * sin
    rz = z + dx * sin + dz * cos
    return Polygon(list(zip(rx, rz)))


def rotated_bev_iou(a, b):
    """a: (N,7), b: (M,7) boxes [x,y,z,w,h,l,yaw] -> (N,M) BEV IoU."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    out = np.zeros((a.shape[0], b.shape[0]))
    if not _HAS_SHAPELY or a.shape[0] == 0 or b.shape[0] == 0:
        return out
    pa = [_bev_polygon(*ab[[0, 2, 3, 5, 6]]) for ab in a]
    pb = [_bev_polygon(*bb[[0, 2, 3, 5, 6]]) for bb in b]
    for i, ai in enumerate(pa):
        for j, bj in enumerate(pb):
            inter = ai.intersection(bj).area
            union = ai.area + bj.area - inter
            out[i, j] = inter / union if union > 1e-9 else 0.0
    return out


def rotated_nms3d(boxes, scores, iou_thresh=0.1):
    """boxes (N,7), scores (N,) -> kept indices (BEV rotated IoU)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        ovs = rotated_bev_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ovs < iou_thresh]
    return np.array(keep, dtype=int)