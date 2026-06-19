"""2D boxes: CenterNet-style encode/decode + IoU + NMS.

Grid is Hg x Wg = (H/stride) x (W/stride). Box center in pixel coords -> cell (cy, cx).
Targets: heatmap (C,Hg,Wg) gaussian peaks; offset (2,Hg,Wg) sub-cell (dy,dx); size (2,Hg,Wg) (h,w) px.
"""
import numpy as np
import torch


def gaussian_radius(h, w, min_overlap=0.7):
    """CenterNet-style radius so a box and its center-box have >= min_overlap IoU."""
    a1 = 1
    b1 = h + w
    a2 = 4
    b2 = 2 * (h + w) + 1
    c2 = 4 * min_overlap
    dc = b1 ** 2 - 4 * a2 * (c2 + w * h)
    dr = b2 ** 2 - 16 * c2 * w * h
    if dc <= 0:
        rc = 0.0
    else:
        rc = (b1 + np.sqrt(dc)) / (2 * a2)
    if dr <= 0:
        rr = 0.0
    else:
        rr = (b2 + np.sqrt(dr)) / (2 * a2)
    return max(0, int(min(rc, rr)))


def _draw_gaussian(heatmap, center, radius, k=1):
    """In-place 2D gaussian on heatmap (C,Hg,Wg) at integer center (cy,cx)."""
    import torch.nn.functional as F
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    g = torch.tensor(
        np.exp(-((np.arange(diameter) - radius) ** 2) / (2 * sigma ** 2)),
        dtype=heatmap.dtype, device=heatmap.device,
    )
    g2d = torch.outer(g, g)
    cy, cx = int(center[0]), int(center[1])
    h, w = heatmap.shape[-2:]
    y0 = max(0, cy - radius); y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius); x1 = min(w, cx + radius + 1)
    gy0 = y0 - (cy - radius); gy1 = gy0 + (y1 - y0)
    gx0 = x0 - (cx - radius); gx1 = gx0 + (x1 - x0)
    if y1 > y0 and x1 > x0:
        heatmap[..., y0:y1, x0:x1] = torch.maximum(
            heatmap[..., y0:y1, x0:x1], k * g2d[gy0:gy1, gx0:gx1]
        )


def encode_boxes2d(boxes, labels, Hg, Wg, stride, num_classes, device, min_overlap=0.7):
    """boxes: (M,4) x1y1x2y2 pixels; labels: (M,) long. -> dict of target tensors."""
    heat = torch.zeros((num_classes, Hg, Wg), device=device)
    off = torch.zeros((2, Hg, Wg), device=device)
    size = torch.zeros((2, Hg, Wg), device=device)
    if boxes.shape[0] == 0:
        return {"heat": heat, "off": off, "size": size}
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    gcy = (cy / stride).long().clamp(0, Hg - 1)
    gcx = (cx / stride).long().clamp(0, Wg - 1)
    off_y = (cy / stride) - gcy.float()
    off_x = (cx / stride) - gcx.float()
    for i in range(boxes.shape[0]):
        r = gaussian_radius(h[i].item(), w[i].item(), min_overlap)
        _draw_gaussian(heat[labels[i]], (gcy[i].item(), gcx[i].item()), r)
    # last writer for off/size at each cell (fine for non-overlapping centers)
    off[0, gcy, gcx] = off_y
    off[1, gcy, gcx] = off_x
    size[0, gcy, gcx] = h
    size[1, gcy, gcx] = w
    return {"heat": heat, "off": off, "size": size}


def _topk_heatmap(heat, k=40):
    """heat: (B,C,Hg,Wg) -> scores (B,k), cls (B,k), indices (B,k,2) as (y,x) cell."""
    B, C, H, W = heat.shape
    heat = heat.view(B, C, -1)
    scores, idx = heat.topk(k, dim=-1)
    topk_scores, topk_idx = scores.view(B, -1), idx.view(B, -1)
    topk_cls = (topk_idx // (H * W)).long()
    topk_cell = topk_idx % (H * W)
    topk_y = (topk_cell // W).long()
    topk_x = (topk_cell % W).long()
    return topk_scores, topk_cls, topk_y, topk_x


def decode_boxes2d(pred, stride, k=40, thresh=0.1):
    """pred: dict with heat (B,C,Hg,Wg), off (B,2,..), size (B,2,..) -> list per batch of
    {boxes:(N,4) x1y1x2y2 px, scores:(N,), labels:(N,)}."""
    heat = pred["heat"].sigmoid()
    off = pred["off"]; size = pred["size"]
    B, C, Hg, Wg = heat.shape
    scores, labels, ys, xs = _topk_heatmap(heat, k)
    off = off.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
    size = size.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
    idx = ys * Wg + xs
    off_sel = torch.gather(off, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    size_sel = torch.gather(size, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    cy = (ys.float() + off_sel[..., 0]) * stride
    cx = (xs.float() + off_sel[..., 1]) * stride
    h = size_sel[..., 0]; w = size_sel[..., 1]
    out = []
    for b in range(B):
        m = scores[b] > thresh
        s = scores[b][m]
        x1 = cx[b][m] - w[b][m] / 2; y1 = cy[b][m] - h[b][m] / 2
        x2 = cx[b][m] + w[b][m] / 2; y2 = cy[b][m] + h[b][m] / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        out.append({"boxes": boxes, "scores": s, "labels": labels[b][m]})
    return out


# ---------------- IoU + NMS (numpy, for eval) ----------------

def iou2d(a, b):
    """a: (N,4), b: (M,4) -> (N,M) IoU."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    xa1, ya1, xa2, ya2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    xb1, yb1, xb2, yb2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(xa1, xb1); inter_y1 = np.maximum(ya1, yb1)
    inter_x2 = np.minimum(xa2, xb2); inter_y2 = np.minimum(ya2, yb2)
    iw = (inter_x2 - inter_x1).clip(min=0); ih = (inter_y2 - inter_y1).clip(min=0)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1); area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-9)


def nms2d(boxes, scores, iou_thresh=0.45):
    """boxes (N,4), scores (N,) -> kept indices."""
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
        ovs = iou2d(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ovs < iou_thresh]
    return np.array(keep, dtype=int)