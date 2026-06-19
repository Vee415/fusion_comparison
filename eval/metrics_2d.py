"""2D AP. Simple, honest: AP @ IoU 0.5 via 40-point interpolation (KITTI-style averaging).

NOTE: this is a clean stand-in, not the official KITTI Easy/Moderate/Hard AP (which needs
per-box difficulty from truncation/occlusion/height). For publication-grade 2D AP, adapt
pycocotools. The benchmark is comparative across variants under the SAME eval, so this is
fair for measuring fusion gain.
"""
import numpy as np
from common.geometry.boxes2d import iou2d


def _match_frame(det_boxes, det_scores, gt_boxes, iou_thresh=0.5):
    """Greedy match dets->gts by score. Returns list of (score, tp/fp)."""
    order = np.argsort(-det_scores)
    gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
    out = []
    for idx in order:
        if gt_boxes.shape[0] == 0:
            out.append((det_scores[idx], "fp")); continue
        ious = iou2d(det_boxes[idx:idx + 1], gt_boxes)[0]
        best = -1; best_iou = iou_thresh
        for g in range(gt_boxes.shape[0]):
            if gt_used[g]: continue
            if ious[g] > best_iou:
                best_iou = ious[g]; best = g
        if best >= 0:
            gt_used[best] = True; out.append((det_scores[idx], "tp"))
        else:
            out.append((det_scores[idx], "fp"))
    return out, int(gt_used.sum())


def ap_40point(precision, recall):
    rec_levels = np.linspace(0, 1, 40)
    ap = 0.0
    for r in rec_levels:
        p = precision[recall >= r].max() if (recall >= r).any() else 0.0
        ap += p
    return ap / 40.0


def evaluate_detections_2d(dets, gts, iou_thresh=0.5):
    """dets/gts: dict frame_id -> {boxes (N,4), scores (N,)}. Returns dict with AP, P, R."""
    all_scores, all_tp = [], []
    n_gt = 0; n_tp = 0
    for fid in gts:
        gt = gts[fid].numpy() if hasattr(gts[fid], "numpy") else np.asarray(gts[fid])
        gt = gt.reshape(-1, 4)
        d = dets.get(fid, {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,))})
        db = d["boxes"].numpy() if hasattr(d["boxes"], "numpy") else np.asarray(d["boxes"])
        ds = d["scores"].numpy() if hasattr(d["scores"], "numpy") else np.asarray(d["scores"])
        db = db.reshape(-1, 4)
        res, matched = _match_frame(db, ds, gt, iou_thresh)
        for s, kind in res:
            all_scores.append(s); all_tp.append(1 if kind == "tp" else 0)
        n_gt += gt.shape[0]; n_tp += matched
    if n_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": 0}
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = ap_40point(precision, recall)
    return {"AP": float(ap), "precision": float(precision[-1]) if len(precision) else 0.0,
            "recall": float(recall[-1]) if len(recall) else 0.0, "n_gt": int(n_gt)}