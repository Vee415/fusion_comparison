"""3D AP-BEV. HONEST SCOPE: this is a simplified BEV-IoU AP fallback, NOT official KITTI AP-3D.

For real numbers, install the official KITTI devkit and use its eval (see evaluate_official()).
The benchmark is comparative across variants under the SAME eval, so the fallback is fair for
measuring fusion gain -- but cite it as "AP-BEV (simplified, IoU 0.7)", not "KITTI AP-3D".
"""
import numpy as np
from common.geometry.boxes3d import rotated_bev_iou


def _match_frame_3d(det_boxes, det_scores, gt_boxes, iou_thresh=0.7):
    order = np.argsort(-det_scores)
    gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
    out = []
    for idx in order:
        if gt_boxes.shape[0] == 0:
            out.append((det_scores[idx], "fp")); continue
        ious = rotated_bev_iou(det_boxes[idx:idx + 1], gt_boxes)[0]
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


def evaluate_detections_3d(dets, gts, iou_thresh=0.7):
    all_scores, all_tp = [], []
    n_gt = 0
    for fid in gts:
        gt = gts[fid].numpy() if hasattr(gts[fid], "numpy") else np.asarray(gts[fid])
        gt = gt.reshape(-1, 7)
        d = dets.get(fid, {"boxes": np.zeros((0, 7)), "scores": np.zeros((0,))})
        db = d["boxes"].numpy() if hasattr(d["boxes"], "numpy") else np.asarray(d["boxes"])
        ds = d["scores"].numpy() if hasattr(d["scores"], "numpy") else np.asarray(d["scores"])
        db = db.reshape(-1, 7)
        res, _ = _match_frame_3d(db, ds, gt, iou_thresh)
        for s, kind in res:
            all_scores.append(s); all_tp.append(1 if kind == "tp" else 0)
        n_gt += gt.shape[0]
    if n_gt == 0:
        return {"AP_BEV": 0.0, "n_gt": 0}
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]; fp = 1 - tp
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    rec_levels = np.linspace(0, 1, 40); ap = 0.0
    for r in rec_levels:
        p = precision[recall >= r].max() if (recall >= r).any() else 0.0
        ap += p
    return {"AP_BEV": float(ap / 40.0), "n_gt": int(n_gt)}


def evaluate_official(result_dir, gt_dir, split="val"):
    """Wrap the official KITTI devkit if installed on path. Returns Easy/Mod/Hard AP-3D/AP-BEV.

    Put the KITTI devkit (kitti_eval or OpenPCDet's eval) on PYTHONPATH and call its entry.
    This function tries a couple of common import paths; if none work it raises with guidance.
    """
    try:
        from kitti_eval import eval as kitti_eval  # type: ignore
    except Exception:
        raise ImportError(
            "Official KITTI 3D eval not found. Install the KITTI devkit "
            "(https://github.com/prclibo/kitti_eval) and put it on PYTHONPATH, or use "
            "OpenPCDet's eval. Until then, evaluate_detections_3d() gives a simplified AP-BEV "
            "for relative comparison only -- NOT for citing as KITTI AP-3D."
        )
    return kitti_eval(result_dir, gt_dir, split)