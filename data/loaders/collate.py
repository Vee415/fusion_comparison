"""Collate a list of per-sample dicts into a batched dict."""
import torch


def collate_paired(samples):
    batch = {
        "image": torch.stack([s["image"] for s in samples], dim=0),
        "points": [s["points"] for s in samples],
        "calib": [s["calib"] for s in samples],
        "image_id": [s["image_id"] for s in samples],
        "boxes2d": [s.get("boxes2d") for s in samples],
        "boxes2d_cls": [s.get("boxes2d_cls") for s in samples],
        "boxes3d": [s.get("boxes3d") for s in samples],
    }
    return batch