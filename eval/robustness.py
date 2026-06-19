"""Robustness: re-eval a fusion variant with one sensor blinded.

cam_blind: zero the camera branch (image). lidar_blind: drop the LiDAR branch.
The point of fusion is graceful degradation -- a variant that keeps AP when one sensor
dies beats one that craters. This is the differentiator row in the benchmark table.
"""
import torch

from eval.infer import run_inference
from eval.metrics_2d import evaluate_detections_2d
from eval.metrics_3d import evaluate_detections_3d


def eval_blind(model, loader, cfg, device, blind):
    model.reset_blind()
    if blind == "cam":
        model.set_cam_blind(True)
    elif blind == "lidar":
        model.set_lidar_blind(True)
    dets, gts, space = run_inference(model, loader, cfg, device)
    model.reset_blind()
    if space == "2d":
        return evaluate_detections_2d(dets, gts)["AP"]
    return evaluate_detections_3d(dets, gts)["AP_BEV"]