"""The shared interface. Every variant implements this so the trainer/eval/export
code is written ONCE.

forward(batch) -> dict of raw head outputs (used for the loss).
output_space() -> "2d" | "3d" (trainer picks the loss; eval picks the metric suite).
decode(pred, cfg) -> per-frame list of {boxes, scores, labels} (used for eval).
build_target(batch, cfg, device) -> dict of target tensors (encoded GT).
export_onnx(path, cfg, device) -> writes ONNX (variant-specific input signature).

Robustness hooks (eval/robustness.py calls these): set_cam_blind(True) / set_lidar_blind(True)
zero the corresponding branch so forward still runs with one sensor dropped.
"""
import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._cam_blind = False
        self._lidar_blind = False
        self.custom_loss = False  # variants with non-standard preds (late) override

    # ---- to implement ----
    def forward(self, batch): raise NotImplementedError
    def output_space(self): raise NotImplementedError
    def decode(self, pred, cfg): raise NotImplementedError
    def build_target(self, batch, cfg, device): raise NotImplementedError
    def export_onnx(self, path, cfg, device): raise NotImplementedError

    # ---- robustness hooks ----
    def set_cam_blind(self, flag=True): self._cam_blind = flag
    def set_lidar_blind(self, flag=True): self._lidar_blind = flag
    def reset_blind(self):
        self._cam_blind = False
        self._lidar_blind = False