"""Variant A -- Early-2D (data-level fusion).

Fuse at input: project LiDAR into the image as a depth channel, append to RGB -> (H,W,4).
The backbone's first conv accepts 4 channels. Output: 2D boxes.
Clean to export: the depth rendering is preprocessing (do in C++/CUDA at deploy), so the
network is just a 4-channel detector.
"""
import torch
import torch.nn as nn

from common.backbones.image_backbone import ImageBackbone
from common.sensors.projection import render_depth_torch
from fusion.base import FusionModel
from fusion.heads import CenterHead2D
from fusion.common_2d import build_target_2d, decode_2d


class EarlyFusion2D(FusionModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stride = cfg["stride"]
        extra = cfg.get("extra_input_channels", 1)
        self.backbone = ImageBackbone(in_channels=3 + extra, out_channels=cfg["fusion_channels"])
        self.head = CenterHead2D(cfg["fusion_channels"], num_classes=cfg["num_classes"])

    def _build_input(self, batch, device):
        B, _, H, W = batch["image"].shape
        imgs = batch["image"]                              # (B,3,H,W) 0-1
        depth = []
        for b in range(B):
            pts = batch["points"][b].to(device).float()
            if self._lidar_blind or pts.shape[0] == 0:
                d = torch.zeros((H, W), device=device, dtype=imgs.dtype)
            else:
                d = render_depth_torch(pts, batch["calib"][b], H, W, device, imgs.dtype)
                d = (d / 80.0).clamp(0, 1)                  # normalize depth to ~[0,1]
            depth.append(d)
        depth = torch.stack(depth, dim=0).unsqueeze(1)    # (B,1,H,W)
        if self._cam_blind:
            imgs = torch.zeros_like(imgs)
        return torch.cat([imgs, depth], dim=1)             # (B,4,H,W)

    def forward(self, batch):
        device = batch["image"].device
        x = self._build_input(batch, device)
        feat = self.backbone(x)[str(self.stride)]
        return self.head(feat)

    def output_space(self): return "2d"
    def build_target(self, batch, cfg, device): return build_target_2d(batch, cfg, device)
    def decode(self, pred, cfg, k=40, thresh=0.1): return decode_2d(pred, cfg, k, thresh)

    def export_onnx(self, path, cfg, device):
        B = 1; H, W = cfg["image_size"]
        dummy = torch.zeros((B, 4, H, W), device=device)
        stride = self.stride

        class _Wrap(nn.Module):
            def __init__(s_): super().__init__(); s_.backbone = self.backbone; s_.head = self.head
            def forward(s_, x):
                return s_.head(s_.backbone(x)[str(stride)])

        torch.onnx.export(_Wrap(), dummy, path, input_names=["input_4ch"], output_names=["heat", "off", "size"],
                          dynamic_axes={"input_4ch": {0: "B"}}, opset_version=17)
        return path