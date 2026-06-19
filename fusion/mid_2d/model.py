"""Variant B -- Mid-2D (feature-level fusion). The modern sweet spot; likely deploy winner.

Image backbone -> F_cam (B, C, Hg, Wg) at stride 16.
LiDAR -> image-grid encoder (project points, per-point MLP, max-pool per cell) -> F_lid (B, Cl, Hg, Wg)
  ALIGNED to the camera grid so cells describe the same region.
Fuse: concat([F_cam, F_lid]) + 1x1 conv -> F_fused. Head -> 2D boxes.

Deploy note: the lidar scatter is awkward to export, so export takes a pre-scattered
lidar feature map (do the scatter in C++/CUDA at deploy). Same architecture otherwise.
"""
import torch
import torch.nn as nn

from common.backbones.image_backbone import ImageBackbone
from common.backbones.lidar_encoder import LidarImageGridEncoder
from fusion.base import FusionModel
from fusion.heads import CenterHead2D
from fusion.common_2d import build_target_2d, decode_2d


class MidFusion2D(FusionModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stride = cfg["stride"]
        H, W = cfg["image_size"]
        self.H, self.W = H, W
        self.image_backbone = ImageBackbone(in_channels=3, out_channels=cfg["fusion_channels"])
        self.lidar_encoder = LidarImageGridEncoder(out_channels=cfg["lidar_feat_channels"], stride=self.stride)
        self.fuse = nn.Conv2d(cfg["fusion_channels"] + cfg["lidar_feat_channels"], cfg["fusion_channels"], 1)
        self.head = CenterHead2D(cfg["fusion_channels"], num_classes=cfg["num_classes"])

    def forward(self, batch):
        device = batch["image"].device
        B = batch["image"].shape[0]
        if self._cam_blind:
            img = torch.zeros_like(batch["image"])
        else:
            img = batch["image"]
        f_cam = self.image_backbone(img)[str(self.stride)]     # (B,C,Hg,Wg)
        if self._lidar_blind:
            f_lid = torch.zeros(B, self.lidar_encoder.out_channels, f_cam.shape[2], f_cam.shape[3], device=device, dtype=f_cam.dtype)
        else:
            f_lid = self.lidar_encoder(batch["points"], batch["calib"], self.H, self.W, device)
        f_lid = f_lid.to(f_cam.dtype)                                  # match image-branch dtype (AMP)
        fused = self.fuse(torch.cat([f_cam, f_lid], dim=1))
        return self.head(fused)

    def output_space(self): return "2d"
    def build_target(self, batch, cfg, device): return build_target_2d(batch, cfg, device)
    def decode(self, pred, cfg, k=40, thresh=0.1): return decode_2d(pred, cfg, k, thresh)

    def export_onnx(self, path, cfg, device):
        B = 1; H, W = cfg["image_size"]; s = cfg["stride"]
        Hg, Wg = H // s, W // s
        img = torch.zeros((B, 3, H, W), device=device)
        lid = torch.zeros((B, cfg["lidar_feat_channels"], Hg, Wg), device=device)
        class _Wrap(nn.Module):
            def __init__(s_): super().__init__(); s_.net = self
            def forward(s_, i, l):
                f = self.image_backbone(i)[str(self.stride)]
                fused = self.fuse(torch.cat([f, l], dim=1))
                return self.head(fused)
        torch.onnx.export(_Wrap(), (img, lid), path,
                          input_names=["image", "lidar_feat"], output_names=["heat", "off", "size"],
                          dynamic_axes={"image": {0: "B"}, "lidar_feat": {0: "B"}}, opset_version=17)
        return path