"""Variant D -- Fusion-3D (BEV feature fusion). The canonical automotive approach.

Camera branch: image backbone -> features -> simplified lift-splat-shoot
    (predict per-pixel depth distribution, lift each pixel to 3D, splat into BEV) -> BEV_cam.
LiDAR branch: simplified PointPillars (points -> pillars -> per-point MLP -> max-pool -> BEV) -> BEV_lid.
Fuse in BEV: concat([BEV_cam, BEV_lid]) + conv -> CenterPoint head -> 3D boxes (x,y,z,w,h,l,yaw).

Scope (honest): the lift-splat here is a SIMPLIFIED version (direct splat, no frustum pooling).
For real 3D-AP, reuse the KITTI devkit (see eval/metrics_3d.py) -- do not trust this simplified
pipeline for publication numbers. The build-vs-reuse split is documented in the guide.
Deploy (v2): export only the BEV fusion + head; lift-splat/pillarize are CUDA preprocess
(matching production PointPillars-TRT pipelines). Full lift-splat in C++ = v2/v3.
"""
import torch
import torch.nn as nn

from common.backbones.image_backbone import ImageBackbone
from common.backbones.lidar_encoder import PillarBEVEncoder
from common.geometry.bev import grid_shape
from fusion.base import FusionModel
from fusion.heads import CenterHead3D
from fusion.common_3d import build_target_3d, decode_3d

CAM_BEV = 64
LID_BEV = 64
FUSED_BEV = 128


class Fusion3D(FusionModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        bev = cfg.get("bev", cfg)
        self.range_m = float(bev["range_m"]); self.res = float(bev["res_m"])
        self.Hg, self.Wg = grid_shape(self.range_m, self.res)
        self.s = cfg["stride"]                      # image feature stride for lift
        self.K = cfg.get("lift", {}).get("depth_bins", 16)
        self.depth_max = cfg.get("lift", {}).get("depth_max_m", 40.0)

        self.image_backbone = ImageBackbone(in_channels=3, out_channels=cfg["fusion_channels"])
        self.depth_head = nn.Conv2d(cfg["fusion_channels"], self.K, 1)
        self.cam_reduce = nn.Conv2d(cfg["fusion_channels"], CAM_BEV, 1)
        self.lidar_encoder = PillarBEVEncoder(out_channels=LID_BEV)
        self.fuse = nn.Conv2d(CAM_BEV + LID_BEV, FUSED_BEV, 1)
        self.head = CenterHead3D(FUSED_BEV, num_classes=cfg["num_classes"],
                                 predict_velocity=cfg.get("head_3d", {}).get("predict_velocity", False))

    def _lift_splat(self, feat, calib, device, dtype):
        """feat: (Cf,Hf,Wf) -> BEV_cam (CAM_BEV, Hg, Wg). Simplified LSS (direct splat)."""
        Cf, Hf, Wf = feat.shape
        feat = feat.float()                                          # scatter in fp32 (AMP-safe)
        s = self.s
        m = calib.torch_matrices(device, feat.dtype)
        fx, fy, cx, cy = m["P"][0, 0], m["P"][1, 1], m["P"][0, 2], m["P"][1, 2]
        weights = torch.softmax(self.depth_head(feat.unsqueeze(0)).squeeze(0), dim=0)  # (K,Hf,Wf)
        # pixel -> image coords
        us = (torch.arange(Wf, device=device, dtype=dtype) * s + s / 2).view(1, Wf).expand(Hf, Wf).reshape(-1)
        vs = (torch.arange(Hf, device=device, dtype=dtype) * s + s / 2).view(Hf, 1).expand(Hf, Wf).reshape(-1)
        bin_c = (torch.arange(self.K, device=device, dtype=dtype) + 0.5) * (self.depth_max / self.K)
        # (K,P)
        d = bin_c.view(self.K, 1).expand(self.K, Hf * Wf)
        u = us.view(1, -1).expand(self.K, Hf * Wf)
        v = vs.view(1, -1).expand(self.K, Hf * Wf)
        x = (u - cx) * d / fx; y = (v - cy) * d / fy; z = d
        r = (z + self.range_m) / self.res; c = (x + self.range_m) / self.res
        valid = (z > 0) & (r >= 0) & (r < self.Hg) & (c >= 0) & (c < self.Wg)
        valid = valid.reshape(-1)
        if valid.sum() == 0:
            bev = torch.zeros(Cf, self.Hg * self.Wg, device=device, dtype=feat.dtype)
            return self.cam_reduce(bev.view(Cf, self.Hg, self.Wg))
        w = weights.reshape(self.K, Hf * Wf).reshape(-1)[valid]                     # (M,)
        p = torch.arange(Hf * Wf, device=device).view(1, -1).expand(self.K, Hf * Wf).reshape(-1)[valid]
        feat_flat = feat.view(Cf, Hf * Wf)
        src = feat_flat[:, p] * w                                   # (Cf, M)
        lin = (r.reshape(-1)[valid].long() * self.Wg + c.reshape(-1)[valid].long())
        bev = torch.zeros(Cf, self.Hg * self.Wg, device=device, dtype=feat.dtype)
        bev.scatter_add_(1, lin.unsqueeze(0).expand(Cf, -1), src)
        bev = bev.view(Cf, self.Hg, self.Wg)
        return self.cam_reduce(bev)                                  # (CAM_BEV, Hg, Wg)

    def forward(self, batch):
        device = batch["image"].device
        dtype = batch["image"].dtype
        B = batch["image"].shape[0]
        img = torch.zeros_like(batch["image"]) if self._cam_blind else batch["image"]
        feats = self.image_backbone(img)[str(self.s)]               # (B,Cf,Hf,Wf)
        bev_cam = []
        for b in range(B):
            bev_cam.append(self._lift_splat(feats[b], batch["calib"][b], device, dtype))
        bev_cam = torch.stack(bev_cam, dim=0)                       # (B,CAM_BEV,Hg,Wg)
        if self._lidar_blind:
            bev_lid = torch.zeros(B, LID_BEV, self.Hg, self.Wg, device=device, dtype=bev_cam.dtype)
        else:
            bev_lid = self.lidar_encoder(batch["points"], batch["calib"], self.cfg, device)
        bev_lid = bev_lid.to(bev_cam.dtype)                          # match camera-branch dtype (AMP)
        fused = self.fuse(torch.cat([bev_cam, bev_lid], dim=1))
        return self.head(fused)

    def output_space(self): return "3d"
    def build_target(self, batch, cfg, device): return build_target_3d(batch, cfg, device)
    def decode(self, pred, cfg, k=40, thresh=0.1): return decode_3d(pred, cfg, k, thresh)

    def export_onnx(self, path, cfg, device):
        # Export only the BEV-fusion + head (lift-splat/pillarize are CUDA preprocess at deploy).
        bev = cfg.get("bev", cfg); Hg, Wg = grid_shape(float(bev["range_m"]), float(bev["res_m"]))
        bev_cam = torch.zeros((1, CAM_BEV, Hg, Wg), device=device)
        bev_lid = torch.zeros((1, LID_BEV, Hg, Wg), device=device)

        class _Wrap(nn.Module):
            def __init__(s_): super().__init__(); s_.fuse = self.fuse; s_.head = self.head
            def forward(s_, a, b):
                return s_.head(s_.fuse(torch.cat([a, b], dim=1)))

        torch.onnx.export(_Wrap(), (bev_cam, bev_lid), path,
                          input_names=["bev_cam", "bev_lid"], output_names=["heat", "off", "height", "size", "yaw"],
                          dynamic_axes={"bev_cam": {0: "B"}, "bev_lid": {0: "B"}}, opset_version=17)
        return path