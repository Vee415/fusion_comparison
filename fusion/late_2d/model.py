"""Variant C -- Late-2D (decision-level fusion). Two independent detectors -> merge.

Camera detector: image -> ImageBackbone -> 2D head.
LiDAR detector: render a 3-channel [depth,height,intensity] image -> ImageBackbone -> 2D head.
Merge (eval only): associate by IoU, fuse confidence (mean or max), NMS the merged list.
Training: sum of the two detectors' 2D losses (each sees the same GT). custom_loss=True.
Robustness: cam_blind zeros the camera detector's image; lidar_blind zeros the lidar image.
Deploy: late needs TWO engines + a C++ merge/NMS (the guide notes this).
"""
import numpy as np
import torch
import torch.nn as nn

from common.backbones.image_backbone import ImageBackbone
from common.sensors.projection import render_lidar_3ch_torch
from common.geometry.boxes2d import iou2d, nms2d, decode_boxes2d
from train.losses_2d import compute_2d_loss
from fusion.base import FusionModel
from fusion.heads import CenterHead2D
from fusion.common_2d import build_target_2d


class LateFusion2D(FusionModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.stride = cfg["stride"]
        self.custom_loss = True
        pt = cfg.get("pretrained_backbone", False)
        self.cam_backbone = ImageBackbone(in_channels=3, out_channels=cfg["fusion_channels"], pretrained=pt)
        self.lid_backbone = ImageBackbone(in_channels=3, out_channels=cfg["fusion_channels"], pretrained=pt)
        self.cam_head = CenterHead2D(cfg["fusion_channels"], num_classes=cfg["num_classes"])
        self.lid_head = CenterHead2D(cfg["fusion_channels"], num_classes=cfg["num_classes"])

    def forward(self, batch):
        device = batch["image"].device
        img = torch.zeros_like(batch["image"]) if self._cam_blind else batch["image"]
        cam_pred = self.cam_head(self.cam_backbone(img)[str(self.stride)])
        B, _, H, W = batch["image"].shape
        if self._lidar_blind:
            lid_img = torch.zeros((B, 3, H, W), device=device, dtype=img.dtype)
        else:
            lid_frames = []
            for b in range(B):
                pts = batch["points"][b].to(device).float()
                lid_frames.append(render_lidar_3ch_torch(pts, batch["calib"][b], H, W, device, img.dtype))
            lid_img = torch.stack(lid_frames, dim=0)
        lid_pred = self.lid_head(self.lid_backbone(lid_img)[str(self.stride)])
        return {"cam": cam_pred, "lid": lid_pred}

    def loss(self, pred, target):
        return compute_2d_loss(pred["cam"], target)["loss"] + compute_2d_loss(pred["lid"], target)["loss"]

    def output_space(self): return "2d"
    def build_target(self, batch, cfg, device): return build_target_2d(batch, cfg, device)

    def decode(self, pred, cfg, k=40, thresh=0.1):
        cam = decode_boxes2d(pred["cam"], cfg["stride"], k, thresh)
        lid = decode_boxes2d(pred["lid"], cfg["stride"], k, thresh)
        return [self._merge(c, l, cfg) for c, l in zip(cam, lid)]

    def _merge(self, c, l, cfg):
        thresh = cfg.get("merge_iou_thresh", 0.5)
        mode = cfg.get("merge_conf_mode", "mean")
        cb, cs, ccl = c["boxes"], c["scores"], c["labels"]
        lb, ls, lcl = l["boxes"], l["scores"], l["labels"]
        if cb.shape[0] == 0: return l
        if lb.shape[0] == 0: return c
        iou = iou2d(cb.detach().cpu().numpy(), lb.detach().cpu().numpy())
        used = set()
        boxes, scores, labels = [], [], []
        for i in range(cb.shape[0]):
            jv = np.where(iou[i] > thresh)[0]
            matched = None
            for j in jv:
                if j not in used:
                    used.add(j); matched = j; break
            if matched is None:
                boxes.append(cb[i]); scores.append(cs[i]); labels.append(ccl[i])
            else:
                s = (cs[i] + ls[matched]) / 2 if mode == "mean" else torch.max(cs[i], ls[matched])
                boxes.append(cb[i]); scores.append(s); labels.append(ccl[i])
        for j in range(lb.shape[0]):
            if j not in used:
                boxes.append(lb[j]); scores.append(ls[j]); labels.append(lcl[j])
        boxes = torch.stack(boxes); scores = torch.stack(scores); labels = torch.stack(labels)
        keep = nms2d(boxes.detach().cpu().numpy(), scores.detach().cpu().numpy(), thresh)
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def export_onnx(self, path, cfg, device):
        B = 1; H, W = cfg["image_size"]; s = self.stride
        img = torch.zeros((B, 3, H, W), device=device)
        base = path[:-5] if path.endswith(".onnx") else path

        class _CamNet(nn.Module):
            def __init__(s_): super().__init__(); s_.backbone = self.cam_backbone; s_.head = self.cam_head
            def forward(s_, x):
                return s_.head(s_.backbone(x)[str(s)])

        class _LidNet(nn.Module):
            def __init__(s_): super().__init__(); s_.backbone = self.lid_backbone; s_.head = self.lid_head
            def forward(s_, x):
                return s_.head(s_.backbone(x)[str(s)])

        cam_net = _CamNet().eval()
        lid_net = _LidNet().eval()
        torch.onnx.export(cam_net, img, f"{base}_cam.onnx",
                          input_names=["image"], output_names=["heat", "off", "size"],
                          dynamic_axes={"image": {0: "B"}}, opset_version=17)
        torch.onnx.export(lid_net, img, f"{base}_lid.onnx",
                          input_names=["lidar_image"], output_names=["heat", "off", "size"],
                          dynamic_axes={"lidar_image": {0: "B"}}, opset_version=17)
        return f"{base}_cam.onnx (+{base}_lid.onnx)"