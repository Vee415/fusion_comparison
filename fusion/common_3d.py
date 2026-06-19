"""Shared 3D target encoding + decoding for the 3D variant."""
import torch

from common.geometry.boxes3d import encode_boxes3d, decode_boxes3d
from common.geometry.bev import grid_shape


def build_target_3d(batch, cfg, device):
    bev = cfg.get("bev", cfg)
    range_m, res = float(bev["range_m"]), float(bev["res_m"])
    Hg, Wg = grid_shape(range_m, res)
    C = cfg["num_classes"]
    heats, offs, heights, sizes, yaws = [], [], [], [], []
    for boxes, cls in zip(batch["boxes3d"], batch["boxes2d_cls"]):
        boxes = boxes.to(device).float(); cls = cls.to(device).long()
        t = encode_boxes3d(boxes, cls, Hg, Wg, res, range_m, C, device)
        heats.append(t["heat"]); offs.append(t["off"]); heights.append(t["height"])
        sizes.append(t["size"]); yaws.append(t["yaw"])
    return {"heat": torch.stack(heats), "off": torch.stack(offs), "height": torch.stack(heights),
            "size": torch.stack(sizes), "yaw": torch.stack(yaws)}


def decode_3d(pred, cfg, k=40, thresh=0.1):
    bev = cfg.get("bev", cfg)
    range_m, res = float(bev["range_m"]), float(bev["res_m"])
    return decode_boxes3d(pred, res, range_m, k=k, thresh=thresh)