"""Shared 2D target encoding + decoding for all 2D variants."""
import torch

from common.geometry.boxes2d import encode_boxes2d, decode_boxes2d


def build_target_2d(batch, cfg, device):
    stride = cfg["stride"]; H, W = cfg["image_size"]
    Hg, Wg = H // stride, W // stride
    C = cfg["num_classes"]
    heats, offs, sizes = [], [], []
    for boxes, cls in zip(batch["boxes2d"], batch["boxes2d_cls"]):
        boxes = boxes.to(device).float(); cls = cls.to(device).long()
        t = encode_boxes2d(boxes, cls, Hg, Wg, stride, C, device)
        heats.append(t["heat"]); offs.append(t["off"]); sizes.append(t["size"])
    return {"heat": torch.stack(heats), "off": torch.stack(offs), "size": torch.stack(sizes)}


def decode_2d(pred, cfg, k=40, thresh=0.1):
    return decode_boxes2d(pred, cfg["stride"], k=k, thresh=thresh)