"""Synthetic KITTI-like loader: run the whole pipeline WITHOUT downloading 12 GB.

Generates a fake frame (image, point cloud, calib, 2D+3D boxes) so you can smoke-test
train/eval/export. NOT for real numbers — only for verifying code runs end-to-end.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from common.sensors.calibration import Calib
from data.loaders.collate import collate_paired


def _random_boxes2d(H, W, n):
    if n == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    cx = np.random.uniform(W * 0.2, W * 0.8, n)
    cy = np.random.uniform(H * 0.3, H * 0.8, n)
    w = np.random.uniform(30, 90, n); h = np.random.uniform(25, 70, n)
    x1 = (cx - w / 2).clip(0, W - 1); y1 = (cy - h / 2).clip(0, H - 1)
    x2 = (cx + w / 2).clip(1, W); y2 = (cy + h / 2).clip(1, H)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    cls = np.zeros((n,), np.int64)  # Car = 0
    return boxes, cls


def _box2d_to_box3d(b, calib, depth):
    """Place a 3D box at the center bottom of a 2D box at a given depth (cam frame)."""
    cx_px = (b[0] + b[2]) / 2; cy_px = b[3]  # bottom center
    fx, cx, cy = calib.P2[0, 0], calib.P2[0, 2], calib.P2[1, 2]
    x = (cx_px - cx) * depth / fx
    z = depth
    y = (cy_px - cy) * depth / fx  # cam y (down)
    w3 = 1.8; h3 = 1.5; l3 = 4.0
    yaw = float(np.random.uniform(-np.pi, np.pi))
    return np.array([x, y - h3 / 2, z, w3, h3, l3, yaw], np.float32)


class SyntheticPairedDataset(Dataset):
    def __init__(self, cfg, split="train", length=64):
        self.H, self.W = cfg["image_size"]
        self.max_points = cfg.get("lidar", {}).get("max_points", 15000)
        self.length = length
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np.random.seed(idx if self.split == "train" else 10000 + idx)
        calib = Calib.synthetic(self.H, self.W)
        img = np.random.rand(self.H, self.W, 3).astype(np.float32) * 0.5 + 0.1
        # random points in front of the camera (cam frame x right, y down, z forward)
        n = np.random.randint(2000, 5000)
        pts = np.zeros((n, 4), np.float32)
        pts[:, 0] = np.random.uniform(-15, 15, n)          # x (right)
        pts[:, 1] = np.random.uniform(-2, 2, n)           # y (down)
        pts[:, 2] = np.random.uniform(2, 40, n)           # z (forward)
        pts[:, 3] = np.random.rand(n)                     # intensity
        if pts.shape[0] > self.max_points:
            sel = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[sel]
        # 0-3 random "cars"
        nb = np.random.randint(0, 3)
        boxes2d, cls = _random_boxes2d(self.H, self.W, nb)
        boxes3d = []
        for b in boxes2d:
            d = np.random.uniform(8, 30)
            boxes3d.append(_box2d_to_box3d(b, calib, d))
        boxes3d = np.stack(boxes3d, axis=0).astype(np.float32) if boxes3d else np.zeros((0, 7), np.float32)
        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).contiguous(),   # (3,H,W) 0-1
            "points": torch.from_numpy(pts),                               # (N,4) cam frame
            "calib": calib,
            "image_id": f"synthetic_{idx:04d}",
            "boxes2d": torch.from_numpy(boxes2d),
            "boxes2d_cls": torch.from_numpy(cls),
            "boxes3d": torch.from_numpy(boxes3d),
        }