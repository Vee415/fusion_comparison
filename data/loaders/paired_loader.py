"""KITTI paired loader: yields (image, points[velo], calib, boxes2d, boxes3d).

Layout expected under data_root:
    image_2/<id>.png  velodyne/<id>.bin  label_2/<id>.txt  calib/<id>.txt
Points are returned in the LiDAR (velo) frame; models project to cam via calib.
3D boxes are returned in the cam frame as (x, y_center, z, w, h, l, yaw) where
yaw ~ KITTI ry. NOTE: real 3D-AP must use the KITTI devkit (see eval/metrics_3d.py);
this internal convention is for training targets only.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from common.sensors.calibration import Calib
from data.loaders.collate import collate_paired


def _read_label(path, classes):
    if not os.path.exists(path):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), np.zeros((0, 7), np.float32)
    b2, cls, b3 = [], [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if not p or p[0] not in classes:
                continue
            cls.append(classes.index(p[0]))
            b2.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])  # x1 y1 x2 y2
            h, w, l = float(p[8]), float(p[9]), float(p[10])
            x, y, z = float(p[11]), float(p[12]), float(p[13])
            ry = float(p[14])
            b3.append([x, y - h / 2, z, w, h, l, ry])  # y_center = bottom - h/2
    if not b2:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), np.zeros((0, 7), np.float32)
    return (np.asarray(b2, np.float32), np.asarray(cls, np.int64),
            np.asarray(b3, np.float32))


class KittiPairedDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.root = os.path.join(cfg["data_root"], split) if os.path.isdir(
            os.path.join(cfg["data_root"], split)) else cfg["data_root"]
        self.H, self.W = cfg["image_size"]
        self.max_points = cfg.get("lidar", {}).get("max_points", 15000)
        self.classes = cfg["classes"]
        self.split = split
        img_dir = os.path.join(self.root, "image_2")
        if not os.path.isdir(img_dir):
            img_dir = self.root
        self.ids = sorted(
            os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))
        ) if os.path.isdir(img_dir) else []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        calib = Calib.from_file(os.path.join(self.root, "calib", f"{i}.txt"))
        # image
        import cv2
        img = cv2.imread(os.path.join(self.root, "image_2", f"{i}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.W, self.H))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        # points (velo frame)
        pts = np.fromfile(os.path.join(self.root, "velodyne", f"{i}.bin"), dtype=np.float32).reshape(-1, 4)
        if pts.shape[0] > self.max_points:
            sel = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[sel]
        pts = torch.from_numpy(pts.astype(np.float32))
        # labels
        b2, cls, b3 = _read_label(os.path.join(self.root, "label_2", f"{i}.txt"), self.classes)
        return {
            "image": img, "points": pts, "calib": calib, "image_id": i,
            "boxes2d": torch.from_numpy(b2), "boxes2d_cls": torch.from_numpy(cls),
            "boxes3d": torch.from_numpy(b3),
        }