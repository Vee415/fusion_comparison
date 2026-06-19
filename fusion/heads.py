"""Detection heads used by all variants (shared so variants differ only in fusion).

CenterHead2D: features (B,C,h,w) -> heat (B,1), off (B,2), size (B,2) [image px]
CenterHead3D: BEV features (B,C,Hg,Wg) -> heat (B,1), off (B,2), height (B,1),
              size (B,3) [w,h,l], yaw (B,2) [sin,cos]
"""
import torch
import torch.nn as nn


class CenterHead2D(nn.Module):
    def __init__(self, in_channels, num_classes=1, hidden=128):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(hidden, num_classes, 1)
        self.off = nn.Conv2d(hidden, 2, 1)
        self.size = nn.Conv2d(hidden, 2, 1)
        # init heat bias so initial sigmoid ~ small
        self.heat.bias.data.fill_(-2.19)  # sigmoid(-2.19) ~ 0.1

    def forward(self, feat):
        h = self.head(feat)
        return {"heat": self.heat(h), "off": self.off(h), "size": self.size(h)}


class CenterHead3D(nn.Module):
    def __init__(self, in_channels, num_classes=1, hidden=128, predict_velocity=False):
        super().__init__()
        self.num_classes = num_classes
        self.predict_velocity = predict_velocity
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(hidden, num_classes, 1)
        self.off = nn.Conv2d(hidden, 2, 1)
        self.height = nn.Conv2d(hidden, 1, 1)
        self.size = nn.Conv2d(hidden, 3, 1)
        self.yaw = nn.Conv2d(hidden, 2, 1)
        self.heat.bias.data.fill_(-2.19)
        if predict_velocity:
            self.vel = nn.Conv2d(hidden, 2, 1)

    def forward(self, feat):
        h = self.head(feat)
        out = {"heat": self.heat(h), "off": self.off(h), "height": self.height(h),
               "size": self.size(h), "yaw": self.yaw(h)}
        if self.predict_velocity:
            out["vel"] = self.vel(h)
        return out