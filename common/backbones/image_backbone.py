"""Shared image backbone: ResNet-18 + FPN, returns multi-scale features.

forward(x) -> {stride: feat} dict with strides {4, 8, 16, 32}, each out_channels (256).
in_channels can be != 3 (early-2D uses 4: RGB + depth).
pretrained defaults False so it runs offline; set True in a real run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImageBackbone(nn.Module):
    def __init__(self, name="resnet18", out_channels=256, in_channels=3, pretrained=False):
        super().__init__()
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = torchvision.models.resnet18(weights=weights)
        except Exception:
            resnet = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                     kernel_size=old.kernel_size, stride=old.stride,
                                     padding=old.padding, bias=False)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        self.strides = [4, 8, 16, 32]
        self.lats_channels = [64, 128, 256, 512]
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in self.lats_channels])
        self.smooths = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1)
                                      for _ in self.lats_channels])
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1); c3 = self.layer3(c2); c4 = self.layer4(c3)
        feats = [c1, c2, c3, c4]
        lat = [l(f) for l, f in zip(self.laterals, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:],
                                                    mode="bilinear", align_corners=False)
        outs = [s(l) for s, l in zip(self.smooths, lat)]
        return {str(s): outs[i] for i, s in enumerate(self.strides)}