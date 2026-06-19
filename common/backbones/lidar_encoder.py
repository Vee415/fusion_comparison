"""LiDAR encoders, both based on the same PointNet atom (per-point MLP + pool).

1. LidarImageGridEncoder: points -> project to image grid (stride s) -> per-point MLP
   -> max-pool per cell -> (B, C, Hg, Wg) aligned to the camera feature grid.
   Used by mid-2D (F_lid).

2. PillarBEVEncoder: points -> assign to BEV pillars -> per-point MLP (with cell-relative
   features) -> max-pool per BEV cell -> (B, C, Hg, Wg) top-down pseudo-image.
   Used by fusion-3D (BEV_lid). This is a simplified PointPillars (max-pool variant,
   not fixed-P padded pillars) -- lighter, differentiable-friendly, laptop-friendly.
"""
import torch
import torch.nn as nn

from common.sensors.projection import points_to_image_grid, lidar_to_image_torch
from common.geometry.bev import points_to_bev_cells_torch, grid_shape


class PointMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LidarImageGridEncoder(nn.Module):
    """points (velo) -> image-grid feature map (B, C, Hg, Wg) at stride s."""

    def __init__(self, out_channels=32, stride=16):
        super().__init__()
        self.stride = stride
        self.mlp = PointMLP(in_dim=4, out_dim=out_channels)  # (x,y,z,intensity) cam
        self.out_channels = out_channels

    def forward(self, points_list, calib_list, H, W, device):
        s = self.stride
        Hg, Wg = H // s, W // s
        B = len(points_list)
        out = torch.zeros(B, self.out_channels, Hg, Wg, device=device)
        for b in range(B):
            pts = points_list[b].to(device).float()
            if pts.shape[0] == 0:
                continue
            m = calib_list[b].torch_matrices(device, pts.dtype)
            n = pts.shape[0]
            h = torch.cat([pts[:, :3], torch.ones((n, 1), device=device, dtype=pts.dtype)], dim=1)
            cam = (m["R0"] @ m["V2C"] @ h.t()).t()[:, :3]          # (N,3) cam
            inten = pts[:, 3] if pts.shape[1] >= 4 else torch.zeros(n, device=device, dtype=pts.dtype)
            cell, valid, depth, _ = points_to_image_grid(pts, calib_list[b], H, W, s, device, pts.dtype)
            valid = valid & (depth > 0)
            if valid.sum() == 0:
                continue
            feat_in = torch.cat([cam[valid], inten[valid].unsqueeze(-1)], dim=-1)  # (M,4)
            feats = self.mlp(feat_in).float()                          # (M,C) fp32 for stable scatter under AMP
            r = cell[valid, 1].long().clamp(0, Hg - 1)                  # v/stride -> row
            c = cell[valid, 0].long().clamp(0, Wg - 1)                 # u/stride -> col
            lin = r * Wg + c
            grid = torch.zeros(self.out_channels, Hg * Wg, device=device)
            grid.scatter_reduce_(1, lin.unsqueeze(0).expand(self.out_channels, -1),
                                  feats.t(), reduce="amax", include_self=True)
            out[b] = grid.view(self.out_channels, Hg, Wg)
        return out


class PillarBEVEncoder(nn.Module):
    """points (velo) -> BEV pseudo-image (B, C, Hg, Wg). Simplified PointPillars."""

    def __init__(self, out_channels=64, in_dim=6):
        super().__init__()
        self.mlp = PointMLP(in_dim=in_dim, out_dim=out_channels)  # x,y,z,intensity,dx,dz
        self.out_channels = out_channels
        self.in_dim = in_dim

    def forward(self, points_list, calib_list, cfg, device):
        bev = cfg.get("bev", cfg) if isinstance(cfg, dict) else cfg
        range_m = float(bev["range_m"]); res = float(bev["res_m"])
        Hg, Wg = grid_shape(range_m, res)
        B = len(points_list)
        out = torch.zeros(B, self.out_channels, Hg, Wg, device=device)
        for b in range(B):
            pts = points_list[b].to(device).float()
            if pts.shape[0] == 0:
                continue
            m = calib_list[b].torch_matrices(device, pts.dtype)
            n = pts.shape[0]
            h = torch.cat([pts[:, :3], torch.ones((n, 1), device=device, dtype=pts.dtype)], dim=1)
            cam = (m["R0"] @ m["V2C"] @ h.t()).t()[:, :3]          # (N,3) cam frame
            inten = pts[:, 3] if pts.shape[1] >= 4 else torch.zeros(n, device=device, dtype=pts.dtype)
            r, c, valid, _ = points_to_bev_cells_torch(cam, cfg, device, pts.dtype)
            if valid.sum() == 0:
                continue
            x = cam[valid, 0]; y = cam[valid, 1]; z = cam[valid, 2]; i = inten[valid]
            rr = r[valid].long().clamp(0, Hg - 1); cc = c[valid].long().clamp(0, Wg - 1)
            cell_x = -range_m + (cc.float() + 0.5) * res
            cell_z = -range_m + (rr.float() + 0.5) * res
            dx = x - cell_x; dz = z - cell_z
            feat_in = torch.stack([x, y, z, i, dx, dz], dim=-1)
            feats = self.mlp(feat_in).float()                          # fp32 for stable scatter under AMP
            lin = rr * Wg + cc
            grid = torch.zeros(self.out_channels, Hg * Wg, device=device)
            grid.scatter_reduce_(1, lin.unsqueeze(0).expand(self.out_channels, -1),
                                  feats.t(), reduce="amax", include_self=True)
            out[b] = grid.view(self.out_channels, Hg, Wg)
        return out