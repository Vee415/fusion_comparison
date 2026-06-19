"""Projection helpers (numpy for loaders/notebooks, torch for models).

All functions use a Calib object. Convention: cam frame x right, y down, z forward.
"""
import numpy as np


# ---------------- numpy (loaders, notebook, tests) ----------------

def lidar_to_image(points: np.ndarray, calib, H: int, W: int):
    """(N,>=3) velo points -> uv (N,2), depth (N,), valid (N,) bool (in-front + in-frame)."""
    uv, depth, _ = calib.velo_to_image(points[:, :3])
    valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, depth, valid


def render_depth_image(points: np.ndarray, calib, H: int, W: int, fill: float = 0.0):
    """Project LiDAR onto an (H,W) depth image (nearest by min depth). For baselines/early."""
    uv, depth, valid = lidar_to_image(points, calib, H, W)
    img = np.full((H, W), fill, dtype=np.float32)
    uv, depth = uv[valid], depth[valid]
    if uv.shape[0] == 0:
        return img
    # nearest via floor; if multiple points per pixel keep min depth (closest)
    u = uv[:, 0].astype(np.int64)
    v = uv[:, 1].astype(np.int64)
    order = np.argsort(-depth)  # far first, then near overwrites
    u, v, depth = u[order], v[order], depth[order]
    img[v, u] = depth.astype(np.float32)
    return img


def render_lidar_3ch_image(points: np.ndarray, calib, H: int, W: int):
    """(H,W,3) = [depth, height, intensity] for the LiDAR-only-2D baseline."""
    uv, depth, valid = lidar_to_image(points, calib, H, W)
    img = np.zeros((H, W, 3), dtype=np.float32)
    uv, depth = uv[valid], depth[valid]
    if uv.shape[0] == 0:
        return img
    u = uv[:, 0].astype(np.int64)
    v = uv[:, 1].astype(np.int64)
    hgt = points[valid, 2].astype(np.float32)  # z (height axis in velo) -- proxy
    inten = points[valid, 3].astype(np.float32) if points.shape[1] >= 4 else np.zeros_like(hgt)
    order = np.argsort(-depth)
    u, v, depth, hgt, inten = u[order], v[order], depth[order], hgt[order], inten[order]
    img[v, u, 0] = depth
    img[v, u, 1] = hgt
    img[v, u, 2] = inten
    return img


# ---------------- torch (models) ----------------

def _dev_type(device):
    s = str(device)
    return "cuda" if s.startswith("cuda") else "cpu"


def lidar_to_image_torch(points, calib, H: int, W: int, device, dtype=None):
    """points: (N,>=3) torch -> uv (N,2), depth (N,), valid (N,). One frame (no batch).

    Runs in fp32 with autocast disabled: this is fixed geometry (preprocess), and AMP would
    otherwise promote the matmuls to fp16 and break downstream dtype assumptions.
    """
    import torch
    with torch.amp.autocast(_dev_type(device), enabled=False):
        points = points.float()
        m = calib.torch_matrices(device, torch.float32)
        n = points.shape[0]
        h = torch.cat([points[:, :3], torch.ones((n, 1), dtype=torch.float32, device=device)], dim=1)
        cam = (m["R0"] @ m["V2C"] @ h.t()).t()[:, :3]
        img = (m["P"] @ torch.cat([cam, torch.ones((n, 1), dtype=torch.float32, device=device)], dim=1).t()).t()[:, :3]
        depth = img[:, 2]
        uv = img[:, :2] / torch.where(depth == 0, torch.full_like(depth, 1e-9), depth).unsqueeze(1)
        valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, depth, valid


def render_depth_torch(points, calib, H: int, W: int, device, dtype=None, fill: float = 0.0):
    """(N,>=3) velo points -> (H,W) depth image (nearest = min depth, closest wins)."""
    import torch
    dtype = dtype or points.dtype
    uv, depth, valid = lidar_to_image_torch(points, calib, H, W, device, dtype)
    img = torch.full((H, W), fill, dtype=dtype, device=device)
    if valid.sum() == 0:
        return img
    u = uv[valid, 0].long().clamp(0, W - 1)
    v = uv[valid, 1].long().clamp(0, H - 1)
    d = depth[valid]
    order = torch.argsort(-d)  # far first so near overwrites
    u, v, d = u[order], v[order], d[order]
    img[v, u] = d
    return img


def render_lidar_3ch_torch(points, calib, H: int, W: int, device, dtype=None):
    """(N,>=4) velo points -> (3,H,W) [depth, height(z), intensity] for the lidar-2D baseline."""
    import torch
    dtype = dtype or points.dtype
    uv, depth, valid = lidar_to_image_torch(points, calib, H, W, device, dtype)
    img = torch.zeros((3, H, W), dtype=dtype, device=device)
    if valid.sum() == 0:
        return img
    u = uv[valid, 0].long().clamp(0, W - 1)
    v = uv[valid, 1].long().clamp(0, H - 1)
    d = depth[valid]
    hgt = points[valid, 2] if points.shape[1] >= 3 else torch.zeros_like(d)
    inten = points[valid, 3] if points.shape[1] >= 4 else torch.zeros_like(d)
    order = torch.argsort(-d)
    u, v, d, hgt, inten = u[order], v[order], d[order], hgt[order], inten[order]
    img[0, v, u] = d; img[1, v, u] = hgt; img[2, v, u] = inten
    return img


def points_to_image_grid(points, calib, H: int, W: int, stride: int, device, dtype=None):
    """Project points to feature-grid cells (fu,fv) in [0,W/stride) x [0,H/stride).

    Returns cell coords (N,2) long, valid (N,) bool, depth (N,), per-point intensity (N,).
    Used by the mid-2D lidar image-grid encoder.
    """
    import torch
    uv, depth, valid = lidar_to_image_torch(points, calib, H, W, device, dtype)
    cell = (uv / float(stride)).long()
    gw, gh = W // stride, H // stride
    cell_valid = valid & (cell[:, 0] >= 0) & (cell[:, 0] < gw) & (cell[:, 1] >= 0) & (cell[:, 1] < gh)
    inten = points[:, 3] if points.shape[1] >= 4 else torch.zeros((points.shape[0],), dtype=points.dtype, device=device)
    return cell, cell_valid, depth, inten