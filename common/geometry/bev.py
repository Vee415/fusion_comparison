"""BEV geometry: point/pixel -> BEV cell mappings.

Convention (asserted): cam frame x right, y down, z forward.
BEV grid: dim0 (r) = z (forward), dim1 (c) = x (right), centered at ego.
    z = -range + (r+0.5)*res,  x = -range + (c+0.5)*res
"""
import numpy as np
import torch


def grid_shape(range_m: float, res_m: float) -> tuple:
    g = int(2 * range_m / res_m)
    return g, g


def _cfg(cfg):
    bev = cfg.get("bev", cfg) if isinstance(cfg, dict) else cfg
    return float(bev["range_m"]), float(bev["res_m"])


# ---------------- points -> BEV cells (LiDAR) ----------------

def points_to_bev_cells_torch(points, cfg, device, dtype=None):
    """(N,>=3) velo/cam points (assumed already in cam frame for 3D branch) -> r,c,valid.
    NOTE: expects cam-frame points (x right, y down, z forward). Convert velo->cam first.
    """
    range_m, res = _cfg(cfg)
    Hg, Wg = grid_shape(range_m, res)
    dtype = dtype or points.dtype
    x = points[:, 0]; z = points[:, 2]
    r = ((z + range_m) / res).long()
    c = ((x + range_m) / res).long()
    valid = (r >= 0) & (r < Hg) & (c >= 0) & (c < Wg)
    return r, c, valid, (Hg, Wg)


def pixel_depth_to_bev_torch(uv, depth, calib, cfg, device, dtype=None):
    """Camera->BEV mapping for lift-splat. Given pixels (N,2) + depths (N,),
    back-project to cam-frame 3D points and map (x,z) -> BEV cells.
    Returns r, c, valid, cam_pts (N,3), (Hg,Wg).
    """
    import torch
    range_m, res = _cfg(cfg)
    Hg, Wg = grid_shape(range_m, res)
    dtype = dtype or uv.dtype
    m = calib.torch_matrices(device, dtype)
    fx, fy, cx, cy = m["P"][0, 0], m["P"][1, 1], m["P"][0, 2], m["P"][1, 2]
    x = (uv[:, 0] - cx) * depth / fx
    y = (uv[:, 1] - cy) * depth / fy
    cam = torch.stack([x, y, depth], dim=-1)
    r = ((cam[:, 2] + range_m) / res).long()
    c = ((cam[:, 0] + range_m) / res).long()
    valid = (r >= 0) & (r < Hg) & (c >= 0) & (c < Wg)
    return r, c, valid, cam, (Hg, Wg)


# ---------------- numpy viz (notebook / milestone 1) ----------------

def points_to_bev_image(points, range_m=32.0, res=0.2, max_h=3.0):
    """(N,>=3) cam-frame points -> (Hg,Wg,3) BEV image [height, intensity, density].
    Top-down map; cars show up as blobs. For the milestone-1 visual check.
    """
    Hg, Wg = grid_shape(range_m, res)
    img = np.zeros((Hg, Wg, 3), dtype=np.float32)
    if points.shape[0] == 0:
        return img
    x = points[:, 0]; z = points[:, 2]; y = points[:, 1]
    inten = points[:, 3] if points.shape[1] >= 4 else np.zeros_like(x)
    r = ((z + range_m) / res).astype(np.int64)
    c = ((x + range_m) / res).astype(np.int64)
    m = (r >= 0) & (r < Hg) & (c >= 0) & (c < Wg)
    r, c, y, inten = r[m], c[m], y[m], inten[m]
    # height normalized (y down -> height above ground ~ -y)
    hgt = np.clip(-y / max(max_h, 1e-6), 0, 1)
    order = np.argsort(-hgt)  # tall on top
    r, c, hgt, inten = r[order], c[order], hgt[order], inten[order]
    img[r, c, 0] = hgt.astype(np.float32)
    img[r, c, 1] = inten.astype(np.float32)
    img[r, c, 2] = 1.0  # density (any hit)
    return img


def camera_to_bev_image(calib, depth_img, cfg=None, range_m=32.0, res=0.2):
    """Lift a (H,W) depth image back to a (Hg,Wg) BEV occupancy map. Simplified LSS viz."""
    Hg, Wg = grid_shape(range_m, res)
    bev = np.zeros((Hg, Wg), dtype=np.float32)
    H, W = depth_img.shape
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    us = us.reshape(-1).astype(np.float64)
    vs = vs.reshape(-1).astype(np.float64)
    d = depth_img.reshape(-1).astype(np.float64)
    m = d > 0
    if m.sum() == 0:
        return bev
    fx, fy, cx, cy = calib.P2[0, 0], calib.P2[1, 1], calib.P2[0, 2], calib.P2[1, 2]
    x = (us[m] - cx) * d[m] / fx
    z = (vs[m] - cy) * d[m] / fy  # note: vs is image row -> not depth axis; use d as forward
    # In cam frame forward is z; here we used depth as the forward distance directly.
    z = d[m]
    r = ((z + range_m) / res).astype(np.int64)
    c = ((x + range_m) / res).astype(np.int64)
    mm = (r >= 0) & (r < Hg) & (c >= 0) & (c < Wg)
    bev[r[mm], c[mm]] = 1.0
    return bev