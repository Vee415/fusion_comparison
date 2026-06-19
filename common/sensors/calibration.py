"""KITTI calibration loading + projection math.

Coordinate convention: KITTI camera frame -> x right, y down, z forward.
Pipeline:
    velo (LiDAR)  --Tr_velo_to_cam-->  cam (unrect)  --R0_rect-->  cam (rect)
    cam (rect)    --P2-->  image (u,v) + depth
We store 4x4 homogeneous matrices internally so composition is just matmul.
"""
import numpy as np


class Calib:
    """Holds P2, R0_rect (3x3), Tr_velo_to_cam (3x4) as homogeneous 4x4 matrices."""

    def __init__(self, P2, R0_rect, Tr_velo_to_cam):
        self.P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
        self.R0 = np.eye(4)
        self.R0[:3, :3] = np.asarray(R0_rect, dtype=np.float64).reshape(3, 3)
        self.V2C = np.eye(4)
        self.V2C[:3, :] = np.asarray(Tr_velo_to_cam, dtype=np.float64).reshape(3, 4)
        self.P = np.eye(4)
        self.P[:3, :] = self.P2
        self._torch_cache = {}

    # ---- constructors ----
    @classmethod
    def from_file(cls, path: str) -> "Calib":
        """Parse a KITTI calib/*.txt file."""
        d = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, _, vals = line.partition(":")
                d[key.strip()] = np.array([float(x) for x in vals.split()])
        return cls(
            P2=d.get("P2", np.eye(3, 4).flatten()),
            R0_rect=d.get("R0_rect", np.eye(3).flatten()),
            Tr_velo_to_cam=d.get("Tr_velo_to_cam", np.eye(3, 4).flatten()),
        )

    @classmethod
    def from_arrays(cls, P2=None, R0_rect=None, Tr_velo_to_cam=None):
        """Build a synthetic/identity-ish calib (used by the synthetic loader)."""
        P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4) if P2 is not None else np.eye(3, 4)
        R0_rect = np.asarray(R0_rect, dtype=np.float64).reshape(3, 3) if R0_rect is not None else np.eye(3)
        Tr = np.asarray(Tr_velo_to_cam, dtype=np.float64).reshape(3, 4) if Tr_velo_to_cam is not None else np.eye(3, 4)
        return cls(P2, R0_rect, Tr)

    @classmethod
    def synthetic(cls, H: int, W: int, fx: float = 700.0, fy: float = 700.0):
        """A plausible camera (identity extrinsics) for smoke testing without KITTI."""
        cx, cy = W / 2.0, H / 2.0
        P2 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        return cls.from_arrays(P2=P2)

    # ---- numpy projection ----
    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) LiDAR points -> (N,3) rectified camera coords."""
        pts = np.asarray(pts, dtype=np.float64)
        n = pts.shape[0]
        h = np.hstack([pts[:, :3], np.ones((n, 1))])
        cam = (self.R0 @ self.V2C @ h.T).T
        return cam[:, :3]

    def cam_to_image(self, pts_cam: np.ndarray):
        """(N,3) cam -> (N,2) pixels, (N,) depth."""
        pts_cam = np.asarray(pts_cam, dtype=np.float64)
        n = pts_cam.shape[0]
        h = np.hstack([pts_cam, np.ones((n, 1))])
        img = (self.P @ h.T).T[:, :3]
        depth = img[:, 2]
        uv = img[:, :2] / np.where(depth == 0, 1e-9, depth)[:, None]
        return uv, depth

    def velo_to_image(self, pts: np.ndarray):
        """(N,3) velo -> uv (N,2), depth (N,), cam (N,3)."""
        cam = self.velo_to_cam(pts)
        uv, depth = self.cam_to_image(cam)
        return uv, depth, cam

    def image_to_cam(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project pixels + depth -> (N,3) rectified cam coords."""
        uv = np.asarray(uv, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        fx, fy = self.P2[0, 0], self.P2[1, 1]
        cx, cy = self.P2[0, 2], self.P2[1, 2]
        x = (uv[:, 0] - cx) * depth / fx
        y = (uv[:, 1] - cy) * depth / fy
        return np.stack([x, y, depth], axis=1)

    def cam_to_velo(self, pts_cam: np.ndarray) -> np.ndarray:
        """(N,3) rect cam -> (N,3) LiDAR velo coords (inverse of velo_to_cam)."""
        pts_cam = np.asarray(pts_cam, dtype=np.float64)
        n = pts_cam.shape[0]
        h = np.hstack([pts_cam, np.ones((n, 1))])
        velo = (np.linalg.inv(self.R0 @ self.V2C) @ h.T).T[:, :3]
        return velo

    # ---- torch helpers (lazy, per-device cache) ----
    def torch_matrices(self, device, dtype=None):
        import torch
        dtype = dtype or torch.float32
        key = (str(device), str(dtype))
        if key not in self._torch_cache:
            self._torch_cache[key] = {
                "P": torch.tensor(self.P, dtype=dtype, device=device),
                "R0": torch.tensor(self.R0, dtype=dtype, device=device),
                "V2C": torch.tensor(self.V2C, dtype=dtype, device=device),
            }
        else:
            for v in self._torch_cache[key].values():
                v = v.to(device)
        return self._torch_cache[key]