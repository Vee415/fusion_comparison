"""Geometry tests. Run: python -m pytest tests/test_projection.py -q

These run WITHOUT KITTI (synthetic identity calib). The KITTI overlay check is skipped
if no data is present.
"""
import os
import numpy as np
import pytest

from common.sensors.calibration import Calib
from common.sensors.projection import lidar_to_image, render_depth_image
from common.geometry.bev import points_to_bev_image, grid_shape


def test_identity_roundtrip():
    """With identity extrinsics (synthetic calib), velo==cam and pixel->point->pixel round-trips."""
    calib = Calib.synthetic(384, 1280)
    pts = np.array([[1.0, -1.0, 10.0, 0.5], [2.0, 0.0, 20.0, 0.3]], dtype=np.float64)
    uv, depth, valid = lidar_to_image(pts, calib, 384, 1280)
    assert valid.all()
    cam = calib.image_to_cam(uv, depth)
    assert np.allclose(cam, pts[:, :3], atol=1e-6), "round-trip failed"


def test_bev_image_shape():
    pts = np.random.rand(1000, 4) * 40 - 20
    img = points_to_bev_image(pts, range_m=32.0, res=0.2)
    assert img.shape == (320, 320, 3)
    assert img.sum() > 0  # something landed


def test_grid_shape():
    assert grid_shape(32.0, 0.2) == (320, 320)
    assert grid_shape(32.0, 0.4) == (160, 160)


def test_depth_image():
    calib = Calib.synthetic(384, 1280)
    pts = np.array([[1.0, -1.0, 10.0, 0.5], [2.0, 0.0, 20.0, 0.3]], dtype=np.float64)
    dimg = render_depth_image(pts, calib, 384, 1280)
    assert dimg.shape == (384, 1280)
    assert dimg.max() > 0


@pytest.mark.skipif(not os.path.isdir("data/kitti/image_2"), reason="KITTI not downloaded")
def test_kitti_overlay_smoke():
    """If KITTI is present, load one frame and assert some points land in-frame."""
    from data.loaders.paired_loader import KittiPairedDataset
    cfg = {"data_root": "data/kitti", "image_size": [384, 1280],
           "lidar": {"max_points": 15000}, "classes": ["Car"]}
    ds = KittiPairedDataset(cfg, split="train")
    if len(ds) == 0:
        pytest.skip("no frames")
    s = ds[0]
    uv, depth, valid = lidar_to_image(s["points"].numpy(), s["calib"], 384, 1280)
    assert valid.sum() > 0, "no LiDAR points projected into the image -- check calibration"