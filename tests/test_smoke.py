"""End-to-end smoke test on SYNTHETIC data (no KITTI needed). Builds every variant,
runs forward + target + loss + decode, and exports ONNX. Run: python -m pytest tests/test_smoke.py -q
"""
import os
import torch

from common.config import load_config
from data.loaders.synthetic_loader import SyntheticPairedDataset
from data.loaders.collate import collate_paired
from fusion.factory import build_model, variant_names
from train.losses_2d import compute_2d_loss
from train.losses_3d import compute_3d_loss


def _cfg(name):
    cfg = load_config(os.path.join("config", f"{name}.yaml"))
    cfg["dataset"] = "synthetic"
    cfg["image_size"] = [192, 640]  # small for fast smoke
    cfg["bev"] = {"range_m": 20.0, "res_m": 0.4}
    return cfg


def _batch(cfg, device):
    ds = SyntheticPairedDataset(cfg, "val", length=2)
    samples = [ds[i] for i in range(2)]
    batch = collate_paired(samples)
    batch["image"] = batch["image"].to(device)
    return batch


def _run_variant(name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _cfg(name)
    model = build_model(cfg).to(device)
    space = model.output_space()
    batch = _batch(cfg, device)
    pred = model(batch)
    target = model.build_target(batch, cfg, device)
    if getattr(model, "custom_loss", False):
        loss = model.loss(pred, target)
    elif space == "2d":
        loss = compute_2d_loss(pred, target)["loss"]
    else:
        loss = compute_3d_loss(pred, target)["loss"]
    assert torch.isfinite(loss), f"{name}: loss not finite"
    outs = model.decode(pred, cfg)
    assert len(outs) == batch["image"].shape[0]
    # robustness flags run without error
    model.set_cam_blind(True); model(batch); model.reset_blind()
    model.set_lidar_blind(True); model(batch); model.reset_blind()


def test_all_variants_forward():
    for name in variant_names():
        _run_variant(name)


def test_export_onnx(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in variant_names():
        cfg = _cfg(name)
        model = build_model(cfg).to(device).eval()
        path = str(tmp_path / f"{name}.onnx")
        out = model.export_onnx(path, cfg, device)
        main_path = out.split(" ")[0] if isinstance(out, str) else path
        assert os.path.exists(main_path), f"{name}: ONNX file not written ({main_path})"