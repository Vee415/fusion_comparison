"""Build a variant model from config. Single source of truth for variant -> class."""
from fusion.early_2d.model import EarlyFusion2D
from fusion.mid_2d.model import MidFusion2D
from fusion.late_2d.model import LateFusion2D
from fusion.fusion_3d.model import Fusion3D

_REGISTRY = {
    "early_2d": EarlyFusion2D,
    "mid_2d": MidFusion2D,
    "late_2d": LateFusion2D,
    "fusion_3d": Fusion3D,
}


def build_model(cfg):
    name = cfg.get("variant") or cfg.get("variant_name")
    if name not in _REGISTRY:
        raise ValueError(f"unknown variant '{name}'. known: {list(_REGISTRY)}")
    return _REGISTRY[name](cfg)


def variant_names():
    return list(_REGISTRY)