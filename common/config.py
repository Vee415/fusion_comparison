"""Config loading + seed. Merge base.yaml with a variant delta."""
import os
import random
import numpy as np
import yaml


def load_config(variant_yaml: str, base_yaml: str = None) -> dict:
    """Load base config, then overlay the variant-specific yaml (deep-ish merge)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if base_yaml is None:
        base_yaml = os.path.join(root, "config", "base.yaml")
    cfg = {}
    if os.path.exists(base_yaml):
        with open(base_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    with open(variant_yaml) as f:
        variant = yaml.safe_load(f) or {}
    # shallow merge with dict-aware top level
    for k, v in variant.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    cfg.setdefault("variant_yaml", variant_yaml)
    cfg.setdefault("variant_name", os.path.splitext(os.path.basename(variant_yaml))[0])
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass