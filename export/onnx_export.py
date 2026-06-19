"""Export each variant to ONNX + shape check.

Each variant implements export_onnx(path, cfg, device) with a clean input signature:
  early_2d : 4-channel image (depth rendering is preprocess)
  mid_2d   : image + pre-scattered lidar feature map
  late_2d  : two backbones (cam + lid) -> two ONNX files
  fusion_3d: pre-built BEV_cam + BEV_lid (lift-splat/pillarize are CUDA preprocess at deploy)

This split is exactly why the 2D variants export cleanly and the 3D C++ deploy is v2
(documented, not built now) -- see guide section 8/9.
"""
import os
import argparse
import torch

from common.config import load_config
from fusion.factory import build_model, variant_names


def export_variant(name, base_cfg, out_dir="onnx"):
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", f"{name}.yaml")
    cfg = load_config(cfg_path)
    if base_cfg.get("dataset"): cfg["dataset"] = base_cfg["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    ckpt = os.path.join("checkpoints", f"{name}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.onnx")
    out = model.export_onnx(path, cfg, device)
    print(f"[{name}] exported -> {out}")
    # shape check with onnxruntime if available
    try:
        import onnx
        onnx.checker.check_model(onnx.load(path if path.endswith(".onnx") else out.split(" ")[0]))
        print(f"[{name}] onnx.checker OK")
    except Exception as e:
        print(f"[{name}] onnx check skipped/failed: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="single variant yaml; else export all")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()
    base = {"dataset": args.dataset} if args.dataset else {}
    if args.config:
        name = os.path.splitext(os.path.basename(args.config))[0]
        export_variant(name, base)
    else:
        for name in (args.variants or variant_names()):
            try:
                export_variant(name, base)
            except Exception as e:
                print(f"[{name}] export failed: {e}")


if __name__ == "__main__":
    main()