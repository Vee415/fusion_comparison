"""The one benchmark table: AP + p50/p95 latency + cam-dropped + lidar-dropped AP,
across all variants (and baselines if you build them). Numbers + the robustness row =
the artifact. Run: python -m eval.benchmark [--dataset synthetic] [--iters N]
"""
import os
import time
import argparse
import csv

import numpy as np
import torch

from common.config import load_config
from fusion.factory import build_model, variant_names
from train.trainer import build_loader
from eval.infer import run_inference
from eval.metrics_2d import evaluate_detections_2d
from eval.metrics_3d import evaluate_detections_3d
from eval.robustness import eval_blind


def latency(model, loader, cfg, device, warmup=None, iters=None):
    model.eval()
    warmup = warmup or cfg.get("benchmark", {}).get("warmup", 20)
    iters = iters or cfg.get("benchmark", {}).get("iters", 100)
    batch = next(iter(loader))
    batch["image"] = batch["image"].to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(batch)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def run_variant(name, base_cfg, device, val_length=16, iters=None):
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", f"{name}.yaml")
    cfg = load_config(cfg_path)
    if base_cfg.get("dataset"): cfg["dataset"] = base_cfg["dataset"]
    model = build_model(cfg).to(device)
    ckpt = os.path.join("checkpoints", f"{name}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    else:
        print(f"  [warn] no checkpoint for {name}; using untrained weights (smoke test)")
    loader = build_loader(cfg, "val" if cfg["dataset"] == "synthetic" else "train", length=val_length)
    space = model.output_space()
    dets, gts, _ = run_inference(model, loader, cfg, device)
    if space == "2d":
        ap = evaluate_detections_2d(dets, gts)["AP"]
    else:
        ap = evaluate_detections_3d(dets, gts)["AP_BEV"]
    p50, p95 = latency(model, loader, cfg, device, iters=iters)
    cam = eval_blind(model, loader, cfg, device, "cam")
    lid = eval_blind(model, loader, cfg, device, "lidar")
    return {"variant": name, "space": space, "AP": ap, "lat_p50": p50, "lat_p95": p95,
            "cam_dropped": cam, "lidar_dropped": lid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--val-length", type=int, default=16)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = {"dataset": args.dataset} if args.dataset else {}
    names = args.variants or variant_names()
    rows = []
    for name in names:
        print(f"== {name} ==")
        rows.append(run_variant(name, base, device, args.val_length, args.iters))
    cols = ["variant", "space", "AP", "lat_p50", "lat_p95", "cam_dropped", "lidar_dropped"]
    print("\n" + "\t".join(cols))
    for r in rows:
        print("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("\nwrote benchmark_results.csv")


if __name__ == "__main__":
    main()