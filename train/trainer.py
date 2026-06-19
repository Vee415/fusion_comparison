"""One trainer for all variants. Picks the loss via model.output_space()."""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from common.config import load_config, set_seed
from data.loaders.collate import collate_paired
from data.loaders.synthetic_loader import SyntheticPairedDataset
from data.loaders.paired_loader import KittiPairedDataset
from fusion.factory import build_model
from train.losses_2d import compute_2d_loss
from train.losses_3d import compute_3d_loss


def build_loader(cfg, split="train", length=None):
    if cfg["dataset"] == "synthetic":
        ds = SyntheticPairedDataset(cfg, split, length or (64 if split == "train" else 16))
    else:
        ds = KittiPairedDataset(cfg, split)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"),
                      collate_fn=collate_paired, num_workers=cfg.get("workers", 2),
                      drop_last=False)


def compute_loss(pred, target, space):
    return compute_2d_loss(pred, target) if space == "2d" else compute_3d_loss(pred, target)


def move_batch(batch, device):
    batch["image"] = batch["image"].to(device)
    # points / calib / boxes are accessed per-frame in the model/build_target; move there
    return batch


def train(cfg, iters_per_epoch=None, max_steps=None, ckpt_dir="checkpoints"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    loader = build_loader(cfg, "train")
    space = model.output_space()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * max(1, len(loader)))
    amp = cfg.get("amp", True) and device.startswith("cuda")
    scaler = torch.amp.GradScaler(device.split(":")[0]) if amp else None

    os.makedirs(ckpt_dir, exist_ok=True)
    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for i, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.amp.autocast(device.split(":")[0], enabled=amp):
                pred = model(batch)
                target = model.build_target(batch, cfg, device)
                if getattr(model, "custom_loss", False):
                    loss = model.loss(pred, target)
                else:
                    loss = compute_loss(pred, target, space)["loss"]
            opt.zero_grad()
            if amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            if i % 10 == 0:
                print(f"[{cfg['variant_name']}] epoch {epoch} step {i} loss {loss.item():.4f}")
            step += 1
            if max_steps is not None and step >= max_steps:
                break
            if iters_per_epoch is not None and i + 1 >= iters_per_epoch:
                break
        sched.step()
        if max_steps is not None and step >= max_steps:
            break
    path = os.path.join(ckpt_dir, f"{cfg['variant_name']}.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    print(f"saved {path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="override dataset: kitti | synthetic")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None, help="limit steps/epoch (smoke test)")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset: cfg["dataset"] = args.dataset
    if args.epochs: cfg["epochs"] = args.epochs
    train(cfg, iters_per_epoch=args.iters, max_steps=args.max_steps)


if __name__ == "__main__":
    main()