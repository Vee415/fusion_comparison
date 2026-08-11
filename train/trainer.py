"""One trainer for all variants. Picks the loss via model.output_space().

Features:
- AMP (mixed precision) with gradient clipping (max_norm=10) to prevent nan instability
- Backbone freeze/unfreeze for pretrained-backbone warmup (freeze_backbone_epochs)
- Train/val split (held-out fraction) for honest AP tracking
- Best-checkpoint saving (keeps the weights with the lowest val loss, not just the last epoch)
- Early stopping (stops if val loss hasn't improved for `patience` epochs)
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from common.config import load_config, set_seed
from data.loaders.collate import collate_paired
from data.loaders.synthetic_loader import SyntheticPairedDataset
from data.loaders.paired_loader import KittiPairedDataset
from fusion.factory import build_model
from train.losses_2d import compute_2d_loss
from train.losses_3d import compute_3d_loss


def build_loader(cfg, split="train", length=None, val_fraction=0.0):
    """Build a DataLoader. If val_fraction > 0, splits the dataset into train/val subsets."""
    if cfg["dataset"] == "synthetic":
        ds = SyntheticPairedDataset(cfg, split, length or (64 if split == "train" else 16))
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"),
                          collate_fn=collate_paired, num_workers=cfg.get("workers", 2),
                          drop_last=False)
    ds = KittiPairedDataset(cfg, split)
    if val_fraction > 0 and split == "train":
        n = len(ds)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        perm = torch.randperm(n, generator=gen).tolist()
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                   collate_fn=collate_paired,
                                   num_workers=cfg.get("workers", 2), drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                 collate_fn=collate_paired,
                                 num_workers=cfg.get("workers", 2), drop_last=False)
        return train_loader, val_loader
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"),
                      collate_fn=collate_paired, num_workers=cfg.get("workers", 2),
                      drop_last=False)


def compute_loss(pred, target, space):
    return compute_2d_loss(pred, target) if space == "2d" else compute_3d_loss(pred, target)


def move_batch(batch, device):
    batch["image"] = batch["image"].to(device)
    # points / calib / boxes are accessed per-frame in the model/build_target; move there
    return batch


@torch.no_grad()
def validate(model, val_loader, cfg, device, space):
    """Compute average val loss over the val subset. Returns (avg_loss, n_batches).
    Returns inf if any batch is nan — so nan never wins the 'best checkpoint' race."""
    model.eval()
    total_loss, n = 0.0, 0
    for batch in val_loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device.split(":")[0], enabled=cfg.get("amp", True)):
            pred = model(batch)
            target = model.build_target(batch, cfg, device)
            if getattr(model, "custom_loss", False):
                loss = model.loss(pred, target)
            else:
                loss = compute_loss(pred, target, space)["loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            model.train()
            return (float("inf"), n)   # nan = worst possible, never save as best
        total_loss += loss.item()
        n += 1
    model.train()
    return (total_loss / max(n, 1), n)


def train(cfg, iters_per_epoch=None, max_steps=None, ckpt_dir="checkpoints"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)

    # ---- train/val split ----
    val_fraction = cfg.get("val_fraction", 0.1)  # hold out 10% for val by default
    if cfg["dataset"] == "synthetic":
        loader = build_loader(cfg, "train")
        val_loader = None
    else:
        result = build_loader(cfg, "train", val_fraction=val_fraction)
        if isinstance(result, tuple):
            loader, val_loader = result
        else:
            loader, val_loader = result, None
    space = model.output_space()

    # ---- backbone freeze/unfreeze ----
    freeze_epochs = cfg.get("freeze_backbone_epochs", 0)
    backbone_mods = []
    for name, mod in model.named_modules():
        if name.startswith(("backbone.stem", "backbone.layer", "image_backbone.stem",
                            "image_backbone.layer", "cam_backbone.stem", "cam_backbone.layer",
                            "lid_backbone.stem", "lid_backbone.layer")):
            backbone_mods.append((name, mod))

    def set_backbone_frozen(frozen):
        for _, mod in backbone_mods:
            for p in mod.parameters():
                p.requires_grad = not frozen
        if frozen and backbone_mods:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  backbone frozen ({len(backbone_mods)} modules, {trainable} trainable params)")
        elif backbone_mods:
            print(f"  backbone unfrozen for joint fine-tuning")

    if freeze_epochs > 0:
        set_backbone_frozen(True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * max(1, len(loader)))
    amp = cfg.get("amp", True) and device.startswith("cuda")
    scaler = torch.amp.GradScaler(device.split(":")[0]) if amp else None

    # ---- early stopping + best checkpoint ----
    patience = cfg.get("early_stop_patience", 5)   # stop after N epochs without val improvement
    min_delta = cfg.get("early_stop_min_delta", 1e-4)
    best_val_loss = float("inf")
    epochs_without_improve = 0
    best_path = os.path.join(ckpt_dir, f"{cfg['variant_name']}_best.pt")
    final_path = os.path.join(ckpt_dir, f"{cfg['variant_name']}.pt")

    os.makedirs(ckpt_dir, exist_ok=True)
    if val_loader:
        print(f"  train: {len(loader)} batches/epoch, val: {len(val_loader)} batches/epoch "
              f"(val_fraction={val_fraction}, patience={patience})")
    else:
        print(f"  train: {len(loader)} batches/epoch (no val split)")

    step = 0
    for epoch in range(cfg["epochs"]):
        # Unfreeze backbone after the warmup epochs
        if freeze_epochs > 0 and epoch == freeze_epochs:
            set_backbone_frozen(False)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=(cfg["epochs"] - freeze_epochs) * max(1, len(loader)))
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
            # Skip the optimizer step on nan loss — don't corrupt the weights
            if torch.isnan(loss) or torch.isinf(loss):
                if i % 10 == 0:
                    print(f"[{cfg['variant_name']}] epoch {epoch} step {i} loss nan (skipped)")
                step += 1
                continue
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()
            if i % 10 == 0:
                print(f"[{cfg['variant_name']}] epoch {epoch} step {i} loss {loss.item():.4f}")
            step += 1
            if max_steps is not None and step >= max_steps:
                break
            if iters_per_epoch is not None and i + 1 >= iters_per_epoch:
                break
        sched.step()

        # ---- validation + early stopping + best checkpoint ----
        if val_loader is not None:
            val_loss, n_val = validate(model, val_loader, cfg, device, space)
            improved = val_loss < best_val_loss - min_delta
            marker = "  *best*" if improved else ""
            print(f"  epoch {epoch}: val_loss={val_loss:.4f} ({n_val} batches){marker}")
            if improved:
                best_val_loss = val_loss
                epochs_without_improve = 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                            "val_loss": val_loss}, best_path)
                print(f"  saved best checkpoint -> {best_path}")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"  early stopping: {epochs_without_improve} epochs without val improvement")
                    break

        if max_steps is not None and step >= max_steps:
            break

    # save final checkpoint (last epoch, regardless of best)
    torch.save({"model": model.state_dict(), "cfg": cfg}, final_path)
    print(f"saved {final_path}")
    if val_loader is not None and os.path.exists(best_path):
        print(f"best checkpoint at {best_path} (val_loss={best_val_loss:.4f})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="override dataset: kitti | synthetic")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None, help="limit steps/epoch (smoke test)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None, help="override val split fraction")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset: cfg["dataset"] = args.dataset
    if args.epochs: cfg["epochs"] = args.epochs
    if args.val_fraction is not None: cfg["val_fraction"] = args.val_fraction
    train(cfg, iters_per_epoch=args.iters, max_steps=args.max_steps)


if __name__ == "__main__":
    main()